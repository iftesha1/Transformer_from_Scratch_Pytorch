from model import Transformer
import yaml
import mlflow
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from data_loader import get_data_loader
from bpemb import BPEmb


exp_id = mlflow.set_experiment("Translation-1")
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data=pd.read_csv("../transformer_from_scratch/english-french.csv")
train_dataloader,eval_dataloader,max_input_len,max_target_len,source_vocab_size, tgt_vocab_size= get_data_loader(data)

model = Transformer(
    num_blocks = config.num_hidden_layers,
    d_model = config.hidden_size,
    num_heads = config.num_attention_heads,
    hidden_dim = config.intermediate_size,
    target_vocab_size = tgt_vocab_size,
    max_target_len = max_target_len)

def decode_batch(token_id_sequences, lang_bpemb):
    decoded_sentences = []
    for token_ids in token_id_sequences:
        # Convert tensor to list of IDs and remove SOS and EOS tokens
        token_ids = token_ids.tolist()[1:-1]
        # Decode the list of token IDs to a sentence
        decoded_sentence = lang_bpemb.decode_ids(token_ids)
        decoded_sentences.append(decoded_sentence)
    return decoded_sentences



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr =config.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

def compute_accuracy(output_logits, target):
    _, predicted_classes = torch.max(output_logits, dim=-1)
    correct_predictions = (predicted_classes == target).float()
    accuracy = correct_predictions.sum() / correct_predictions.numel()
    return accuracy.item()

def encoder_mask(input_seqs):
    enc_mask = (input_seqs.ne(0)).float()
    enc_mask = enc_mask.unsqueeze(1).unsqueeze(2).to(device)
    return enc_mask

def decoder_mask(target_input_seqs):
    dec_padding_mask = (target_input_seqs != 0).float()
    dec_padding_mask = dec_padding_mask.unsqueeze(1).unsqueeze(1)

    target_input_seq_len = target_input_seqs.shape[1]
    look_ahead_mask = torch.ones((target_input_seq_len, target_input_seq_len)).tril().to(device)

    dec_mask = torch.min(dec_padding_mask, look_ahead_mask).to(device)
    return dec_mask

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size).to(device)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)

def greedy_decode(model, source, max_len):

    source_mask=encoder_mask(source)   
    encoder_output,encoder_attn_w = model.encode(source)

    decoder_input = torch.empty(1, 1).fill_(2).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        # calculate output
        decoder_out, out_logits = model.decode(encoder_output, decoder_input,False, decoder_mask,source_mask)        
        proj=ProjectionLayer(d_model=config.hidden_size,target_vocab_size=config.target_vocab_size)
        
        # get next token
        prob = proj(decoder_out[:, -1])
        _, next_word = torch.max(prob, dim=1)                                    
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )
        if next_word == 1:
            break            
        output = decoder_input.squeeze(0)
    
        # Check size of output, and pad if necessary
        if output.size(0) < max_len:
            pad_size = max_len - output.size(0)

            # Pad output
            final_output = F.pad(output, (0, pad_size))

    return decoder_out, final_output, out_logits

def run_validation(model, validation_ds, max_len):
    model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []
    test_accuracy=0

    print("Starting Validation")
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['english_input']
            target_seqs = batch['french_input']
            labels=batch['french_output']
            
            print("target-seq:",target_seqs)
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"
            model_out,final_output,out_logits = greedy_decode(model, encoder_input, max_len)

            accuracy = compute_accuracy(out_logits, labels)
            test_accuracy+=accuracy


            bpemb_fr = BPEmb(lang='fr')
            predicted_sen = decode_batch(model_out, bpemb_fr)

            source_texts.append(encoder_input)
            expected.append(labels)
            predicted.append(predicted_sen)
    
    avg_test_acc=test_accuracy/count
    return source_texts, expected, predicted,avg_test_acc



with mlflow.start_run():
    mlflow.log_params({"lr":config.lr , "train_batch_size": config.train_batch_size, "epochs": config.num_epochs}) 
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    for epoch in tqdm(range(config.num_epochs), desc=f"Epochs", leave=False):
        # print(f"Epoch {epoch+1}/{config.num_epochs}")
        for batch in (train_dataloader):
            input_seqs = batch['english_input']
            target_input_seqs = batch['french_input']
            labels=batch['french_output']

            enc_mask=encoder_mask(input_seqs)
            dec_mask = decoder_mask(target_input_seqs)
            optimizer.zero_grad()

            # Forward pass
            encoder_output, encoder_attn_w = model.encode(input_seqs)
            dec_out, output_logits,decoder_attn_w = model.decode(encoder_output, 
                                                            target_input_seqs, True,dec_mask,
                                                             enc_mask)
            
            # Compute the loss
            loss = criterion(output_logits.view(-1, output_logits.size(-1)), labels.view(-1))

            # Compute the accuracy
            accuracy = compute_accuracy(output_logits, target_input_seqs)

            # Backward pass 
            loss.backward()
            optimizer.step()


        source_texts, expected, predicted,avg_test_acc=run_validation(model, eval_dataloader, config.max_len)   
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches 
    
        print(f"Loss:{loss.item()}, \nAvg Loss: {avg_loss}, Avg Accuracy: {avg_accuracy},Test Accuracy:{avg_test_acc}") 
        mlflow.log_metrics({"Loss":loss,"Avg_loss": avg_loss,
                            "Avg_accuracy": avg_accuracy,"Test Accuracy":avg_test_acc}, epoch)  
        # Step the scheduler
        scheduler.step()
