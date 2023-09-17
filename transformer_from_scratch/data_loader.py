import pandas as pd
from sklearn.model_selection import train_test_split
from bpemb import BPEmb
import torch
import yaml
from torch.utils.data import Dataset, DataLoader

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def pad_sequences_post(sequences, pad_value=0):
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    padded_sequences = torch.full((len(sequences), max_len), pad_value)

    for i, seq in enumerate(sequences):
        seq_len = lengths[i]
        padded_sequences[i, :seq_len] = torch.tensor(seq)

    return padded_sequences

class CustomDataset(Dataset):
    def __init__(self, english_padded, french_inputs_padded, french_outputs_padded):
        self.english_padded = english_padded
        self.french_inputs_padded = french_inputs_padded
        self.french_outputs_padded = french_outputs_padded

    def __len__(self):
        return len(self.english_padded)

    def __getitem__(self, idx):
        return {
            'english_input': self.english_padded[idx],
            'french_input': self.french_inputs_padded[idx],
            'french_output': self.french_outputs_padded[idx]
        }

def get_data_loader(data:pd.DataFrame):
    df=data
    train_data, eval_data = train_test_split(df, test_size=0.2, random_state=42)
    bpemb_en = BPEmb(lang=config.src_lng)
    bpemb_fr = BPEmb(lang=config.tgt_lng)

    source_vocab_size, _ = bpemb_en.vectors.shape
    tgt_vocab_size,_ =bpemb_fr.vectors.shape


    # Tokenize English and French phrases
    df['English_tokens'] = df['Input'].apply(bpemb_en.encode_ids)
    df['French_tokens'] = df['Output'].apply(bpemb_fr.encode_ids)

    df['English_inputs'] = df['English_tokens'].apply(lambda x: [2] + x + [1])
    df['French_inputs'] = df['French_tokens'].apply(lambda x: [2] + x)
    df['French_outputs'] = df['French_tokens'].apply(lambda x: x + [1])

    english_padded = pad_sequences_post(df['English_inputs'].tolist())
    french_inputs_padded = pad_sequences_post(df['French_inputs'].tolist())
    french_outputs_padded = pad_sequences_post(df['French_outputs'].tolist())

    # Step 6: Save the padded data as lists in new DataFrame columns
    df['English_inputs'] = english_padded.tolist()
    df['French_inputs'] = french_inputs_padded.tolist()
    df['French_outputs'] = french_outputs_padded.tolist()

    english_padded = [torch.tensor(seq) for seq in df['English_inputs']]
    french_inputs_padded = [torch.tensor(seq) for seq in df['French_inputs']]
    french_outputs_padded = [torch.tensor(seq) for seq in df['French_outputs']]

    max_input_len = english_padded.shape[1],
    max_target_len = french_inputs_padded.shape[1]

    train_data, eval_data = train_test_split(
    list(zip(english_padded, french_inputs_padded, french_outputs_padded)), 
    test_size=0.2, 
    random_state=42)

    train_dataset = CustomDataset(*zip(*train_data))
    eval_dataset = CustomDataset(*zip(*eval_data))

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=2, shuffle=False)

    return train_dataloader,eval_dataloader,max_input_len,max_target_len, source_vocab_size, tgt_vocab_size


# trans = pd.read_csv("../transformer_from_scratch/english-french.csv")
# train,test=get_data_loader(trans)
# for i in train:
#     print(i)
#     break





