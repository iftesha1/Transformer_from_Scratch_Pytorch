import math
import numpy as np
from bpemb import BPEmb
import torch
from torch import nn
import torch.nn.functional as F

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")

def scaled_dot_product_attention(query, key, value, mask=None):
    key_dim = float(key.shape[-1])
    scaled_scores = torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(key_dim)

    if mask is not None:
        scaled_scores.masked_fill_(mask==0, float('-inf'))

    weights = F.softmax(scaled_scores, dim=-1) 
    return torch.matmul(weights, value), weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = self.d_model // self.num_heads

        self.wq = nn.Linear(self.d_model, self.d_model).to(device)
        self.wk = nn.Linear(self.d_model, self.d_model).to(device)
        self.wv = nn.Linear(self.d_model, self.d_model).to(device)

        self.dense = nn.Linear(self.d_model, self.d_model).to(device)
    
    def split_heads(self, x):
        batch_size = x.size(0)
        split_inputs = x.view(batch_size, -1, self.num_heads, self.d_head)
        return split_inputs.transpose(1, 2)
    
    def merge_heads(self, x):
        batch_size = x.size(0)
        merged_inputs = x.transpose(1, 2).contiguous()
        return merged_inputs.view(batch_size, -1, self.d_model)

    def forward(self, q, k, v, mask):
        mask = mask.to(device)
        qs = self.wq(q)
        ks = self.wk(k)
        vs = self.wv(v)

        qs = self.split_heads(qs)
        ks = self.split_heads(ks)
        vs = self.split_heads(vs)

        output, attn_weights = scaled_dot_product_attention(qs, ks, vs, mask)
        output = self.merge_heads(output).to(device)
        output = self.dense(output)

        return output, attn_weights
    
def feed_forward_network(d_model, hidden_dim):

    return nn.Sequential(
        nn.Linear(d_model, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, d_model)
    )


class EncoderBlock(nn.Module):
  def __init__(self, d_model, num_heads, hidden_dim, dropout_rate=0.1):
    super(EncoderBlock, self).__init__()

    self.mhsa = MultiHeadSelfAttention(d_model, num_heads)
    self.ffn = feed_forward_network(d_model, hidden_dim)

    self.dropout1 = nn.Dropout(dropout_rate)
    self.dropout2 = nn.Dropout(dropout_rate)

    self.layernorm1 = nn.LayerNorm(d_model)
    self.layernorm2 = nn.LayerNorm(d_model)
  
  def call(self, x, training, mask):
    mhsa_output, attn_weights = self.mhsa(x, x, x, mask)
    mhsa_output = self.dropout1(mhsa_output, training=training)
    mhsa_output = self.layernorm1(x + mhsa_output)

    ffn_output = self.ffn(mhsa_output)
    ffn_output = self.dropout2(ffn_output, training=training)
    output = self.layernorm2(mhsa_output + ffn_output)

    return output, attn_weights
  
class Encoder(nn.Module):
  def __init__(self, num_blocks, d_model, num_heads, hidden_dim, src_vocab_size,
               max_seq_len, dropout_rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.max_seq_len = max_seq_len

    self.token_embed = nn.Embedding(src_vocab_size, self.d_model)
    self.pos_embed = nn.Embedding(max_seq_len, self.d_model)


    self.dropout = nn.Dropout(dropout_rate)

    # Create encoder blocks.
    self.blocks = [EncoderBlock(self.d_model, num_heads, hidden_dim, dropout_rate) 
    for _ in range(num_blocks)]
  
  def call(self, input, training, mask):
    token_embeds = self.token_embed(input)

    # Generate position indices for a batch of input sequences.
    num_pos = input.shape[0] * self.max_seq_len
    pos_idx = np.resize(np.arange(self.max_seq_len), num_pos)
    pos_idx = np.reshape(pos_idx, input.shape)
    pos_embeds = self.pos_embed(pos_idx)

    x = self.dropout(token_embeds + pos_embeds, training=training)

    # Run input through successive encoder blocks.
    for block in self.blocks:
      x, weights = block(x, training, mask)

    return x, weights
  

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout_rate=0.2):
        super(DecoderBlock, self).__init__()

        self.mhsa1 = MultiHeadSelfAttention(d_model, num_heads)
        self.mhsa2 = MultiHeadSelfAttention(d_model, num_heads)

        self.ffn = feed_forward_network(d_model, hidden_dim).to(device)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.layernorm1 = nn.LayerNorm(d_model).to(device)
        self.layernorm2 = nn.LayerNorm(d_model).to(device)
        self.layernorm3 = nn.LayerNorm(d_model).to(device)

    def forward(self, encoder_output, target, training, decoder_mask, memory_mask):
        mhsa_output1, attn_weights = self.mhsa1(target, target, target, decoder_mask)
        mhsa_output1 = self.dropout1(mhsa_output1)
        mhsa_output1 = self.layernorm1(mhsa_output1 + target)

        mhsa_output2, attn_weights = self.mhsa2(mhsa_output1, encoder_output, encoder_output, memory_mask)
        mhsa_output2 = self.dropout2(mhsa_output2)
        mhsa_output2 = self.layernorm2(mhsa_output2 + mhsa_output1)

        ffn_output = self.ffn(mhsa_output2)
        ffn_output = self.dropout3(ffn_output)
        output = self.layernorm3(ffn_output + mhsa_output2)

        return output, attn_weights

class Decoder(nn.Module):
    def __init__(self, num_blocks, d_model, num_heads, hidden_dim, target_vocab_size, max_seq_len, dropout_rate=0.2):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_embed = nn.Embedding(target_vocab_size, self.d_model).to(device)
        self.pos_embed = nn.Embedding(max_seq_len, self.d_model).to(device)
        self.dropout = nn.Dropout(dropout_rate)

        self.blocks = nn.ModuleList([DecoderBlock(self.d_model, num_heads, hidden_dim, dropout_rate) for _ in range(num_blocks)])

    def forward(self, encoder_output, target, training, decoder_mask, memory_mask):
        b, n = target.shape
        token_embeds = self.token_embed(target)
        pos_idx = torch.arange(n).unsqueeze(0).repeat(b, 1)  
        pos_idx = pos_idx.to(target.device)  
        pos_embeds = self.pos_embed(pos_idx)


        x = self.dropout(token_embeds + pos_embeds)

        for block in self.blocks:
            x, weights = block(encoder_output, x, training, decoder_mask, memory_mask)

        return x, weights


class Transformer(nn.Module):
    def __init__(self, num_blocks, d_model, num_heads, hidden_dim,source_vocab_size,
                   target_vocab_size, max_input_len,max_target_len, dropout_rate=0.2):
        super(Transformer, self).__init__()

        
        self.encoder = Encoder(num_blocks, d_model, num_heads, hidden_dim, source_vocab_size, 
                           max_input_len, dropout_rate)
       
        self.decoder = Decoder(num_blocks, d_model, num_heads, hidden_dim, target_vocab_size,
                               max_target_len, dropout_rate)

        # The final dense layer to generate logits from the decoder output.
        self.output_layer = nn.Linear(d_model, target_vocab_size).to(device)

    def encode(self,input_seqs,training,encoder_mask):
                
        encoder_output,encoder_attn_weights = self.encoder(input_seqs, 
                                                        training, encoder_mask)
        
        return encoder_output,encoder_attn_weights
    
    def decode(self,encoder_output,target_input_seqs,training,decoder_mask,memory_mask):
          
        decoder_output, decoder_attn_weights = self.decoder(encoder_output, 
                                                            target_input_seqs, training,
                                                            decoder_mask, memory_mask)
        output_logits = self.output_layer(decoder_output)

        return  decoder_output, output_logits,decoder_attn_weights


