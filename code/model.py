import torch
import torch.nn as nn
import math
import numpy as np

# Set your parameters
vocab_size = 10000
d_model = 768
n_layers = 12
n_heads = 12
max_len = 70
batch_size = 64
d_ff = 2048  

class Embedding(nn.Module):
   def __init__(self, vocab_size, d_model, max_len, n_segments=2):
       super(Embedding, self).__init__()
       self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
       self.pos_embed = nn.Embedding(max_len, d_model)  # position embedding
       self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
       self.norm = nn.LayerNorm(d_model)

   def forward(self, x, seg):
       seq_len = x.size(1)
       pos = torch.arange(seq_len, dtype=torch.long)
       pos = pos.unsqueeze(0).expand_as(x)  # (seq_len,) -> (batch_size, seq_len)
       embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
       return self.norm(embedding)
   
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
def get_attn_pad_mask(seq_q, seq_k):
   batch_size, len_q = seq_q.size()
   batch_size, len_k = seq_k.size()
   # eq(zero) is PAD token
   pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
   return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

class EncoderLayer(nn.Module):
    def __init__(self, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def forward(self, enc_inputs, enc_self_attn_mask):
       
       enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
       enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
       return enc_outputs, attn 
   
    
class MultiHeadAttention(nn.Module):
   def __init__(self):
       super(MultiHeadAttention, self).__init__()
       self.d_k = d_model // n_heads  # Define d_k based on d_model and n_heads
       self.d_v = d_model // n_heads
       self.W_Q = nn.Linear(d_model, self.d_k * n_heads)
       self.W_K = nn.Linear(d_model, self.d_k * n_heads)
       self.W_V = nn.Linear(d_model, self.d_k * n_heads)

   def forward(self, Q, K, V, attn_mask):
       
       # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
       residual, batch_size = Q, Q.size(0)
       # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
       q_s = self.W_Q(Q).view(batch_size, -1, n_heads,self.d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
       k_s = self.W_K(K).view(batch_size, -1, n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
       v_s = self.W_V(V).view(batch_size, -1, n_heads,self.d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

       attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

       # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
       context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
       context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
       output = nn.Linear(n_heads * d_v, d_model)(context)
      
       return nn.LayerNorm(d_model)(output + residual), attn # output: [batch_size x len_q x d_model]   
   
class ScaledDotProductAttention(nn.Module):
   def __init__(self):
       super(ScaledDotProductAttention, self).__init__()
       self.d_k = d_k 

   def forward(self, Q, K, V, attn_mask):
       scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
       scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
       attn = nn.Softmax(dim=-1)(scores)
       context = torch.matmul(attn, V)
       return score, context, attn
   
class BERT(nn.Module):
   def __init__(self, vocab_size, d_model, n_layers, n_heads, max_len, d_ff, dropout, n_segments=2):
       super(BERT, self).__init__()
       self.embedding = Embedding(vocab_size, d_model, max_len, n_segments)
       self.layers = nn.ModuleList([EncoderLayer(d_ff, dropout) for _ in range(n_layers)])


   def forward(self, input_ids, segment_ids):
       output = self.embedding(input_ids, segment_ids)
       enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
       for layer in self.layers:
           output, _ = layer(output, enc_self_attn_mask)

       return output  # Return the output of the last encoder layer

def gelu(x):
   return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))