import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging
import pdb
from einops import rearrange
import copy

LOGGER = logging.getLogger()

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s: [ %(message)s ]",
                            "%m %d %Y %I:%M:%s %p")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)


class Attention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, is_mask=False):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.is_mask = is_mask
        self.d_q = self.d_k = self.d_v = int(d_model/num_heads)

        self.linear_q = nn.Linear(self.d_model, self.d_q)
        self.linear_k = nn.Linear(self.d_model, self.d_k)
        self.linear_v = nn.Linear(self.d_model, self.d_v)
        self.linear = nn.Linear(self.num_heads*self.d_v, self.d_model)
    
    def forward(self, Q, K, V):
        Q = self.linear_q(Q)
        K = self.linear_k(K)
        V = self.linear_v(V)
        output = self.linear(self.multihead_attention(Q, K, V))
        
        return output

    
    def mask(self, input, seq_len):
        # masked = torch.triu(torch.ones(seq_len,seq_len),diagonal=1) * -1e9
        masked = rearrange(torch.triu(torch.ones(seq_len,seq_len)), 'h w -> w h')
        # masked_fill = torch.triu(torch.ones(3,3),diagonal=1)*-1e9
        
        # if torch.cuda.is_available():
        #     masked = masked.cuda()
        # masked_input = input * masked + masked_fill
        
        masked_input = input.masked_fill(masked == 0, -1e9)
        return masked_input

    
    # Scaled Dot Product Attention
    def attention(self, Q, K, V):
        # Scaled Dot Product
        x = torch.bmm(Q, rearrange(K, 'b h w -> b w h'))
        x = x/math.sqrt(self.d_k)

        # Mask(Optional)
        if self.is_mask:
            seq_len = Q.size()[-2]
            self.mask(x, seq_len)
        
        # SoftMax
        x = F.softmax(x, dim=-1)

        # MatMul
        x = torch.bmm(x, V)

        return x

    def multihead_attention(self, Q, K, V):
        mh_attentions = []
        for _ in range(self.num_heads):
            mh_attentions.append(self.attention(Q, K, V))
        concatenated = torch.cat(mh_attentions, dim=2) # TODO: dimension: 
        return concatenated

class Encoder(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=.1, d_ff=2048):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.attention = Attention(d_model=self.d_model, num_heads=self.num_heads)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_model, d_ff),
            nn.ReLU(), # max(0, x)
            nn.Linear(d_ff, self.d_model)
        )

    def forward(self, input):
        # Multi Head Self Attention
        x = Q = K = V = input

        ## Residual Dropout
        x = x + self.dropout(self.attention(Q, K, V))
        x = self.layer_norm(x)

        # ----------------------------------------
        # Feed Forward + Residual Dropout
        x = x + self.dropout(self.feed_forward(x))
        x = self.layer_norm(x)

        return x
                
class Decoder(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=.1, d_ff=2048):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.attention = Attention(d_model=self.d_model, num_heads=self.num_heads)
        self.masked_attention = Attention(d_model=self.d_model, num_heads=self.num_heads, is_mask=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_model, d_ff),
            nn.ReLU(), # max(0, x)
            nn.Linear(d_ff, self.d_model)
        )

    def forward(self, input, output):
        # Masked Multi Head Self Attention
        x = Q = K = V = output

        ## Residual Dropout
        x = x + self.dropout(self.masked_attention(Q, K, V))
        x = self.layer_norm(x)

        # ----------------------------------------
        # Multi Head Cross Attention
        Q = x
        K = V = input

        ## Residual Dropout
        x = x + self.dropout(self.attention(Q, K, V))
        x = self.layer_norm(x)

        # ----------------------------------------
        # Feed Forward + Residual Dropout
        x = x + self.dropout(self.feed_forward(x))
        x = self.layer_norm(x)

        return x

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size=16000, d_output=512):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim = d_output,
            padding_idx = 0
        )
    
    def forward(self, input):
        x = self.embed(input)

        _, seq_len, d_model = x.size()
        pos_encoding = self.positional_encoding(seq_len, d_model)
        x += pos_encoding
        return x


    def positional_encoding(self, seq_len, d_model):
        pos = np.arange(seq_len)[:, np.newaxis]
        dim = np.arange(d_model)[np.newaxis, :]
        angles = pos / np.power(10000, 2*(dim//2)/d_model)

        pos_encoding = np.zeros(angles.shape)
        pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])

        pos_encoding = torch.FloatTensor(pos_encoding)
        if torch.cuda.is_available():
            pos_encoding = pos_encoding.cuda()
        return pos_encoding



class TransformerModel(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_encoder=6, num_decoder=6, input_vocab_size=16000, output_vocab_size=16000, dropout=.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder = num_encoder
        self.num_decoder = num_decoder
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size

        self.input_embedding = WordEmbedding(vocab_size=input_vocab_size)
        self.output_embedding = WordEmbedding(vocab_size=output_vocab_size)
        self.encoders = nn.ModuleList([Encoder(d_model=self.d_model, num_heads=self.num_heads) for _ in range(self.num_encoder) ])
        self.decoders = nn.ModuleList([Decoder(d_model=self.d_model, num_heads=self.num_heads) for _ in range(self.num_encoder) ])

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.d_model, self.output_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        

    def forward(self, input, output):
        encoded_input = self.encode(input)
        decoded_output = self.decode(encoded_input, output)
        output_probabilities = self.prob(decoded_output)

        return output_probabilities

    def encode(self, input):
        encoded_input = self.input_embedding(input)
        encoded_input = self.dropout(encoded_input)

        for encoder in self.encoders:
            encoded_input = encoder(encoded_input)
        return encoded_input

    def decode(self, encoded_input, output):
        encoded_output = self.output_embedding(output)
        encoded_output = self.dropout(encoded_output)

        decoded_output = encoded_output
        for decoder in self.decoders:
            decoded_output = decoder(encoded_input, decoded_output)
        
        return decoded_output

    def prob(self, decoded_output):
        x = self.linear(decoded_output)
        output_prob = self.softmax(x)

        return output_prob

def test():
    init_logging()

    device = "cuda: 0" if torch.cuda.is_available() else "cpu"
    torch.device(device)   

    inputs = torch.LongTensor([[1,2,3,4]])
    outputs = torch.LongTensor([[1,2,3,4]])

    if device != "cpu":
        inputs = inputs.cuda()
        outputs = outputs.cuda()

    model = TransformerModel(d_model=512, num_heads=8, num_encoder=6, num_decoder=6, input_vocab_size=16000, output_vocab_size=16000, dropout=0.1)
    output_prob = model(inputs, outputs)
    print(output_prob.size())

if __name__=="__main__":
    test()