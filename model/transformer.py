from torch import Tensor
import torch
import torch.nn as nn
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VisualPositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 512):
        super(VisualPositionalEncoding, self).__init__()
        pos_embedding = nn.Parameter(torch.randn(maxlen, emb_size))
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('visual_pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.visual_pos_embedding[:token_embedding.size(0), :])

class Multitask_transformer(nn.Module):
    def __init__(self,
                 disable_transformer,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 dim_feedforward: int,
                 dropout: float):
        super(Multitask_transformer, self).__init__()
        decoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead, dim_feedforward=dim_feedforward,
                                                         dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(decoder_layer, num_layers=num_decoder_layers)
        self.positional_encoding = VisualPositionalEncoding(emb_size, dropout=dropout)
        self.embedding1 = nn.Sequential(nn.Conv2d(1, 3, (3, 3), padding='same'), nn.ReLU(),
                                        nn.MaxPool2d(5, 5))
        self.embedding2 = nn.Sequential(nn.Conv2d(1, 5, (3, 3), padding='same'), nn.ReLU(),
                                        nn.MaxPool2d(5, 5))
        self.embedding3 = nn.Sequential(nn.Conv2d(1, 8, (3, 3), padding='same'), nn.ReLU(),
                                        nn.MaxPool2d(5, 5))
        self.readout = nn.Linear(emb_size, 1)
        self.disable_transformer = disable_transformer

    def forward(self, input_x): #bs, 512, 3, 30, 30
        bs = input_x.size()[0]
        src_len = input_x.size()[1]
        input = input_x.view(-1, 3, 30, 30)
        x1, x2, x3 = input[:, 0], input[:, 1], input[:, 2]
        x1_embed = self.embedding1(x1.unsqueeze(1))
        x2_embed = self.embedding2(x2.unsqueeze(1))
        x3_embed = self.embedding3(x3.unsqueeze(1)) # bs*512, 400
        x = torch.concatenate([x1_embed, x2_embed, x3_embed], dim=1) #1024, 16, 6, 6
        x = torch.flatten(x, 1, -1).view(bs, src_len, -1) #2, 512, 576
        if self.disable_transformer:
            x = self.positional_encoding(x)
            x = self.transformer_encoder(x) # batch, seq, feature
        x = self.readout(x).squeeze(-1)
        return x