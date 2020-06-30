import torch
from torch import nn
from torch.nn import functional as F

class Lstm_1_in_2_out(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, ls_dim, 
                 output_1_dim:int, output_2_dim:int):
        super().__init__()        
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.ls_dim = ls_dim
        self.output_1_dim = output_1_dim
        self.output_2_dim = output_2_dim
        
        self.word_embeddings = nn.Embedding(self.vocab_size, self.emb_dim)
        
        self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_dim) # not batch first
        
        self.hidden2out = nn.Linear(self.hidden_dim, self.ls_dim)
        
        self.fc1 = nn.Linear(self.ls_dim, self.output_1_dim)
        self.fc2 = nn.Linear(self.ls_dim, self.output_2_dim)
        
    def forward(self, sent):
        embeds = self.word_embeddings(sent)
        lstm_out, (final_hidden, c) = self.lstm(embeds)
        x = self.hidden2out(final_hidden)
        out1 = self.fc1(x).squeeze(0)
        out2 = self.fc2(x).squeeze(0)
        return out1, out2

    
class Lstm_1_in_1_out(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, ls_dim, output_dim:int):
        super().__init__()        
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.ls_dim = ls_dim
        self.output_dim = output_dim
        
        self.word_embeddings = nn.Embedding(self.vocab_size, self.emb_dim)
        
        self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_dim) # not batch first
        
        self.hidden2out = nn.Linear(self.hidden_dim, self.ls_dim)
        
        self.fc1 = nn.Linear(self.ls_dim, self.output_dim)
        
    def forward(self, sent):
        embeds = self.word_embeddings(sent)
        lstm_out, (final_hidden, c) = self.lstm(embeds)
        x = self.hidden2out(final_hidden)
        out = self.fc1(x).squeeze(0)
        return out
    

class Lstm_2_in_1_out(nn.Module):
    def __init__(self, vocab_size, num_pos, emb_dim, pos_dim, 
                 hidden_dim, ls_dim, output_dim:int):
        super().__init__()        
        self.vocab_size = vocab_size
        self.num_pos = num_pos
        self.emb_dim = emb_dim
        self.pos_dim = pos_dim
        self.tot_dim = emb_dim + pos_dim
        self.hidden_dim = hidden_dim
        self.ls_dim = ls_dim
        self.output_dim = output_dim
        
        self.word_embeddings = nn.Embedding(self.vocab_size, self.emb_dim)
        self.pos_embeddings = nn.Embedding(self.num_pos, self.pos_dim)
        
        self.lstm = nn.LSTM(input_size=self.tot_dim, hidden_size=self.hidden_dim) # not batch first
        
        self.hidden2out = nn.Linear(self.hidden_dim, self.ls_dim)
        
        self.fc1 = nn.Linear(self.ls_dim, self.output_dim)
        
    def forward(self, sent, pos):
        words = self.word_embeddings(sent)
        tags = self.pos_embeddings(pos)
        embeds = torch.cat((words, tags), 2)
        lstm_out, (final_hidden, c) = self.lstm(embeds)
        x = self.hidden2out(final_hidden)
        out = self.fc1(x).squeeze(0)
        return out
