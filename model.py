import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.batch_norm = nn.BatchNorm1d(2048)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.batch_norm(features)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    
    def __init__(self, embed_size=256, hidden_size=512, vocab_size=9955, num_layers=1):
        super().__init__()
        #features=256
        self.embed_size=embed_size
        self.hidden_size=hidden_size
        self.vocab_size=vocab_size
        self.num_layers=num_layers
        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embed_size)

        
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first=True)
        #self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        
#         self.hidden_state=(torch.zeros( 1, 10, self.hidden_size),
#                 torch.zeros( 1, 10, self.hidden_size))
    def forward(self, features, captions):
        #print(features.shape)
        #Take out last word "<end>" 
        captions=captions[:,:-1]
        captions=self.word_embeddings(captions)
        features = features.unsqueeze(1)
        join=torch.cat((features, captions), 1)
        x, self.hidden_state = self.lstm(join)#, self.hidden_state
        x = self.linear(x)
        return x

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        #h=self.init_hidden(1)
        sentence=[]
        word=0
        #x=inputs
        while word!=1:
            inputs, states=self.lstm(inputs,states)
            x = self.linear(inputs)
            p, w = x.max(2)
            #print(p,w)
            word=w.item()
            #x=torch.argmax(x).item()
            inputs=self.word_embeddings(w)
            sentence.append(word)
            if len(sentence)>= max_len:break
        return sentence

