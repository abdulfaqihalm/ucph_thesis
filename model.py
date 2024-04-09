from torch import nn
import torch
import numpy as np
from wrapper.model_utils import BahdanauAttention, EmbeddingSeq
import math

class NaiveModelV1(nn.Module):
    ## Naive model from MultiRM Paper by Song et al.
    def __init__(self, input_size=None) -> None:
        super().__init__()
        self.NaiveCNN = nn.Sequential(
            nn.Conv1d(in_channels=4,out_channels=8,kernel_size=7,stride=2,padding=0), #[bs, 8, 48]
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=8,out_channels=32,kernel_size=3,stride=1,padding=1),#[bs 32 48]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,padding=0),                                    #[bs 32 24]
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=32,out_channels=128,kernel_size=3,stride=1,padding=1),#[bs 128 24]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,padding=0) #[bs 128 12]
            )
        in_features_1 = (input_size - 7) // 2 + 1   # 48
        in_features_2 = (in_features_1 - 2) // 2 + 1    # 24
        in_features_3 = (in_features_2 - 2) // 2 + 1    # 12
        self.Flatten = nn.Flatten() # 128*12
        self.FC1= nn.Sequential(nn.Linear(in_features=128*in_features_3,out_features=1024),
                                      nn.ReLU(),
                                      nn.Dropout()
                                      )
        self.FC2 = nn.Sequential(nn.Linear(in_features=1024,out_features=256),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(in_features=256,out_features=64),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(in_features=64,out_features=1),
                                nn.Hardtanh(min_val=0, max_val=1)
                                )


    # Define the model    
    def forward(self, x) -> None:
        out = self.NaiveCNN(x)
        out = self.Flatten(out) # flatten output
        out = self.FC1(out)
        out = self.FC2(out)
        return out


class NaiveModelV2WoBatchNormDropOut(nn.Module):
    ## CNN + LSTM
    def __init__(self, input_channel=None, input_size = 1001, cnn_first_filter=8, cnn_first_kernel_size=7, output_dim=1) -> None:
        super().__init__()
        self.NaiveCNN = nn.Sequential(
            nn.Conv1d(in_channels=input_channel,out_channels=cnn_first_filter,kernel_size=cnn_first_kernel_size,stride=2,padding=0), #[bs, 8, 498]
            nn.ReLU(),
            nn.Conv1d(in_channels=8,out_channels=32,kernel_size=3,stride=1,padding=1),#[bs 32 498]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,padding=0),                                    #[bs 32 249]
            nn.Conv1d(in_channels=32,out_channels=128,kernel_size=3,stride=1,padding=1),#[bs 128 249]
            nn.ReLU(),
             #nn.MaxPool1d(kernel_size=2,padding=0)
            )
        in_size = ((input_size - (1 * (cnn_first_kernel_size - 1 )) - 1)/2) + 1
        in_size = ((in_size + (2*1) - (1 * (3 - 1 )) - 1)/1) + 1
        in_size = in_size/2
        in_size = int(((in_size + (2*1) - (1 * (3 - 1 )) - 1)/1) + 1)
        self.biLSTM = nn.LSTM(input_size=128,hidden_size=128,batch_first=True,bidirectional=True, num_layers=3)
        self.Flatten = nn.Flatten() # 128*12*2 -> biLSTM
        self.FC1= nn.Sequential(nn.Linear(in_features=2*128*in_size,out_features=1024),
                                nn.ReLU(),
                                nn.Linear(in_features=1024,out_features=512),
                                nn.ReLU(),
                                )
        self.FC2 = nn.Sequential(nn.Linear(in_features=512,out_features=256),
                                nn.ReLU(),
                                nn.Linear(in_features=256,out_features=64),
                                nn.ReLU(),
                                nn.Linear(in_features=64,out_features=output_dim),
                                nn.Hardtanh(0, 1)
                                )
class NaiveModelV2(nn.Module):
    ## CNN + LSTM
    def __init__(self, input_channel=None, input_size = 1001, cnn_first_filter=8, cnn_first_kernel_size=7, output_dim=1) -> None:
        super().__init__()
        self.NaiveCNN = nn.Sequential(
            nn.Conv1d(in_channels=input_channel,out_channels=cnn_first_filter,kernel_size=cnn_first_kernel_size,stride=2,padding=0), #[bs, 8, 498]
            nn.BatchNorm1d(num_features=8),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=8,out_channels=32,kernel_size=3,stride=1,padding=1),#[bs 32 498]
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,padding=0),                                    #[bs 32 249]
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=32,out_channels=128,kernel_size=3,stride=1,padding=1),#[bs 128 249]
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
             #nn.MaxPool1d(kernel_size=2,padding=0)
            )
        in_size = ((input_size - (1 * (cnn_first_kernel_size - 1 )) - 1)/2) + 1
        in_size = ((in_size + (2*1) - (1 * (3 - 1 )) - 1)/1) + 1
        in_size = in_size/2
        in_size = int(((in_size + (2*1) - (1 * (3 - 1 )) - 1)/1) + 1)
        self.biLSTM = nn.LSTM(input_size=128,hidden_size=128,batch_first=True,bidirectional=True, num_layers=3)
        self.Flatten = nn.Flatten() # 128*12*2 -> biLSTM
        self.FC1= nn.Sequential(nn.Linear(in_features=2*128*in_size,out_features=1024),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(in_features=1024,out_features=512),
                                nn.ReLU(),
                                nn.Dropout(p=0.2)
                                )
        self.FC2 = nn.Sequential(nn.Linear(in_features=512,out_features=256),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(in_features=256,out_features=64),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(in_features=64,out_features=output_dim),
                                nn.Hardtanh(0, 1)
                                )


    # Define the model    
    def forward(self, x) -> None:
        out = self.NaiveCNN(x)
        out = out.permute(0, 2, 1) #[bs 12 128]
        out, h = self.biLSTM(out)
        out = self.Flatten(out) # flatten output
        out = self.FC1(out)
        out = self.FC2(out)
        out = out
        return out
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=4, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ConvTransformerModel(nn.Module):
    def __init__(self, seq_size=1001, input_channel=4, dropout_rate=0.2, l2_param=0.001, conv_win=6, num_heads=4, transformer_units=[32, 16]):
        super(ConvTransformerModel, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channel, 32, kernel_size=conv_win, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 32, kernel_size=conv_win, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 32, kernel_size=conv_win, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=4),
        )

        self.flatten = nn.Flatten()

        self.positional_encoding = PositionalEncoding(dropout=dropout_rate)
        # Transformer Encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=4, nhead=num_heads, dim_feedforward=2048, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=len(transformer_units))
        self.flatten_trans = nn.Flatten()

        self.dense_layers = nn.Sequential(
            #nn.Linear(seq_size * (32 + input_channel), 32),  # Adjust input dimension 
            nn.Linear(5988, 32),
            # based on your final concatenated features size
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout_rate),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(dropout_rate),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Convolutional branch
        #x_conv = x.permute(0, 2, 1)  # Change to (batch, channels, length)
        x_conv = self.conv_layers(x)
        x_conv = self.flatten(x_conv)
        #print(f"x_conv {x_conv.shape}")

        # Transformer branch
  # Transformer branch, reshape x from [batch, channels, length] to [length, batch, channels] for transformer
        x_trans = x.permute(2, 0, 1)  # Reshape for transformer
        #print(f"x_trans {x_trans.shape}")
        x_trans = self.positional_encoding(x_trans)
        x_trans = self.transformer_encoder(x_trans)
       #print(f"x_trans transformer {x_trans.shape}")
        x_trans = x_trans.permute(1, 2, 0)  # Revert to (batch, features, seq_len) for Flatten

        #print(f"x_trans transformer permute {x_trans.shape}")
        x_trans = self.flatten_trans(x_trans)

        #print(f"x_trans transformer flatten {x_trans.shape}")


        # Concatenate convolutional and transformer branches
        x = torch.cat((x_conv, x_trans), dim=1)

        # Dense layers
        x = self.dense_layers(x)

        return x*100



class model_v3(nn.Module):

    def __init__(self,num_task,use_embedding):
        super(model_v3,self).__init__()

        self.num_task = num_task
        self.use_embedding = use_embedding
        if self.use_embedding:
            self.embed = EmbeddingSeq('data/embeddings/embeddings_12RM.pkl') # Word2Vec
            # self.embed = EmbeddingHmm(t=3,out_dims=300) # hmm
            self.NaiveBiLSTM = nn.LSTM(input_size=300,hidden_size=256,batch_first=True,bidirectional=True)
        else:
            self.NaiveBiLSTM = nn.LSTM(input_size=4,hidden_size=256,batch_first=True,bidirectional=True)

        self.Attention = BahdanauAttention(in_features=512,hidden_units=100,num_task=num_task)
        for i in range(num_task):
            setattr(self, "NaiveFC%d" %i, nn.Sequential(
                                       nn.Linear(in_features=512,out_features=128),
                                       nn.ReLU(),
                                       nn.Dropout(),
                                       nn.Linear(in_features=128,out_features=1),
                                       nn.Sigmoid()
                                                    ))

    def forward(self,x):

        if self.use_embedding:
            x = self.embed(x)
        else:
            x = torch.transpose(x,1,2)
        batch_size = x.size()[0]
        # x = torch.transpose(x,1,2)

        output,(h_n,c_n) = self.NaiveBiLSTM(x)
        h_n = h_n.view(batch_size,output.size()[-1])
        context_vector,attention_weights = self.Attention(h_n,output)
        # print(attention_weights.shape)
        outs = []
        for i in range(self.num_task):
            FClayer = getattr(self, "NaiveFC%d" %i)
            y = FClayer(context_vector[:,i,:])
            y = torch.squeeze(y, dim=-1)
            outs.append(y)

        return outs



class NaiveModelV3(nn.Module):
    ## CNN + LSTM
    def __init__(self, input_size=None) -> None:
        super().__init__()
        self.NaiveCNN = nn.Sequential(
            nn.Conv1d(in_channels=4,out_channels=8,kernel_size=11,stride=2,padding=0), #[bs, 8, 496]
            nn.BatchNorm1d(num_features=8),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=8,out_channels=64,kernel_size=3,stride=1,padding=1),#[bs 64 496]
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,padding=0),                                    #[bs 64 248]
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),#[bs 128 248]
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,padding=0), #[bs 128 124]
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),#[bs 128 124]
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,padding=0) #[bs 256 62]
            )
        self.biLSTM = nn.LSTM(input_size=256,hidden_size=256,batch_first=True,bidirectional=True, num_layers=2)
        cnn_features = 62
        self.Flatten = nn.Flatten() # 128*12*2 -> biLSTM
        self.FC1= nn.Sequential(nn.Linear(in_features=2*256*cnn_features,out_features=1024),
                                      nn.ReLU(),
                                      nn.Dropout()
                                      )
        self.FC2 = nn.Sequential(nn.Linear(in_features=1024,out_features=256),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(in_features=256,out_features=64),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(in_features=64,out_features=1),
                                nn.Hardtanh(0, 1)
                                )
    # Define the model    
    def forward(self, x) -> None:
        out = self.NaiveCNN(x)
        out = out.permute(0, 2, 1) #[bs 12 128]
        out, h = self.biLSTM(out)
        out = self.Flatten(out) # flatten output
        out = self.FC1(out)
        out = self.FC2(out)
        out = out
        return out


class NaiveModelV4(nn.Module):
    ## Naive model from MultiRM Paper by Song et al.
    def __init__(self, input_size=None) -> None:
        super().__init__()
        self.NaiveCNN = nn.Sequential(
                        nn.Conv1d(in_channels=4,out_channels=8,kernel_size=7,stride=2,padding=0),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Conv1d(in_channels=8,out_channels=32,kernel_size=3,stride=1,padding=1),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=2,padding=0),
                        nn.Dropout(p=0.2),
                        nn.Conv1d(in_channels=32,out_channels=128,kernel_size=3,stride=1,padding=1),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=2,padding=0)
                        )
        self.NaiveBiLSTM = nn.LSTM(input_size=128,hidden_size=128,batch_first=True,bidirectional=True)
        in_features_1 = (input_size - 7) // 2 + 1   # 48
        in_features_2 = (in_features_1 - 2) // 2 + 1    # 24
        in_features_3 = (in_features_2 - 2) // 2 + 1    # 12
        self.Flatten = nn.Flatten() # 128*12*2 -> biLSTM
        self.Attention = BahdanauAttention(in_features=256,hidden_units=10,num_task=1)
        self.FC1 = nn.Sequential(nn.Linear(in_features=256,out_features=128),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(in_features=128,out_features=64),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(in_features=64,out_features=1),
                                nn.Hardtanh(0, 1)
                                )
    # Define the model    
    def forward(self, x) -> None:
        x = self.NaiveCNN(x)
        batch_size, features, seq_len = x.size()
        x = x.view(batch_size,seq_len, features) # parepare input for LSTM
        output, (h_n, c_n) = self.NaiveBiLSTM(x)
        h_n = h_n.view(batch_size,output.size()[-1]) # pareprae input for Attention
        context_vector,attention_weights = self.Attention(h_n,output) # Attention (batch_size, num_task, unit)num_task, unit)
        out = self.Flatten(context_vector[:,0,:]) # flatten output
        out = self.FC1(out)
        out = out
        return out
    

class MultiRMModel(nn.Module):
    def __init__(self,num_task=1,use_embedding=False):
        super(MultiRMModel,self).__init__()

        self.num_task = num_task
        self.use_embedding = use_embedding
        if self.use_embedding:
            self.embed = EmbeddingSeq('data/embeddings/embeddings_12RM.pkl') # Word2Vec
            # self.embed = EmbeddingHmm(t=3,out_dims=300) # hmm
            self.NaiveBiLSTM = nn.LSTM(input_size=300,hidden_size=256,batch_first=True,bidirectional=True)
        else:
            self.NaiveBiLSTM = nn.LSTM(input_size=4,hidden_size=256,batch_first=True,bidirectional=True)

        self.Attention = BahdanauAttention(in_features=512,hidden_units=100,num_task=num_task)
        self.NaiveFC1 = nn.Sequential(
                                       nn.Linear(in_features=512,out_features=128),
                                       nn.ReLU(),
                                       nn.Dropout(),
                                       nn.Linear(in_features=128,out_features=1),
                                       nn.Sigmoid()
                                                    )

    def forward(self,x):
        x = torch.transpose(x,1,2)
        batch_size = x.size()[0]
        # x = torch.transpose(x,1,2)

        output,(h_n,c_n) = self.NaiveBiLSTM(x)
        h_n = h_n.view(batch_size,output.size()[-1])
        context_vector,attention_weights = self.Attention(h_n,output)
        # print(attention_weights.shape)
        out = self.NaiveFC1(context_vector[:,0,:])
        out = torch.squeeze(out*100, dim=-1)
        return out.unsqueeze(1) 
    
class ConfigurableModel(nn.Module):
    ## CNN + LSTM
    def __init__(self, input_channel=4, input_size = 1001, cnn_first_filter=8, cnn_first_kernel_size=7, cnn_length=3, 
                 cnn_other_filter=32, cnn_other_kernel_size=6, bilstm_layer=2, bilstm_hidden_size=128, fc_size=256, output_size=1) -> None:
        super().__init__()
        self.CNN = torch.nn.Sequential()

        seq_length = input_size
        for i in range(cnn_length):
            if i == 0:
                self.CNN.add_module(f"CNN_{i+1}", nn.Conv1d(in_channels=input_channel, out_channels=cnn_first_filter,kernel_size=cnn_first_kernel_size,stride=2,padding=0)
                )
                self.CNN.add_module(f"BatchNorm_{i+1}", nn.BatchNorm1d(num_features=cnn_first_filter))

                seq_length = ((input_size - (1 * (cnn_first_kernel_size - 1 )) - 1)/2) + 1
            elif i == 1:
                self.CNN.add_module(f"CNN_{i+1}", nn.Conv1d(in_channels=cnn_first_filter, out_channels=cnn_other_filter, kernel_size=cnn_other_kernel_size, stride=1, padding=1))
            else:
                self.CNN.add_module(f"CNN_{i+1}", nn.Conv1d(in_channels=cnn_other_filter, out_channels=cnn_other_filter, kernel_size=cnn_other_kernel_size, stride=1, padding=1))
                seq_length = ((seq_length + (2*1) - (1 * (cnn_other_kernel_size - 1 )) - 1)/1) + 1

            if i != 0:
                self.CNN.add_module(f"BATCHNORM_{i+1}", nn.BatchNorm1d(num_features=cnn_other_filter))

            self.CNN.add_module(f"RELU_{i+1}", torch.nn.ReLU())

            self.CNN.add_module(f"DROPOUT_{i+1}", torch.nn.Dropout())
        
        self.biLSTM = nn.LSTM(input_size=cnn_other_filter,hidden_size=bilstm_hidden_size,batch_first=True,bidirectional=True, num_layers=bilstm_layer)
        
        self.flatten = nn.Flatten()

        self.FC = nn.Sequential(
            # nn.Linear(in_features=int(2*bilstm_hidden_size*seq_length),out_features=fc_size),
            nn.LazyLinear(out_features=fc_size),
            # based on your final concatenated features size
            nn.ReLU(),
            nn.BatchNorm1d(fc_size),
            nn.Dropout(),
            nn.Linear(fc_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(),

            nn.Linear(64, output_size),
            nn.Sigmoid()

        )
    
    def forward(self, x) -> None:
        out = self.CNN(x) #[bs feature_dim seq_length]
        out = out.permute(0, 2, 1) #[bs seq_length feature_dim]
        out, h = self.biLSTM(out) #[bs seq_length feature_dim]
        out = self.flatten(out) 
        out = self.FC(out)
        out = out
        return out


    
class ConfigurableModelWoBatchNormDropout(nn.Module):
    ## CNN + LSTM
    def __init__(self, input_channel=4, input_size = 1001, cnn_first_filter=8, cnn_first_kernel_size=7, cnn_length=3, 
                 cnn_other_filter=32, cnn_other_kernel_size=6, bilstm_layer=2, bilstm_hidden_size=128, fc_size=256, output_size=1) -> None:
        super().__init__()
        self.CNN = torch.nn.Sequential()

        seq_length = input_size
        for i in range(cnn_length):
            if i == 0:
                self.CNN.add_module(f"CNN_{i+1}", nn.Conv1d(in_channels=input_channel, out_channels=cnn_first_filter,kernel_size=cnn_first_kernel_size,stride=2,padding=0)
                )

                seq_length = ((input_size - (1 * (cnn_first_kernel_size - 1 )) - 1)/2) + 1
            elif i == 1:
                self.CNN.add_module(f"CNN_{i+1}", nn.Conv1d(in_channels=cnn_first_filter, out_channels=cnn_other_filter, kernel_size=cnn_other_kernel_size, stride=1, padding=1))
            else:
                self.CNN.add_module(f"CNN_{i+1}", nn.Conv1d(in_channels=cnn_other_filter, out_channels=cnn_other_filter, kernel_size=cnn_other_kernel_size, stride=1, padding=1))
                seq_length = ((seq_length + (2*1) - (1 * (cnn_other_kernel_size - 1 )) - 1)/1) + 1

            self.CNN.add_module(f"RELU_{i+1}", torch.nn.ReLU())
        
        self.biLSTM = nn.LSTM(input_size=cnn_other_filter,hidden_size=bilstm_hidden_size,batch_first=True,bidirectional=True, num_layers=bilstm_layer)
        
        self.flatten = nn.Flatten()

        self.FC = nn.Sequential(
            # nn.Linear(in_features=int(2*bilstm_hidden_size*seq_length),out_features=fc_size),
            nn.LazyLinear(out_features=fc_size),
            # based on your final concatenated features size
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(fc_size, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x) -> None:
        out = self.CNN(x) #[bs feature_dim seq_length]
        out = out.permute(0, 2, 1) #[bs seq_length feature_dim]
        out, h = self.biLSTM(out) #[bs seq_length feature_dim]
        out = self.flatten(out) 
        out = self.FC(out)
        out = out
        return out

class ConfigurableModelWoBatchNorm(nn.Module):
    ## CNN + LSTM
    def __init__(self, input_channel=4, input_size = 1001, cnn_first_filter=8, cnn_first_kernel_size=7, cnn_length=3, 
                 cnn_other_filter=32, cnn_other_kernel_size=6, bilstm_layer=2, bilstm_hidden_size=128, fc_size=256, output_size=1) -> None:
        super().__init__()
        self.CNN = torch.nn.Sequential()

        seq_length = input_size
        for i in range(cnn_length):
            if i == 0:
                self.CNN.add_module(f"CNN_{i+1}", nn.Conv1d(in_channels=input_channel, out_channels=cnn_first_filter,kernel_size=cnn_first_kernel_size,stride=2,padding=0)
                )

                seq_length = ((input_size - (1 * (cnn_first_kernel_size - 1 )) - 1)/2) + 1
            elif i == 1:
                self.CNN.add_module(f"CNN_{i+1}", nn.Conv1d(in_channels=cnn_first_filter, out_channels=cnn_other_filter, kernel_size=cnn_other_kernel_size, stride=1, padding=1))
            else:
                self.CNN.add_module(f"CNN_{i+1}", nn.Conv1d(in_channels=cnn_other_filter, out_channels=cnn_other_filter, kernel_size=cnn_other_kernel_size, stride=1, padding=1))
                seq_length = ((seq_length + (2*1) - (1 * (cnn_other_kernel_size - 1 )) - 1)/1) + 1

            self.CNN.add_module(f"RELU_{i+1}", torch.nn.ReLU())
        
        self.biLSTM = nn.LSTM(input_size=cnn_other_filter,hidden_size=bilstm_hidden_size,batch_first=True,bidirectional=True, num_layers=bilstm_layer)
        
        self.flatten = nn.Flatten()

        self.FC = nn.Sequential(
            # nn.Linear(in_features=int(2*bilstm_hidden_size*seq_length),out_features=fc_size),
            nn.LazyLinear(out_features=fc_size),
            # based on your final concatenated features size
            nn.ReLU(),
            nn.Linear(fc_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x) -> None:
        out = self.CNN(x) #[bs feature_dim seq_length]
        out = out.permute(0, 2, 1) #[bs seq_length feature_dim]
        out, h = self.biLSTM(out) #[bs seq_length feature_dim]
        out = self.flatten(out) 
        out = self.FC(out)
        out = out
        return out
    



class TestMotifModel(nn.Module):
    ## Number of filters of the first CNN layer should not be over parametrized. From a paper 25 is the optimal but we can experiment with lesser number of filter 
    ## Number of maxpooling could affect the way the network learn the motif. Higher number of maxpooling could lead to better motif learning. However, not overparaterize it! 
    ## Number of kernel size of the first CNN layer 
    ## However it is tradeoff b/w learning local motifs and distributed. Buet we have LSTM!
    ## Reduce CNN layers reduce performnce 
    ## GRU reduce a little bit performancde 
    ## Concatenating instead hybrid makes .... 
    ## Removing LSTM-like -> with LSTM it is faster to convergence and better performance! without it it barely reaches 0.4!
    ## However it is tradeoff b/w learning local motifs and distributed. Buet we have LSTM!
    ## Adding Promoter -> 
    def __init__(self, input_channel=4, input_size = 1001, cnn_first_filter=8, cnn_first_kernel_size=6,
                 cnn_other_filter=32, cnn_other_kernel_size=6, bilstm_layer=2, bilstm_hidden_size=128, fc_size=256, output_size=1) -> None:
        super().__init__()
        self.CNN = torch.nn.Sequential()

        self.CNN.add_module(f"CNN_{1}", nn.Conv1d(input_channel, cnn_first_filter, kernel_size=cnn_first_kernel_size, padding='same'))
        self.CNN.add_module(f"RELU_{1}", torch.nn.ReLU())
        self.CNN.add_module(f"DROPOUT_{1}", torch.nn.Dropout(0.1))
        self.CNN.add_module(f"BATCHNORM_{1}", torch.nn.BatchNorm1d(cnn_first_filter))
        self.CNN.add_module(f"MAX_POOL_{1}", torch.nn.MaxPool1d(kernel_size=2)) #(Lin-(k-1)-1/k)+1

        self.CNN.add_module(f"CNN_{2}", nn.Conv1d(cnn_first_filter, cnn_other_filter, kernel_size=cnn_other_kernel_size, padding='same'))
        self.CNN.add_module(f"RELU_{2}", torch.nn.ReLU())
        self.CNN.add_module(f"DROPOUT_{1}", torch.nn.Dropout(0.2))
        self.CNN.add_module(f"BATCHNORM_{2}", torch.nn.BatchNorm1d(cnn_other_filter))
        self.CNN.add_module(f"MAX_POOL_{2}", torch.nn.MaxPool1d(kernel_size=2)) #(Lin-(k-1)-1/k)+1

        self.CNN.add_module(f"CNN_{3}", nn.Conv1d(cnn_other_filter, cnn_other_filter, kernel_size=cnn_other_kernel_size, padding='same'))
        self.CNN.add_module(f"RELU_{3}", torch.nn.ReLU())
        self.CNN.add_module(f"DROPOUT_{1}", torch.nn.Dropout(0.2))
        self.CNN.add_module(f"BATCHNORM_{3}", torch.nn.BatchNorm1d(cnn_other_filter))
        self.CNN.add_module(f"MAX_POOL_{3}", torch.nn.MaxPool1d(kernel_size=4)) #(Lin-(k-1)-1/k)+1

        self.biLSTM = nn.LSTM(input_size=cnn_other_filter,hidden_size=bilstm_hidden_size,batch_first=True,bidirectional=True, num_layers=bilstm_layer)
        
        self.flatten = nn.Flatten()

        self.FC = nn.Sequential(
            # nn.Linear(in_features=int(2*bilstm_hidden_size*seq_length),out_features=fc_size
            nn.LazyLinear(out_features=fc_size),
            nn.Dropout(0.2),
            # based on your final concatenated features size
            nn.ReLU(),
            nn.Linear(fc_size, 64),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x) -> None:
        out = self.CNN(x) #[bs feature_dim seq_length]
        out = out.permute(0, 2, 1) #[bs seq_length feature_dim]
        out, h = self.biLSTM(out) #[bs seq_length feature_dim]
        out = self.flatten(out) 
        out = self.FC(out)
        out = out
        return out

class TestMotifModel2(nn.Module):
    ## Number of filters of the first CNN layer should not be over parametrized. From a paper 25 is the optimal but we can experiment with lesser number of filter 
    ## Number of maxpooling could affect the way the network learn the motif. Higher number of maxpooling could lead to better motif learning. However, not overparaterize it! 
    ## Number of kernel size of the first CNN layer 
    ## However it is tradeoff b/w learning local motifs and distributed. Buet we have LSTM!
    ## Reduce CNN layers reduce performnce 
    ## GRU reduce a little bit performancde 
    ## Concatenating instead hybrid makes .... 
    ## Removing LSTM-like -> with LSTM it is faster to convergence and better performance! without it it barely reaches 0.4!
    def __init__(self, input_channel=4, input_size = 1001, cnn_first_filter=8, cnn_first_kernel_size=6,
                 cnn_other_filter=32, cnn_other_kernel_size=6, bilstm_layer=2, bilstm_hidden_size=128, fc_size=256, output_size=1) -> None:
        super().__init__()
        self.CNN = torch.nn.Sequential()

        self.CNN.add_module(f"CNN_{1}", nn.Conv1d(input_channel, cnn_first_filter, kernel_size=cnn_first_kernel_size, padding='same'))
        self.CNN.add_module(f"RELU_{1}", torch.nn.ReLU())
        self.CNN.add_module(f"DROPOUT_{1}", torch.nn.Dropout(0.1))
        self.CNN.add_module(f"BATCHNORM_{1}", torch.nn.BatchNorm1d(cnn_first_filter))
        self.CNN.add_module(f"MAX_POOL_{1}", torch.nn.MaxPool1d(kernel_size=4)) #(Lin-(k-1)-1/k)+1

        self.CNN.add_module(f"CNN_{2}", nn.Conv1d(cnn_first_filter, cnn_other_filter, kernel_size=cnn_other_kernel_size, padding='same'))
        self.CNN.add_module(f"RELU_{2}", torch.nn.ReLU())
        self.CNN.add_module(f"DROPOUT_{1}", torch.nn.Dropout(0.2))
        self.CNN.add_module(f"BATCHNORM_{2}", torch.nn.BatchNorm1d(cnn_other_filter))
        self.CNN.add_module(f"MAX_POOL_{2}", torch.nn.MaxPool1d(kernel_size=2)) #(Lin-(k-1)-1/k)+1

        self.CNN.add_module(f"CNN_{2}", nn.Conv1d(cnn_first_filter, cnn_other_filter, kernel_size=cnn_other_kernel_size, padding='same'))
        self.CNN.add_module(f"RELU_{2}", torch.nn.ReLU())
        self.CNN.add_module(f"DROPOUT_{1}", torch.nn.Dropout(0.2))
        self.CNN.add_module(f"BATCHNORM_{2}", torch.nn.BatchNorm1d(cnn_other_filter))
        self.CNN.add_module(f"MAX_POOL_{2}", torch.nn.MaxPool1d(kernel_size=2)) #(Lin-(k-1)-1/k)+1


        # self.biLSTM = nn.LSTM(input_size=cnn_other_filter,hidden_size=bilstm_hidden_size,batch_first=True,bidirectional=True, num_layers=bilstm_layer)
        # self.biGRU = nn.GRU(input_size=1001,hidden_size=bilstm_hidden_size,batch_first=True,bidirectional=True, num_layers=bilstm_layer)
        
        self.flatten = nn.Flatten()

        self.FC = nn.Sequential(
            # nn.Linear(in_features=int(2*bilstm_hidden_size*seq_length),out_features=fc_size),
            nn.LazyLinear(out_features=fc_size),
            nn.Dropout(0.2),
            # based on your final concatenated features size
            nn.ReLU(),
            nn.Linear(fc_size, 32),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x) -> None:
        out = self.CNN(x) #[bs feature_dim seq_length]
        # out = out.permute(0, 2, 1) #[bs seq_length feature_dim]
        # out, h = self.biLSTM(out) #[bs seq_length feature_dim]
        # out_gru, h = self.biGRU(x) #[bs seq_length feature_dim]
        out = self.flatten(out) 
        # out_gru = self.flatten(out_gru)
        # out = torch.concat(out, out_gru, dim=1)
        out = self.FC(out)
        out = out
        return out

if __name__=="__main__":
    #print(NaiveModelV2(1001))
    kernel_size=7
    stride=2 
    padding=0
    dilation=1
    lout = ((2001 + (2*padding) - (dilation * (kernel_size - 1 )) - 1)/stride) + 1
    print(lout)
    kernel_size=3 
    stride=1
    padding=1
    lout = ((lout + (2*padding) - (dilation * (kernel_size - 1 )) - 1)/stride) + 1
    print(lout)
    lout = (lout + 2*0 - 1*(2-1)-1)/2 + 1



    in_size = ((2001 - (1 * (7 - 1 )) - 1)/2) + 1
    in_size = ((in_size + (2*1) - (1 * (3 - 1 )) - 1)/1) + 1
    in_size = in_size/2
    in_size = ((in_size + (2*1) - (1 * (3 - 1 )) - 1)/1) + 1
    print(in_size)
    # print(lout)
    # kernel_size=3 
    # stride=1
    # padding=1
    # lout = ((lout + (2*padding) - (dilation * (kernel_size - 1 )) - 1)/stride) + 1
    # print(lout)
    # lout = (lout + 2*0 - 1*(2-1)-1)/2 + 1
    # print(lout)