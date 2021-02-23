import torch as tr
import torch.nn as nn
from models.squeeze import Squeeze
from models.attention import AttentionLayer
from models.none import None_
import configurations

class KiNET(nn.Module):
  def __init__(self, middle_layer, first_layer_channels=10, second_layer_channels=10, third_layer_channels=configurations.num_of_filters, preAttentionSize=50, cnnConfig="DBBNNN", attention=False, dropout=.5):
    super().__init__()
    mid_cnn_act = []
    for i in range(len(cnnConfig)):
        if cnnConfig[i] == 'D':
            mid_cnn_act.append(nn.Dropout(dropout))
        elif cnnConfig[i] == 'B' and i == 0:
            mid_cnn_act.append(nn.BatchNorm2d(first_layer_channels))
        elif cnnConfig[i] == 'B' and i == 1:
            mid_cnn_act.append(nn.BatchNorm1d(second_layer_channels))
        elif cnnConfig[i] == 'B' and i == 2:
            mid_cnn_act.append(nn.BatchNorm1d(third_layer_channels))
        elif cnnConfig[i] == 'R':
            mid_cnn_act.append(nn.ReLU())
        elif cnnConfig[i] == 'L':
            mid_cnn_act.append(nn.LeakyReLU())
        elif cnnConfig[i] == 'N':
            mid_cnn_act.append(None_())
            
    self.FirstCNN = nn.Sequential(
      nn.Conv3d(1, first_layer_channels, kernel_size=(5,2,3), stride=(4,2,1)),
      Squeeze(),
      mid_cnn_act[3],
      mid_cnn_act[0],
    ) 
    self.SecondCNN = nn.Sequential(
      nn.Conv2d(first_layer_channels, second_layer_channels, kernel_size=(6,2)),
      Squeeze(),
      mid_cnn_act[4],
      mid_cnn_act[1],
    )   
    self.LastCNN = nn.Sequential(
        nn.Conv1d(second_layer_channels, third_layer_channels, kernel_size=6),
        mid_cnn_act[5],
        mid_cnn_act[2],
    )
    
    self.attention = attention

    self.intermediateLayer = middle_layer

    self.attentionLayer = AttentionLayer(third_layer_channels * self.intermediateLayer.isBi)

    self.FC = nn.Linear(third_layer_channels * self.intermediateLayer.isBi, 1)

    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.FirstCNN(x)
    x = self.SecondCNN(x)
    x = self.LastCNN(x)

    x = x.permute(0,2,1)

    x = self.intermediateLayer(x)
    
    if self.attention:
        x = self.attentionLayer(x)
    else:
        x = tr.sum(x, dim=1)
    
    
    x = self.FC(x)

    x = self.sigmoid(x)
    return x


