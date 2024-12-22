import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import softmax

'''
class GATModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, heads=8, dropout=0.5):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

criterion = torch.nn.CrossEntropyLoss()
def loss_function(output, target):
    return criterion(output, target)
'''

class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=128, dropout=0.3):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
