import torch.nn.functional as F
from layers import *
from FCM import *
from constants import out_dim, hidden1_dim, hidden2_dim


class MyModel(nn.Module):

    def __init__(self, g, num_heads, num_community, K, n):
        super(MyModel, self).__init__()
        self.k = K
        self.num_community = num_community
        self.GAT_1 = MultiHeadGATLayer(g, n, hidden1_dim, num_heads, self.num_community)
        self.GAT_2 = MultiHeadGATLayer(g, hidden1_dim, hidden2_dim, num_heads, self.num_community)
        self.GAT_3 = MultiHeadGATLayer(g, hidden2_dim, out_dim, 1, self.num_community)

    def forward(self, h, v):
        h = self.GAT_1(h, v)
        h = F.elu(h)
        h = self.GAT_2(h, v)
        h = F.elu(h)
        self.emb = self.GAT_3(h, v)
        self.z = FCM(self.emb, self.k, 2)

    def getZ(self):
        return self.z

    def get_emb(self):
        return self.emb