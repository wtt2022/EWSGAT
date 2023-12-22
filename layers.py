import torch
import torch.nn as nn
from entmax import alpha_entmax

class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_community):
        super(GATLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.attn_fc2 = nn.Linear(2 * num_community, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['u'], edges.dst['u']], dim=1)
        a = self.attn_fc2(z2)
        return {'e': alpha_entmax(a, alpha = 1.5, dim = 1)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = alpha_entmax(nodes.mailbox['e'], alpha = 1.5, dim = 1)
        h = torch.sum(alpha * nodes.mailbox['z'] * nodes.mailbox['e'], dim=1)
        return {'h': h}

    def forward(self, h, u):
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.ndata['u'] = u
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, num_community):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim, num_community))

        self.weights = nn.Parameter(torch.ones(num_heads))

    def forward(self, h, u):
        head_outs = [attn_head(h, u) for attn_head in self.heads]

        weights_softmax = torch.softmax(self.weights, dim=0)

        weighted_sum = torch.stack(head_outs, dim=-1) * weights_softmax
        output = torch.sum(weighted_sum, dim=-1)

        return output


