import torch.nn as nn
import torch
from dgl.nn.pytorch import SAGEConv
import dgl
import torch.nn.functional as F
from utility.parser import parse_args
import os


args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
if args.gpu >= 0 and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class UserRecModel(nn.Module):
    def __init__(self, lmbd):
        super(UserRecModel, self).__init__()
        self.lmbd = lmbd

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = (users * pos_items).sum(1)
        neg_scores = (users * neg_items).sum(1)

        mf_loss = nn.LogSigmoid()(pos_scores - neg_scores).mean()
        mf_loss = -1 * mf_loss

        regularize = (torch.norm(users) ** 2 + torch.norm(pos_items) ** 2 + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.lmbd * regularize / users.shape[0]

        return mf_loss + emb_loss, mf_loss, emb_loss

    def create_cross_entropy_loss(self, users, items, target, w=15.0):
        scores = nn.Sigmoid()((users * items).sum(1))
        # weights = torch.tensor([1.0 if i == 0 else w for i in target]).cuda()
        mf_loss = nn.BCELoss()(scores, target)
        # mf_loss = nn.BCELoss(weight=weights)(scores, target)

        regularizer = (torch.norm(users) ** 2 + torch.norm(items) ** 2) / 2
        emb_loss = self.lmbd * regularizer / users.shape[0]
        return mf_loss + emb_loss, mf_loss, emb_loss

    @staticmethod
    def rating(u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def forward(self, *arg, **kwargs):
        raise NotImplementedError

    def update_parameters(self, *arg, **kwargs):
        raise NotImplementedError

    def training_data_processing(self):
        raise NotImplementedError


class TwoSideGraphModel(UserRecModel):
    def __init__(self, in_size, layer_size, dropout, lmbd=1e-5):
        super(TwoSideGraphModel, self).__init__(lmbd)
        self.in_size = in_size
        self.user_layers = nn.ModuleList()
        self.item_layers = nn.ModuleList()
        self.user_layers.append(SAGEConv(in_size, layer_size[0], 'mean', activation=nn.LeakyReLU(0.2)))
        for i in range(len(layer_size) - 1):
            self.user_layers.append(SAGEConv(layer_size[i], layer_size[i + 1], 'mean', activation=nn.LeakyReLU(0.2)))
        self.item_layers.append(nn.Linear(in_size, layer_size[0]))
        for i in range(len(layer_size) - 1):
            self.item_layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))

        self.feature_dict = nn.ParameterDict()
        self.dropout = nn.Dropout(dropout[0])
        self.g_user = dgl.graph([])
        self.g_item = dgl.graph([])

    def update_parameters(self, weight):
        self.load_state_dict(weight)

    def init_parameters(self, g_user, g_item):
        self.g_user = g_user
        self.g_item = g_item
        self.feature_dict = nn.ParameterDict({
            'user': nn.Parameter(nn.init.xavier_uniform_(torch.empty(g_user.num_nodes(), self.in_size))),
            'item': nn.Parameter(nn.init.xavier_uniform_(torch.empty(g_item.num_nodes(), self.in_size)))
        }).to(device)

    def update_graph(self, g_user, g_item):
        self.g_user = g_user
        self.g_item = g_item

    def forward(self, users, pos_items, neg_items):
        h_user = self.feature_dict['user']
        h_item = self.feature_dict['item']
        user_embeds = []
        item_embeds = []
        user_embeds.append(h_user)
        item_embeds.append(h_item)
        for layer in self.user_layers:
            h_user = layer(self.g_user, h_user)
            h_user = self.dropout(h_user)  # dropout
            h_user = F.normalize(h_user, dim=1, p=2)
            user_embeds.append(h_user)
        for layer in self.item_layers:
            h_item = layer(h_item)
            h_item = nn.LeakyReLU(0.2)(h_item)
            h_item = self.dropout(h_item)  # dropout
            h_item = F.normalize(h_item, dim=1, p=2)
            item_embeds.append(h_item)

        user_embd = torch.cat(user_embeds, 1)
        item_embd = torch.cat(item_embeds, 1)
        u_g_embeddings = user_embd[users, :]
        pos_i_g_embeddings = item_embd[pos_items, :]
        neg_i_g_embeddings = item_embd[neg_items, :]
        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings

    def one_train(self, users, pos_items, neg_items):
        u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = self.forward(users, pos_items, neg_items)
        loss, mf_loss, emb_loss = self.create_bpr_loss(u_g_embeddings,
                                                       pos_i_g_embeddings, neg_i_g_embeddings)

        return loss, mf_loss, emb_loss

    def one_train_cross_entropy_loss(self, users, items, targets, w=15.0):
        u_g_embeddings, i_g_embeddings, _ = self.forward(users, items, [])
        loss, mf_loss, emb_loss = self.create_cross_entropy_loss(
            u_g_embeddings, i_g_embeddings, torch.tensor(targets, dtype=torch.float32, device=device), w)

        return loss, mf_loss, emb_loss

    def one_test(self, users, pos_items, neg_items):
        u_g_embeddings, pos_i_g_embeddings, _ = self.forward(users, pos_items, neg_items)
        rate_batch = self.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
        return rate_batch

    def training_data_processing(self):
        pass

class NCF(UserRecModel):
    def __init__(self, in_size, layer_size, dropout, lmbd=1e-5):
        super(NCF, self).__init__(lmbd)
        self.in_size = in_size
        self.user_layers = nn.ModuleList()
        self.item_layers = nn.ModuleList()
        self.user_layers.append(nn.Linear(in_size, layer_size[0]))
        for i in range(len(layer_size) - 1):
            self.user_layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
        self.item_layers.append(nn.Linear(in_size, layer_size[0]))
        for i in range(len(layer_size) - 1):
            self.item_layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))

        self.feature_dict = nn.ParameterDict()
        self.dropout = nn.Dropout(dropout[0])

    def update_parameters(self, weight):
        self.load_state_dict(weight)

    def init_parameters(self, g_user, g_item):
        self.feature_dict = nn.ParameterDict({
            'user': nn.Parameter(nn.init.xavier_uniform_(torch.empty(g_user.num_nodes(), self.in_size))),
            'item': nn.Parameter(nn.init.xavier_uniform_(torch.empty(g_item.num_nodes(), self.in_size)))
        }).to(device)

    def forward(self, users, pos_items, neg_items):
        h_user = self.feature_dict['user']
        h_item = self.feature_dict['item']
        user_embeds = []
        item_embeds = []
        user_embeds.append(h_user)
        item_embeds.append(h_item)
        for layer in self.user_layers:
            h_user = layer(h_user)
            h_user = nn.LeakyReLU(0.2)(h_user)
            h_user = self.dropout(h_user)  # dropout
            h_user = F.normalize(h_user, dim=1, p=2)
            user_embeds.append(h_user)
        for layer in self.item_layers:
            h_item = layer(h_item)
            h_item = nn.LeakyReLU(0.2)(h_item)
            h_item = self.dropout(h_item)  # dropout
            h_item = F.normalize(h_item, dim=1, p=2)
            item_embeds.append(h_item)

        user_embd = torch.cat(user_embeds, 1)
        item_embd = torch.cat(item_embeds, 1)
        u_g_embeddings = user_embd[users, :]
        pos_i_g_embeddings = item_embd[pos_items, :]
        neg_i_g_embeddings = item_embd[neg_items, :]
        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings

    def one_train(self, users, pos_items, neg_items):
        u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = self.forward(users, pos_items, neg_items)
        loss, mf_loss, emb_loss = self.create_bpr_loss(u_g_embeddings,
                                                       pos_i_g_embeddings, neg_i_g_embeddings)

        return loss, mf_loss, emb_loss

    def one_train_cross_entropy_loss(self, users, items, targets, w=15.0):
        u_g_embeddings, i_g_embeddings, _ = self.forward(users, items, [])
        loss, mf_loss, emb_loss = self.create_cross_entropy_loss(
            u_g_embeddings, i_g_embeddings, torch.tensor(targets, dtype=torch.float32, device=device), w)

        return loss, mf_loss, emb_loss

    def one_test(self, users, pos_items, neg_items):
        u_g_embeddings, pos_i_g_embeddings, _ = self.forward(users, pos_items, neg_items)
        rate_batch = self.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
        return rate_batch

    def training_data_processing(self):
        pass
