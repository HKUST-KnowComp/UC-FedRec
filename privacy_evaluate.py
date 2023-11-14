from feature_extracter import AgeDiscriminator, GenderDiscriminator, OccupationDiscriminator, LocationDiscriminator
from main_FedRec_ML import TwoSideGraphModel
import torch
import torch.nn as nn
import numpy as np
from utility.parser import parse_args
import os
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from utility.load_data import Data
import random
import copy
from torch.utils.data import WeightedRandomSampler
import dgl
from sklearn.neural_network import MLPClassifier

args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
if args.gpu >= 0 and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size, dataset=args.dataset)
age_discriminator = AgeDiscriminator(args.embed_size + sum(args.layer_size)).to(device)
occupation_discriminator = OccupationDiscriminator(args.embed_size + sum(args.layer_size)).to(device)
gender_discriminator = GenderDiscriminator(args.embed_size + sum(args.layer_size)).to(device)
location_discriminator = LocationDiscriminator(args.embed_size + sum(args.layer_size)).to(device)


def load_rec_model(name, n_users, n_items):
    state_dict = torch.load(name, map_location=device)
    model = TwoSideGraphModel(args.embed_size, args.layer_size,
                              args.mess_dropout, args.regs[0])
    g_user = dgl.add_self_loop(dgl.graph([], num_nodes=n_users)).to(device)
    g_item = dgl.add_self_loop(dgl.graph([], num_nodes=n_items)).to(device)
    model.init_parameters(g_user, g_item)
    model.load_state_dict(state_dict)
    return model


def sample_user(user_list, batch_size=args.user_batch_size):
    return random.sample(user_list, batch_size)


def privacy_estimator_train(attr, embed):
    if attr == 'gender':
        model = gender_discriminator
        labels = [0 if att['gender'] is 'M' else 1 for att in data_generator.users_features]

        criterion = nn.BCELoss()
        weight = {0: 1 / (len(labels) - sum(labels)), 1: 1 / sum(labels)}
        weight = [weight[i] for i in labels]
        labels = torch.tensor(labels, dtype=torch.float).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif attr == 'age':
        model = age_discriminator
        reindex = {'1': 0, '18': 1, '25': 2, '35': 3, '45': 4, '50': 5, '56': 6}
        labels = [reindex[att['age']] for att in data_generator.users_features]
        weight = [0] * 7
        for i in labels:
            weight[i] += 1
        for i in range(7):
            weight[i] = 1 / weight[i]
        weight = [weight[i] for i in labels]
        labels = torch.tensor(labels).to(device)
        criterion = nn.NLLLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif attr == 'occupation':
        model = occupation_discriminator
        labels = [int(att['occupation']) for att in data_generator.users_features]
        weight = [0] * 21
        for i in labels:
            weight[i] += 1
        for i in range(21):
            weight[i] = 1 / weight[i]
        weight = [weight[i] for i in labels]
        labels = torch.tensor(labels).to(device)
        criterion = nn.NLLLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif attr == 'location':
        model = location_discriminator
        labels = [int(att['location']) for att in data_generator.users_features]
        weight = [0] * 453
        # for i in labels:
        #     weight[i] += 1
        # for i in range(453):
        #     weight[i] = 1 / weight[i]
        # weight = [weight[i] for i in labels]
        labels = torch.tensor(labels).to(device)
        criterion = nn.NLLLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    else:
        print('no {} attribute found'.format(attr))
        return
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    model.train()
    user_embed = embed.to(device)
    user_list = np.array(list(range(data_generator.n_users)))
    user_train, user_test = train_test_split(user_list, test_size=0.3, random_state=1)
    user_test, user_valid = train_test_split(user_test, test_size=0.5, random_state=1)
    best_record = 0
    best_model = None
    best_epoch = 0
    # weight = np.array(weight)
    # user_train_weight = weight[user_train]
    record_loss = []
    for epoch in range(1000):
        model.train()
        users_batch = sample_user(user_train.tolist(), 2000)
        # users_batch = user_train[list(WeightedRandomSampler(user_train_weight, 1024))]
        # users_batch = sample_user(list(range(data_generator.n_users)), 3000)
        embed_batch = user_embed[users_batch]
        label_batch = labels[users_batch]
        # print(sum(label_batch))
        output = model(embed_batch)
        loss = criterion(output.squeeze(), label_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        score, predict = model.predict(user_embed[user_valid])
        record_loss.append(criterion(model(user_embed[user_valid]).squeeze(), labels[user_valid]).detach().cpu())
        if attr == 'gender':
            record = roc_auc_score(labels[user_valid].cpu(), score.cpu())
        else:
            record = f1_score(labels[user_valid].cpu(), predict.cpu(), average='micro')
        if record > best_record:
            best_record = record
            best_model = copy.deepcopy(model)
            best_epoch = epoch
    best_model.eval()
    score, predict = best_model.predict(user_embed[user_test])
    if attr == 'gender':
        auc_score = roc_auc_score(labels[user_test].cpu(), score.cpu())
        msg = '{} discriminator prediction auc score: {}'.format(attr, auc_score)
        print(msg)
    else:
        f1 = f1_score(labels[user_test].cpu(), predict.cpu(), average='micro')
        msg = '{} discriminator prediction f1 score: {}'.format(attr, f1)
        print(msg)
    print(best_epoch)


if __name__ == '__main__':

    # path = '/home/qhuaf/NGCF/model/PartTwoSideGraph.pkl'
    # rec_model = load_rec_model(path,
    #                            data_generator.n_users, data_generator.n_items).to(device)
    # path = '/data/qhuaf/graph_pri/model/test1.pkl'
    # path = '/home/qhuaf/graph_pri/save_model/0.5/with_pri_1_0.5.pkl'
    path = '/home/qhuaf/graph_pri/save_model/douban/douban_with_pri_0.5_0.5.pkl'
    rec_model = load_rec_model(path,
                               data_generator.n_users, data_generator.n_items).to(device)
    # adj = data_generator.g.adj(etype='ui')
    # neighbor_user = torch.sparse.mm(adj, adj.t().to_dense())
    # neighbor_item = torch.sparse.mm(adj.t(), adj.to_dense())
    # tmp_user_out = torch.topk(neighbor_user, args.num_neighbor)[1].flatten()
    # tmp_user_in = torch.LongTensor(list(range(data_generator.n_users))).repeat(args.num_neighbor).reshape(
    #     (-1, data_generator.n_users)).t().flatten()
    # tmp_item_out = torch.topk(neighbor_item, args.num_neighbor)[1].flatten()
    # tmp_item_in = torch.LongTensor(list(range(data_generator.n_items))).repeat(args.num_neighbor).reshape(
    #     (-1, data_generator.n_items)).t().flatten()
    #
    # g_user = dgl.graph((tmp_user_out, tmp_user_in)).to(device)
    # g_item = dgl.graph((tmp_item_out, tmp_item_in)).to(device)

    user_emb = rec_model.feature_dict['user']
    emb_similarity = torch.cosine_similarity(user_emb.unsqueeze(1).cpu(), user_emb.unsqueeze(0).cpu(), dim=2)
    tmp_user_out = torch.topk(emb_similarity, args.num_neighbor)[1].flatten()
    tmp_user_in = torch.LongTensor(list(range(data_generator.n_users))).repeat(args.num_neighbor).reshape(
        (-1, data_generator.n_users)).t().flatten()
    g_user = dgl.graph((tmp_user_out, tmp_user_in)).to(device)
    tmp_item_out = torch.LongTensor(list(range(data_generator.n_items)))
    tmp_item_in = torch.LongTensor(list(range(data_generator.n_items)))
    g_item = dgl.graph((tmp_item_out, tmp_item_in)).to(device)

    rec_model.eval()
    rec_model.g_user = g_user
    rec_model.g_item = g_item
    user_embed, _, _ = rec_model(list(range(data_generator.n_users)), [], [])
    user_embed = user_embed.detach().cpu()
    privacy_estimator_train('location', user_embed)
    # privacy_estimator_train('gender', user_embed)
    # privacy_estimator_train('age', user_embed)
    # privacy_estimator_train('occupation', user_embed)
