from models import *
import os
import torch
import dgl
from utility.load_data import Data, DataGeneratorForTest
import copy
import logging
import torch.multiprocessing as mp
from utility.batch_test import Tester
from time import time
import numpy as np
import random
from utility.helper import early_stopping
from utility.dp_mechanism import cal_sensitivity, Laplace, Gaussian_Simple


class RecUser:
    def __init__(self, model, n_users, n_items):
        super(RecUser, self).__init__()
        self.rec_data = []
        self.pri_data = []
        self.model = model
        self.n_users = n_users
        self.n_items = n_items
        self.train_item = []
        self.train_user = []
        self.flag_embedding_train = True
        self.epoch = 0
        self.dp_eps = args.dp_eps
        self.dp_clip = args.dp_clip
        self.dp_delta = args.dp_delta
        self.dp_mechanism = args.dp_mechanism

    def train_parameters(self, serial, auxiliary_information, weight, rec_data, pri_data):
        self.rec_data = rec_data
        self.pri_data = pri_data
        self.auxiliary_information_update(auxiliary_information)
        self.model.update_parameters(weight)
        if len(self.rec_data) == 0:
            return 0, 0, 0
        optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9)

        self.model.train()
        num_batch = self.n_items // args.local_batch_size + 1
        loss, mf_loss, emb_loss = 0., 0., 0.
        rate = self.n_items // len(self.rec_data) - 1
        self.train_item = []
        for epoch in range(args.local_epoch):
            pos_items, neg_items = self.sample()
            self.train_item += pos_items
            self.train_item += neg_items
            users = [serial] * len(pos_items)
            batch_loss, batch_mf_loss, batch_emb_loss = self.model.one_train(users, pos_items, neg_items)
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.dp_clip)
            optimizer.step()
            loss += batch_loss.detach().cpu()
            mf_loss += batch_mf_loss.detach().cpu()
            emb_loss += batch_emb_loss.detach().cpu()

        # DP
        # self.clip_gradients(self.model)
        # optimizer.step()
        # loss += batch_loss.detach().cpu()
        # mf_loss += batch_mf_loss.detach().cpu()
        # emb_loss += batch_emb_loss.detach().cpu()
        # if not np.all(self.pri_demand == 0):
        #     self.add_noise(self.model)

        self.train_user_update(serial)
        self.train_item = list(set(self.train_item))
        self.train_user = list(set(self.train_user))
        return loss, mf_loss, emb_loss

    def train_user_update(self, serial):
        self.train_user = serial
        if args.model_name == 'two_side_graph':
            for i in range(len(args.layer_size)):
                self.train_user = self.model.g_user.in_subgraph(self.train_user).edges()[0].tolist()

    def test_parameters(self, serial, auxiliary_information, weight, tester):
        self.model.update_parameters(weight)
        pos_items = list(range(self.n_items))
        user_pos_train, test_items, tester = tester
        self.model.eval()
        rate_batch = self.model.one_test([serial], pos_items, [])
        re = tester.test_one_user((rate_batch[0], serial), user_pos_train, test_items)
        return re

    def upload_parameters(self):
        train_user_item = (copy.deepcopy(self.train_user), copy.deepcopy(self.train_item))
        self.train_user = []
        self.train_item = []
        return copy.deepcopy(self.model.state_dict()), len(self.rec_data), train_user_item

    def eliminate_privacy(self):
        pass

    def sample(self, neighbor_item=None):
        def sample_pos_items(num):
            # sample num pos items
            n_pos_items = len(self.rec_data)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = self.rec_data[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items(num):
            # sample num neg items
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.rec_data and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for _ in range(args.local_batch_size):
            pos_items += sample_pos_items(1)
            neg_items += sample_neg_items(1)

        return pos_items, neg_items

    def cross_entropy_sample(self):
        items = np.random.randint(low=0, high=self.n_items, size=args.local_batch_size)
        targets = [1 if item_tmp in self.rec_data else 0 for item_tmp in items]
        return items.tolist(), targets

    def server_train(self):
        pass

    def auxiliary_information_update(self, auxiliary_information):
        try:
            self.model.updata_graph(auxiliary_information['user'], auxiliary_information['item'])
        except:
            pass

    def clip_gradients(self, model):
        if self.dp_mechanism == 'Laplace':
            # Laplace use 1 norm
            for k, v in model.named_parameters():
                v.grad /= max(1, v.grad.norm(1) / self.dp_clip)
        elif self.dp_mechanism == 'Gaussian':
            # Gaussian use 2 norm
            for k, v in model.named_parameters():
                v.grad /= max(1, v.grad.norm(2) / self.dp_clip)

    def add_noise(self, net):
        sensitivity = cal_sensitivity(args.lr, self.dp_clip, 1)
        if self.dp_mechanism == 'Laplace':
            with torch.no_grad():
                for k, v in net.named_parameters():
                    noise = Laplace(epsilon=self.dp_eps, sensitivity=sensitivity, size=v.shape)
                    noise = torch.from_numpy(noise).to(device)
                    v += noise
        elif self.dp_mechanism == 'Gaussian':
            with torch.no_grad():
                for k, v in net.named_parameters():
                    noise = Gaussian_Simple(epsilon=self.dp_eps, delta=self.dp_delta, sensitivity=sensitivity,
                                            size=v.shape)
                    noise = torch.from_numpy(noise).to(device)
                    v += noise

class RecHost:
    def __init__(self, data_generator):
        super(RecHost, self).__init__()
        self.data_generator = data_generator
        self.n_users = data_generator.n_users
        self.n_items = data_generator.n_items

        if args.model_name == 'two_side_graph':
            self.global_rec_model = TwoSideGraphModel(args.embed_size, args.layer_size,
                                                      args.mess_dropout, args.regs[0]).to(device)
        elif args.model_name == 'NCF':
            self.global_rec_model = NCF(args.embed_size, args.layer_size,
                                        args.mess_dropout, args.regs[0]).to(device)
        else:
            raise Exception

        data_generator_for_test = DataGeneratorForTest(data_generator)
        self.tester = Tester(data_generator_for_test, args)
        # initialize the multiprocess type

        # generate the communication tunnel
        self.q_to_server = mp.Queue()
        self.q_to_client = mp.Queue()

        data_dict = {
            ('user', 'user_self', 'user'): (range(self.n_users), range(self.n_users)),
            ('item', 'item_self', 'item'): (range(self.n_items), range(self.n_items)),
            ('user', 'ui', 'item'): ([], []),
            ('item', 'iu', 'user'): ([], [])
        }
        self.g = dgl.heterograph(data_dict)
        self.g_user = dgl.graph((torch.LongTensor(range(self.n_users)), torch.LongTensor(range(self.n_users))))
        self.g_item = dgl.graph((torch.LongTensor(range(self.n_items)), torch.LongTensor(range(self.n_items))))
        self.global_rec_model.init_parameters(self.g_user.to(device), self.g_item.to(device))

        self.subgraph_generate()
        self.model_initialization = {'g_user': self.g_user, 'g_item': self.g_item}
        self.auxiliary_information = None
        self.global_rec_model.eval()
        print()

    @staticmethod
    def sample_user(user_list):
        return random.sample(user_list, args.user_batch_size)

    @staticmethod
    def kmeans_sample_user(user_list, label):
        tmp = []
        for i in range(8):
            tmp += np.random.choice(np.where(label == i)[0], args.user_batch_size // 8).tolist()
        return tmp

    def subgraph_generate(self):
        # adj = self.data_generator.g.adj(etype='ui')
        # neighbor_user = torch.sparse.mm(adj, adj.t().to_dense())
        # neighbor_item = torch.sparse.mm(adj.t(), adj.to_dense())
        # tmp_user_out = torch.topk(neighbor_user, args.num_neighbor)[1].flatten()
        # tmp_user_in = torch.LongTensor(list(range(self.n_users))).repeat(args.num_neighbor).reshape(
        #     (-1, self.n_users)).t().flatten()
        # tmp_item_out = torch.topk(neighbor_item, args.num_neighbor)[1].flatten()
        # tmp_item_in = torch.LongTensor(list(range(self.n_items))).repeat(args.num_neighbor).reshape(
        #     (-1, self.n_items)).t().flatten()
        # self.g_user = dgl.graph((tmp_user_out, tmp_user_in))
        # self.g_item = dgl.graph((tmp_item_out, tmp_item_in))
        user_emb = self.global_rec_model.feature_dict['user']
        emb_similarity = torch.cosine_similarity(user_emb.unsqueeze(1).cpu(), user_emb.unsqueeze(0).cpu(), dim=2)
        tmp_user_out = torch.topk(emb_similarity, args.num_neighbor)[1].flatten()
        tmp_user_in = torch.LongTensor(list(range(self.n_users))).repeat(args.num_neighbor).reshape(
            (-1, self.n_users)).t().flatten()
        self.g_user = dgl.graph((tmp_user_out, tmp_user_in)).to(device)
        tmp_item_out = torch.LongTensor(list(range(self.n_items)))
        tmp_item_in = torch.LongTensor(list(range(self.n_items)))
        self.g_item = dgl.graph((tmp_item_out, tmp_item_in)).to(device)

    def train_model(self):
        cur_best_pre_0, stopping_step = 0, 0
        loss_logger, pre_logger, rec_logger, ndcg_logger, hit_logger = [], [], [], [], []

        # generate multiprocess
        num_process = args.num_process
        process = []
        for rank in range(num_process):
            p = mp.Process(target=generate_one_client, args=(self.q_to_client, self.q_to_server,
                                                             self.n_users, self.n_items,
                                                             copy.deepcopy(self.model_initialization)))
            p.start()
            process.append(p)

        mp.set_start_method('fork', force=True)
        t0 = time()
        for epoch in range(args.epoch):
            t1 = time()
            user_list = self.sample_user(range(self.n_users))
            # kmeans = KMeans(n_clusters=8).fit(self.global_rec_model.state_dict()['feature_dict.user'].cpu())
            # user_list = self.kmeans_sample_user(range(self.n_users), kmeans.labels_)
            # if epoch % 25 == 0:
            #     self.subgraph_generate()
            # self.parallel_subgraph_generate()
            self.subgraph_generate()
            loss, mf_loss, emb_loss = self.parallel_local_train(user_list)

            # if epoch == 12:
            #     for i in range(args.num_process):
            #         self.q_to_client.put(([], [], [], [], [], 0))
            # loss, mf_loss, emb_loss = self.get_global_loss()

            if (epoch + 1) % 10 != 0:
                if args.verbose > 0 and epoch % args.verbose == 0:
                    perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                        epoch, time() - t1, loss, mf_loss, emb_loss)
                    print(perf_str)
                continue
            loss, mf_loss, emb_loss = self.get_global_loss()

            t2 = time()
            # ret = self.test_for_local()
            users_to_test = list(self.data_generator.test_set.keys())
            ret = self.tester.test(self.global_rec_model, self.g_user.to(device), users_to_test)
            t3 = time()

            loss_logger.append(loss)
            rec_logger.append(ret['recall'])
            pre_logger.append(ret['precision'])
            ndcg_logger.append(ret['ndcg'])
            hit_logger.append(ret['hit_ratio'])

            if args.verbose > 0:
                perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                           'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                           (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                            ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1])
                print(perf_str)
                logging.info(perf_str)

            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=50)

            # early stop
            if should_stop:
                break

            if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
                torch.save(self.global_rec_model.state_dict(), args.weights_path + args.save_name)
                print('save the weights in path: ', args.weights_path + args.save_name)
                logging.info('save the weights in path: ' + args.weights_path + args.save_name)

            torch.save(self.global_rec_model.state_dict(), args.weights_path + args.save_name)


        recs = np.array(rec_logger)
        pres = np.array(pre_logger)
        ndcgs = np.array(ndcg_logger)
        hit = np.array(hit_logger)

        best_rec_0 = max(recs[:, 0])
        idx = list(recs[:, 0]).index(best_rec_0)

        final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                     (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                      '\t'.join(['%.5f' % r for r in pres[idx]]),
                      '\t'.join(['%.5f' % r for r in hit[idx]]),
                      '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
        print(final_perf)
        logging.info(final_perf)

        # kill the sub process
        for p in process:
            p.terminate()

    def aggregate_params(self, user_list, params, local_n, train_user_item):
        # n = local_n[user_list[0]]
        params_avg = copy.deepcopy(params[user_list[0]])
        # for keys in params_avg.keys():
        #     params_avg[keys] = params_avg[keys] * n
        for i in range(1, len(user_list)):
            for keys in params_avg.keys():
                # params_avg[keys] += params[user_list[i]][keys] * local_n[user_list[i]]
                params_avg[keys] += params[user_list[i]][keys]
            # n += local_n[user_list[i]]
        for keys in params_avg.keys():
            # params_avg[keys] = params_avg[keys] / n
            params_avg[keys] = params_avg[keys] / len(user_list)

        user_n = torch.zeros(self.n_users).cuda()
        item_n = torch.zeros(self.n_items).cuda()
        tmp_user = torch.zeros((self.n_users, args.embed_size)).cuda()
        tmp_item = torch.zeros((self.n_items, args.embed_size)).cuda()
        for i in range(len(user_list)):
            tmp_user.index_add_(0, torch.LongTensor(train_user_item[user_list[i]][0]).cuda(),
                                params[user_list[i]]['feature_dict.user'][train_user_item[user_list[i]][0]])
            tmp_item.index_add_(0, torch.LongTensor(train_user_item[user_list[i]][1]).cuda(),
                                params[user_list[i]]['feature_dict.item'][train_user_item[user_list[i]][1]])
            user_n.index_add_(0, torch.LongTensor(train_user_item[user_list[i]][0]).cuda(),
                              torch.ones(len(train_user_item[user_list[i]][0])).cuda())
            item_n.index_add_(0, torch.LongTensor(train_user_item[user_list[i]][1]).cuda(),
                              torch.ones(len(train_user_item[user_list[i]][1])).cuda())
        for i in range(self.n_users):
            if user_n[i] == 0:
                tmp_user[i] = self.global_rec_model.state_dict()['feature_dict.user'][i]
            else:
                tmp_user[i] /= user_n[i]
        for i in range(self.n_items):
            if item_n[i] == 0:
                tmp_item[i] = self.global_rec_model.state_dict()['feature_dict.item'][i]
            else:
                tmp_item[i] /= item_n[i]
        params_avg['feature_dict.user'] = tmp_user
        params_avg['feature_dict.item'] = tmp_item

        self.global_rec_model.load_state_dict(params_avg)

    def parallel_local_train(self, user_list):
        # put data: (serial, user_item, weight, rec_data, pri_data, flag)
        num_batch = 0
        loss = []
        params = {}
        n_k = {}
        train_user_item = {}
        self.auxiliary_information = {'g_user': self.g_user, 'g_item': self.g_item}
        for j in range(len(user_list) // 64):
            for i in range(j * 64, min((j + 1) * 64, len(user_list))):
                self.q_to_client.put((user_list[i], copy.deepcopy(self.auxiliary_information),
                                      copy.deepcopy(self.global_rec_model.state_dict()),
                                      self.data_generator.train_items[user_list[i]], [], 1))

            for i in range(j * 64, min((j + 1) * 64, len(user_list))):
                # get data: (record_loss, record_mf_loss, record_emb_loss), (weight, len_data)
                loss_record, weight, train_user_item_client = self.q_to_server.get()
                loss.append(loss_record)
                params[user_list[i]], n_k[user_list[i]] = weight
                train_user_item[user_list[i]] = train_user_item_client

        self.aggregate_params(user_list, params, n_k, train_user_item)
        loss = np.array(loss).sum(axis=0)
        return loss[0], loss[1], loss[2]

    def test_for_local(self):
        count = 0
        Ks = eval(args.Ks)
        users_to_test = list(self.data_generator.test_set.keys())
        n_test_users = len(users_to_test)
        result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
                  'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}
        # 17s
        # sampler, predict_item = self.subgraph_generate(-1)

        # 2s
        for i in range(len(users_to_test)):
            # get data: serial, user_item, weight, tester, _, flag
            try:
                training_items = self.data_generator.train_items[users_to_test[i]]
            except Exception:
                training_items = []
            self.q_to_client.put((users_to_test[i], self.auxiliary_information,
                                  self.global_rec_model.state_dict(), training_items,
                                  self.data_generator.test_set[users_to_test[i]], 0))

        for _ in range(len(users_to_test)):
            re = self.q_to_server.get()
            count += 1

            result['precision'] += re['precision'] / n_test_users
            result['recall'] += re['recall'] / n_test_users
            result['ndcg'] += re['ndcg'] / n_test_users
            result['hit_ratio'] += re['hit_ratio'] / n_test_users
            result['auc'] += re['auc'] / n_test_users
        assert count == len(users_to_test)
        return result

    def privacy_estimator_train(self):
        pass

    def get_global_loss(self):
        # n_batch = self.data_generator.n_train // args.batch_size + 1
        loss, mf_loss, emb_loss = 0., 0., 0.
        for idx in range(1000):
            users, pos_items, neg_items = self.data_generator.sample()

            batch_loss, batch_mf_loss, batch_emb_loss = self.global_rec_model.one_train(users, pos_items, neg_items)
            loss += batch_loss.detach()
            mf_loss += batch_mf_loss.detach()
            emb_loss += batch_emb_loss.detach()

        return loss, mf_loss, emb_loss


def generate_one_client(q_to_client, q_to_server, n_users, n_items, model_initialization):
    print('user simulate id:', os.getpid())

    # generate a training model
    if args.model_name == 'two_side_graph':
        model = TwoSideGraphModel(args.embed_size, args.layer_size, args.mess_dropout, args.regs[0]).to(device)
        model.init_parameters(model_initialization['g_user'].to(device), model_initialization['g_item'].to(device))
    else:
        model = None

    user_rec = RecUser(model, n_users, n_items)

    while True:
        # in the loop, the user listen to the queue and receive the data and put the data to another queue
        # get data: (serial, user_item, weight, rec_data, pri_data, flag)
        serial, auxiliary_information, weight, rec_data, pri_data, flag = q_to_client.get()
        if flag == 1:
            # train
            record_loss, record_mf_loss, record_emb_loss = user_rec.train_parameters(serial, auxiliary_information,
                                                                                     weight, rec_data, pri_data)
            weight, len_data, train_user_item = user_rec.upload_parameters()
            q_to_server.put(((record_loss, record_mf_loss, record_emb_loss), (weight, len_data), train_user_item))
        elif flag == 0:
            args.lr /= 10

        elif flag == -1:
            # exit
            pass
        else:
            raise ValueError


def main():
    mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(args.log_path, 'Dist_{}_{}.log'.format('no_pri', args.log_name)),
                        filemode='a')
    logging.info(args)
    # data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
    data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size, dataset=args.dataset)
    host = RecHost(data_generator)
    host.train_model()


if __name__ == '__main__':
    main()
