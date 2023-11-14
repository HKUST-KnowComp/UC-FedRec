# This file is based on the NGCF author's implementation
# <https://github.com/xiangwang1223/neural_graph_collaborative_filtering/blob/master/NGCF/utility/batch_test.py>.
# It implements the batch test.

import utility.metric as metrics
from utility.load_data import *
import multiprocessing
import heapq


class Tester:
    def __init__(self, data_generator, args):
        self.USR_NUM, self.ITEM_NUM = data_generator.n_users, data_generator.n_items
        self.N_TRAIN, self.N_TEST = data_generator.n_train, data_generator.n_test
        self.BATCH_SIZE = args.batch_size
        self.Ks = eval(args.Ks)
        self.cores = multiprocessing.cpu_count()
        self.args = args
        self.data_generator = data_generator

    @staticmethod
    def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
        item_score = {}
        for i in test_items:
            item_score[i] = rating[i]

        K_max = max(Ks)
        K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

        r = []
        for i in K_max_item_score:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        auc = 0.
        return r, auc

    @staticmethod
    def get_auc(item_score, user_pos_test):
        item_score = sorted(item_score.items(), key=lambda kv: kv[1])
        item_score.reverse()
        item_sort = [x[0] for x in item_score]
        posterior = [x[1] for x in item_score]

        r = []
        for i in item_sort:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        auc = metrics.auc(ground_truth=r, prediction=posterior)
        return auc

    def ranklist_by_sorted(self, user_pos_test, test_items, rating, Ks):
        item_score = {}
        for i in test_items:
            item_score[i] = rating[i]

        K_max = max(Ks)
        K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

        r = []
        for i in K_max_item_score:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        auc = self.get_auc(item_score, user_pos_test)
        return r, auc

    @staticmethod
    def get_performance(user_pos_test, r, auc, Ks):
        precision, recall, ndcg, hit_ratio = [], [], [], []

        for K in Ks:
            precision.append(metrics.precision_at_k(r, K))
            recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
            ndcg.append(metrics.ndcg_at_k(r, K))
            hit_ratio.append(metrics.hit_at_k(r, K))

        return {'recall': np.array(recall), 'precision': np.array(precision),
                'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}

    def test_one_user_parallel(self, x, training_items, user_pos_test):
        # user u's ratings for user u
        rating = x[0]
        # uid
        u = x[1]
        # user u's items in the training set
        # try:
        #     training_items = self.data_generator.train_items[u]
        # except Exception:
        #     training_items = []
        # # user u's items in the test set
        # user_pos_test = self.data_generator.test_set[u]

        all_items = set(range(self.ITEM_NUM))

        test_items = list(all_items - set(training_items))

        if self.args.test_flag == 'part':
            r, auc = self.ranklist_by_heapq(user_pos_test, test_items, rating, self.Ks)
        else:
            r, auc = self.ranklist_by_sorted(user_pos_test, test_items, rating, self.Ks)

        return self.get_performance(user_pos_test, r, auc, self.Ks)

    def test_one_user(self, x):
        # user u's ratings for user u
        rating = x[0]
        # uid
        u = x[1]
        # user u's items in the training set
        try:
            training_items = self.data_generator.train_items[u]
        except Exception:
            training_items = []
        # user u's items in the test set
        user_pos_test = self.data_generator.test_set[u]

        all_items = set(range(self.ITEM_NUM))

        # test_items = list(all_items - set(training_items))
        test_items = list(all_items - set(training_items) - set(user_pos_test))
        test_items = random.sample(test_items, 49)
        test_items += user_pos_test

        if self.args.test_flag == 'part':
            r, auc = self.ranklist_by_heapq(user_pos_test, test_items, rating, self.Ks)
        else:
            r, auc = self.ranklist_by_sorted(user_pos_test, test_items, rating, self.Ks)

        return self.get_performance(user_pos_test, r, auc, self.Ks)

    def test(self, model, g, users_to_test, batch_test_flag=False):
        result = {'precision': np.zeros(len(self.Ks)), 'recall': np.zeros(len(self.Ks)), 'ndcg': np.zeros(len(self.Ks)),
                  'hit_ratio': np.zeros(len(self.Ks)), 'auc': 0.}

        pool = multiprocessing.Pool(self.cores)

        u_batch_size = 1000
        i_batch_size = self.BATCH_SIZE

        test_users = users_to_test
        n_test_users = len(test_users)
        n_user_batchs = n_test_users // u_batch_size + 1

        count = 0

        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size

            user_batch = test_users[start: end]

            if batch_test_flag:
                # batch-item test
                n_item_batchs = self.ITEM_NUM // i_batch_size + 1
                rate_batch = np.zeros(shape=(len(user_batch), self.ITEM_NUM))

                i_count = 0
                for i_batch_id in range(n_item_batchs):
                    i_start = i_batch_id * i_batch_size
                    i_end = min((i_batch_id + 1) * i_batch_size, self.ITEM_NUM)

                    item_batch = range(i_start, i_end)

                    u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch, item_batch, [])
                    i_rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()

                    rate_batch[:, i_start: i_end] = i_rate_batch
                    i_count += i_rate_batch.shape[1]

                assert i_count == self.ITEM_NUM

            else:
                # all-item test
                item_batch = range(self.ITEM_NUM)
                u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch, item_batch, [])
                rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
                # rate_batch = model.rating(torch.LongTensor(user_batch).to(device),
                #                           torch.LongTensor(item_batch).to(device)).detach().cpu()

                # for user in user_batch:
                #     rate_batch.append(model(torch.LongTensor([user] * self.ITEM_NUM).to(device), torch.LongTensor(item_batch).to(device)).detach().cpu().flatten().tolist())

            user_batch_rating_uid = zip(np.array(rate_batch), user_batch)
            batch_result = pool.map(self.test_one_user, user_batch_rating_uid)
            # batch_result = 1
            count += len(batch_result)

            for re in batch_result:
                result['precision'] += re['precision'] / n_test_users
                result['recall'] += re['recall'] / n_test_users
                result['ndcg'] += re['ndcg'] / n_test_users
                result['hit_ratio'] += re['hit_ratio'] / n_test_users
                result['auc'] += re['auc'] / n_test_users

        assert count == n_test_users
        pool.close()
        return result


