import numpy as np
import torch
import os
import dgl
import random


class Data(object):
    def __init__(self, path, batch_size, dataset=''):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        self.dataset = dataset

        # get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.exist_users = []

        user_item_src = []
        user_item_dst = []

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    try:
                        items = [int(i) for i in l[1:]]
                    except Exception:
                        continue
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)
                    for i in l[1:]:
                        user_item_src.append(uid)
                        user_item_dst.append(int(i))

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        # training positive items corresponding to each user; testing positive items corresponding to each user
        self.train_items, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        uid, train_items = int(l), []
                        self.train_items[uid] = train_items
                        continue
                    uid, train_items = items[0], items[1:]
                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

        # construct graph from the train data and add self-loops
        user_selfs = [i for i in range(self.n_users)]
        item_selfs = [i for i in range(self.n_items)]

        self.data_dict = {
            ('user', 'user_self', 'user'): (user_selfs, user_selfs),
            ('item', 'item_self', 'item'): (item_selfs, item_selfs),
            ('user', 'ui', 'item'): (user_item_src, user_item_dst),
            ('item', 'iu', 'user'): (user_item_dst, user_item_src)
        }
        num_dict = {
            'user': self.n_users, 'item': self.n_items
        }

        self.g = dgl.heterograph(self.data_dict, num_nodes_dict=num_dict)
        self.users_features = self.get_user_feature()

    def sample(self):
        if self.batch_size <= self.n_users:
            users = random.sample(self.exist_users, self.batch_size)
        else:
            users = [random.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def get_user_feature(self):
        users_features = []
        with open(os.path.join(self.path, 'users.dat')) as f:
            if self.dataset == 'ml-1m':
                for text in f:
                    id_, gender, age, occupation, _ = text.split('::')
                    users_features.append({
                        'id': int(id_) - 1,
                        'gender': gender,
                        'age': age,
                        'occupation': occupation
                    })
            elif self.dataset == 'douban':
                for text in f:
                    id_, location = text.split()
                    users_features.append({
                        'id': int(id_),
                        'location': location
                    })
        return users_features

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (
            self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))


def process_data_ml_1m():
    path = '../data/recommendation/ml-1m/'
    f_train = open(os.path.join(path, 'train.txt'), 'w')
    f_test = open(os.path.join(path, 'test.txt'), 'w')
    tmp_user = 1
    tmp_item_train = []
    tmp_item_test = []
    latest_time_stamp = 0
    tmp = 0
    with open(os.path.join(path, 'ratings.dat')) as f1:
        for text in f1:
            user_id, movie_id, rating, timestamp = [int(_) for _ in text.split('::')]
            if user_id == tmp_user:
                if latest_time_stamp < timestamp:
                    tmp = str(movie_id-1)
                    latest_time_stamp = timestamp
                tmp_item_train.append(str(movie_id - 1))
            else:
                tmp_item_train.remove(tmp)
                tmp_item_test.append(tmp)
                text_train = str(tmp_user - 1) + ' ' + ' '.join(tmp_item_train)
                text_test = str(tmp_user - 1) + ' ' + ' '.join(tmp_item_test)
                if len(text_train) > 1:
                    f_train.writelines(text_train + '\n')
                if len(text_test) > 1:
                    f_test.writelines(text_test + '\n')
                tmp_user += 1
                tmp_item_train = []
                tmp_item_test = []
    text_train = str(tmp_user - 1) + ' ' + ' '.join(tmp_item_train)
    text_test = str(tmp_user - 1) + ' ' + ' '.join(tmp_item_test)
    if len(text_train) > 1:
        f_train.writelines(text_train + '\n')
    if len(text_test) > 1:
        f_test.writelines(text_test + '\n')
    f_train.close()
    f_test.close()


def process_data_douban_book():
    path = '../data/recommendation/douban/'
    f_data = open(os.path.join(path, 'user_book.dat'))
    f_location = open(os.path.join(path, 'user_location.dat'))
    tmp_item_train = {}
    tmp_item_test = {}
    tmp_data = {}
    tmp_location = {}
    rec_data = {}
    user_map = {}
    location_map = {}
    for text in f_data:
        user, book, rate = [int(_)-1 for _ in text.split()]
        if user not in tmp_data.keys():
            tmp_data[user] = [str(book)]
        else:
            tmp_data[user].append(str(book))
    re_number = 0
    for text in f_location:
        user, location = [int(_)-1 for _ in text.split()]
        if user not in tmp_location.keys():
            if location not in location_map.keys():
                location_map[location] = re_number
                re_number += 1
            tmp_location[user] = str(location_map[location])

    re_number = 0
    for i in range(len(tmp_data)):
        if len(tmp_data[i]) < 20 or i not in tmp_location.keys():
            del(tmp_data[i])
        else:
            user_map[i] = re_number
            re_number += 1
    for key in user_map.keys():
        rec_data[user_map[key]] = tmp_data[key]
    f_data.close()

    f_train = open(os.path.join(path, 'train.txt'), 'w')
    f_test = open(os.path.join(path, 'test.txt'), 'w')
    for i in range(len(rec_data)):
        train = rec_data[i][:-1]
        test = [rec_data[i][-1]]
        text_train = str(i) + ' ' + ' '.join(train)
        text_test = str(i) + ' ' + ' '.join(test)
        if len(text_train) > 1:
            f_train.writelines(text_train + '\n')
        if len(text_test) > 1:
            f_test.writelines(text_test + '\n')
    f_train.close()
    f_test.close()

    # user attribute
    user_location = {}
    f_user_location = open(os.path.join(path, 'users.dat'), 'w')
    for key in tmp_location.keys():
        if key in user_map.keys():
            user_location[user_map[key]] = tmp_location[key]
    for i in range(len(user_location)):
        text_user_location = str(i) + ' ' + user_location[i]
        f_user_location.writelines(text_user_location + '\n')

    f_user_location.close()
    f_location.close()


class PrivacySettingGeneration(Data):
    def __init__(self, path, batch_size, n_privacy=3, p_random=0.2):
        # serial: gender, age, occupation
        super(PrivacySettingGeneration, self).__init__(path, batch_size)
        self.n_privacy = n_privacy  # the number of private attributes
        self.p = p_random  # the probability that the user wish to protect one attribute
        # the mask of private user, [num_user, num_pri], 1: protected; 0: not-protected
        self.private_user_mask = self.random_privacy_sample()
        # the number of those willing to share all information
        self.no_private_user_num = np.squeeze(np.argwhere(np.all(self.private_user_mask == 0, axis=-1))).tolist()
        # self.users_features = self.get_user_feature()  # get user feature list
        # self.user_subgraph_g = self.get_user_subgraph()
        np.save(os.path.join(path, 'private_user_mask_{}'.format(p_random)), self.private_user_mask)
        pass

    def random_privacy_sample(self):
        np.random.seed(1)
        return np.random.choice([0, 1], size=(self.n_users, self.n_privacy), p=[1-self.p, self.p])


class DataGeneratorForTest(object):
    def __init__(self, data_generator):
        self.n_users, self.n_items = data_generator.n_users, data_generator.n_items
        self.n_train, self.n_test = data_generator.n_train, data_generator.n_test
        self.train_items = data_generator.train_items
        self.test_set = data_generator.test_set


def main():
    # process_data_ml_1m()
    # movielens = Data('../data/recommendation/ml-1m/', 1024)
    # privacy_data = PrivacySettingGeneration('../data/recommendation/ml-1m/', 1024, p_random=0.2)
    process_data_douban_book()
    pass


if __name__ == '__main__':
    main()
