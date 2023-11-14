import torch
import torch.nn as nn
import torch.nn.functional as F


class NewGenderDiscriminator(nn.Module):
    def __init__(self, embed_dim):
        super(NewGenderDiscriminator, self).__init__()
        self.embed_dim = int(embed_dim)

        self.out_dim = 2
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

        self.net = nn.Sequential(
            nn.BatchNorm1d(num_features=self.embed_dim),
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 4), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            # nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 4), bias=True),
            # nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 2), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            # nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 2), bias=True),
            # # nn.BatchNorm1d(num_features=self.embed_dim),
            # nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            # nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
            # nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
        )

    def forward(self, ents_emb):
        scores = self.net(ents_emb)
        output = F.log_softmax(scores, dim=1)
        # loss = self.criterion(output.squeeze(), A_labels)
        return output

    def predict(self, ents_emb):
        with torch.no_grad():
            scores = self.net(ents_emb)
            output = F.log_softmax(scores, dim=1)
            preds = output.max(1, keepdim=True)[1]  # get the index of the max
            return output.squeeze(), preds

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))


class AgeDiscriminator(nn.Module):
    def __init__(self, embed_dim):
        super(AgeDiscriminator, self).__init__()
        self.embed_dim = int(embed_dim)
        self.criterion = nn.NLLLoss()
        self.out_dim = 7
        self.drop = 0.5

        self.net = nn.Sequential(
            # nn.BatchNorm1d(num_features=self.embed_dim),
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 2),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 4), bias=True),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 2), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 2),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 2), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
        )

    def forward(self, ents_emb):
        scores = self.net(ents_emb)
        output = F.log_softmax(scores, dim=1)
        # loss = self.criterion(output.squeeze(), A_labels)
        return output

    def predict(self, ents_emb):
        with torch.no_grad():
            scores = self.net(ents_emb)
            output = F.log_softmax(scores, dim=1)
            preds = output.max(1, keepdim=True)[1]  # get the index of the max
            return output.squeeze(), preds

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))

    def create_kl_divergence(self, u_g_embeddings):
        predict = self.forward(u_g_embeddings)
        target = (torch.ones(predict.shape) / predict.shape[1]).to(u_g_embeddings.device)
        loss = F.kl_div(predict, target, reduction='batchmean')
        return loss


class OccupationDiscriminator(nn.Module):
    def __init__(self, embed_dim):
        super(OccupationDiscriminator, self).__init__()
        self.embed_dim = int(embed_dim)
        self.criterion = nn.NLLLoss()

        self.out_dim = 21
        self.drop = 0.3

        self.net = nn.Sequential(
            # nn.BatchNorm1d(num_features=self.embed_dim),
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 2),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim*4),bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 2),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim * 4), int(self.embed_dim*2),bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim*2), int(self.embed_dim*2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim*2), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim), int(self.embed_dim/2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim / 2), self.out_dim,bias=True)
            )

    def forward(self, ents_emb):
        scores = self.net(ents_emb)
        output = F.log_softmax(scores, dim=1)
        # loss = self.criterion(output.squeeze(), A_labels)
        return output

    def predict(self, ents_emb):
        with torch.no_grad():
            scores = self.net(ents_emb)
            output = F.log_softmax(scores, dim=1)
            preds = output.max(1, keepdim=True)[1]  # get the index of the max
            return output.squeeze(), preds

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))

    def create_kl_divergence(self, u_g_embeddings):
        predict = self.forward(u_g_embeddings)
        target = (torch.ones(predict.shape) / predict.shape[1]).to(u_g_embeddings.device)
        loss = F.kl_div(predict, target, reduction='batchmean')
        return loss


class GenderDiscriminator(nn.Module):
    def __init__(self, embed_dim):
        super(GenderDiscriminator, self).__init__()
        self.embed_dim = int(embed_dim)

        self.out_dim = 1
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.drop = 0.5

        self.net = nn.Sequential(
            # nn.BatchNorm1d(num_features=self.embed_dim),
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 4), bias=True),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=self.drop),
            # nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 4), bias=True),
            # nn.LeakyReLU(0.2),
            # nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 2), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=self.drop),
            # nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 2), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim),
            # nn.LeakyReLU(0.2),
            # nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=self.drop),
            # nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
            # nn.LeakyReLU(0.2),
            # nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
        )

    def forward(self, ents_emb):
        scores = self.net(ents_emb)
        output = torch.sigmoid(scores)
        output = output.clamp(1e-5, 1 - 1e-5)
        # loss = self.criterion(output.squeeze(), A_labels)
        return output

    def predict(self, ents_emb):
        with torch.no_grad():
            scores = self.net(ents_emb)
            output = torch.sigmoid(scores)
            preds = (scores > torch.Tensor([0.5]).cuda()).float() * 1  # get the index of the max
            return output.squeeze(), preds

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))

    def create_kl_divergence(self, u_g_embeddings):
        predict = self.forward(u_g_embeddings)
        predict = torch.hstack((predict, 1 - predict))
        target = (torch.ones(predict.shape) / predict.shape[1]).to(u_g_embeddings.device)
        predict = torch.log(predict)
        loss = F.kl_div(predict, target, reduction='batchmean')
        return loss


class LocationDiscriminator(nn.Module):
    def __init__(self, embed_dim):
        super(LocationDiscriminator, self).__init__()
        self.embed_dim = int(embed_dim)
        self.out_dim = 453
        self.criterion = nn.NLLLoss()
        self.drop = 0.5

        self.net = nn.Sequential(
            # nn.BatchNorm1d(num_features=self.embed_dim),
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 2),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim*4),bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 2),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim * 4), int(self.embed_dim*2),bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim*2), int(self.embed_dim*2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim*2), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim), int(self.embed_dim/2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim / 2), self.out_dim,bias=True)
            )


    def forward(self, ents_emb):
        scores = self.net(ents_emb)
        output = F.log_softmax(scores, dim=1)
        # loss = self.criterion(output.squeeze(), A_labels)
        return output

    def predict(self, ents_emb):
        with torch.no_grad():
            scores = self.net(ents_emb)
            output = F.log_softmax(scores, dim=1)
            preds = output.max(1, keepdim=True)[1]  # get the index of the max
            return output.squeeze(), preds

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))

    def create_kl_divergence(self, u_g_embeddings):
        predict = self.forward(u_g_embeddings)
        target = (torch.ones(predict.shape) / predict.shape[1]).to(u_g_embeddings.device)
        loss = F.kl_div(predict, target, reduction='batchmean')
        return loss


if __name__ == '__main__':
    pass