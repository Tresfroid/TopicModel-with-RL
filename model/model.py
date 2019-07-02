import torch
from torch import nn
import torch.nn.functional as F

class Vocab_Selector(nn.Module):
    def __init__(self):
        super(Vocab_Selector, self).__init__()

    def sampler(self, state, policy_prob, cuda):
        batchsize, dv = state.size()
        policy_prob = policy_prob
        random_vector = torch.rand(1, dv).cuda(cuda)
        threshold = (1 - policy_prob) * 3000 / 4999
        mask = (random_vector > threshold) + 0
        policy_action = mask.float()
        policy_action_mask = policy_action.detach()
        total_prob = policy_prob * policy_action + (1 - policy_prob) * (1 - policy_action)  # 1*5000
        prob_log = total_prob.log()  # 1*5000

        return policy_action_mask, prob_log

class TopicModel(nn.Module):
    def __init__(self, selector, d_v, d_e, d_t, encoder_layers=1, generator_layers=4,
                 encoder_shortcut=False, generator_shortcut=False, generator_transform=None):

        super(TopicModel, self).__init__()

        self.d_v = d_v  # vocabulary size
        self.d_e = d_e  # dimensionality of encoder
        self.d_t = d_t  # number of topics
        self.encoder_layers = encoder_layers
        self.generator_layers = generator_layers

        # set various options
        self.generator_transform = generator_transform  # transform to apply after the generator
        self.encoder_shortcut = encoder_shortcut
        self.generator_shortcut = generator_shortcut

        self.en1_fc = nn.Linear(self.d_v, self.d_e)
        self.en2_fc = nn.Linear(self.d_e, self.d_e)
        self.en_drop = nn.Dropout(0.2)
        self.mean_fc = nn.Linear(self.d_e, self.d_t)
        #         self.mean_bn = nn.BatchNorm1d(self.d_t)
        self.logvar_fc = nn.Linear(self.d_e, self.d_t)
        #         self.logvar_bn = nn.BatchNorm1d(self.d_t)

        self.generator1 = nn.Linear(self.d_t, self.d_t)
        self.generator2 = nn.Linear(self.d_t, self.d_t)
        self.generator3 = nn.Linear(self.d_t, self.d_t)
        self.generator4 = nn.Linear(self.d_t, self.d_t)

        self.r_drop = nn.Dropout(0.2)

        self.de = nn.Linear(self.d_t, self.d_v)
        #         self.de_bn = nn.BatchNorm1d(self.d_v)

        # policy gradient:
        self.selector = selector

    def select_data(self, x, x_indices, policy_prob, cuda):

        policy_action_mask, log_prob = self.selector.sampler(x, policy_prob, cuda)
        #         print('selected vocabulary number is: '+ str(policy_action_mask.sum()))
        x_new = torch.mul(policy_action_mask.t(), x.t()).t()  # 点乘 5000*1  *  5000*batchsize
        x_indices_new = torch.mul(policy_action_mask.t(), x_indices.t()).t()
        #         print(policy_action_mask,x,x_new,x_indices,x_indices_new)

        return x_new, x_indices_new, policy_action_mask, log_prob

    def encoder(self, x):
        if self.encoder_layers == 1:
            pi = F.relu(self.en1_fc(x))
            if self.encoder_shortcut:
                pi = self.en_drop(pi)
        else:
            pi = F.relu(self.en1_fc(x))
            pi = F.relu(self.en2_fc(pi))
            if self.encoder_shortcut:
                pi = self.en_drop(pi)

        #         mean = self.mean_bn(self.mean_fc(pi))
        #         logvar = self.logvar_bn(self.logvar_fc(pi))
        mean = self.mean_fc(pi)
        logvar = self.logvar_fc(pi)
        return mean, logvar

    def sampler(self, mean, logvar, cuda):
        eps = torch.randn(mean.size()).cuda(cuda)
        sigma = torch.exp(logvar)
        h = sigma.mul(eps).add_(mean)
        return h

    def generator(self, h):
        if self.generator_layers == 0:
            r = h
        elif self.generator_layers == 1:
            temp = self.generator1(h)
            if self.generator_shortcut:
                r = F.tanh(temp) + h
            else:
                r = temp
        elif self.generator_layers == 2:
            temp = F.tanh(self.generator1(h))
            temp2 = self.generator2(temp)
            if self.generator_shortcut:
                r = F.tanh(temp2) + h
            else:
                r = temp2
        else:
            temp = F.tanh(self.generator1(h))
            temp2 = F.tanh(self.generator2(temp))
            temp3 = F.tanh(self.generator3(temp2))
            temp4 = self.generator4(temp3)
            if self.generator_shortcut:
                r = F.tanh(temp4) + h
            else:
                r = temp4

        if self.generator_transform == 'tanh':
            return self.r_drop(F.tanh(r))
        elif self.generator_transform == 'softmax':
            return self.r_drop(F.softmax(r)[0])
        elif self.generator_transform == 'relu':
            return self.r_drop(F.relu(r))
        else:
            return self.r_drop(r)

    def decoder(self, r):
        #         p_x_given_h = F.softmax(self.de_bn(self.de(r)))
        p_x_given_h = F.softmax(self.de(r))
        return p_x_given_h

    def continuous_parameters(self):
        for name, param in self.named_parameters():
            if not name.startswith("selector"):
                yield param

    def discrete_parameters(self):
        for name, param in self.named_parameters():
            if name.startswith("selector"):
                yield param

    def forward(self, x, x_indices, policy_prob, cuda):
        x_new, x_indices_new, policy_action_mask, log_prob = self.select_data(x, x_indices, policy_prob, cuda)
        mean, logvar = self.encoder(x_new)
        h = self.sampler(mean, logvar, cuda)
        r = self.generator(h)
        p_x_given_h = self.decoder(r)

        return mean, logvar, p_x_given_h, x_new, x_indices_new, policy_action_mask, log_prob