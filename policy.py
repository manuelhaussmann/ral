import torch.nn as nn
from model import CLTLayer
from torch.distributions import Categorical

from utils import *


class PolicyNet(nn.Module):
    def __init__(self, n_points):
        super(PolicyNet, self).__init__()

        self.fc1 = CLTLayer(10 * n_points, 256, isinput=True)
        self.fc2 = CLTLayer(256, n_points, isoutput=True)

        self.gamma = 0.95

        self.loss_archive = []
        self.memory_rewards = []
        self.memory_actions = []
        self.memory_logprob = []
        self.baseline = []

    def forward(self, inp_mean, inp_var):
        inp_mean, inp_var = inp_mean.view(-1), inp_var.view(-1)
        mu_h1, var_h1 = self.fc1(inp_mean, inp_var)
        mu_pred, var_pred = self.fc2(mu_h1, var_h1)

        return mu_pred, var_pred

    def get_newpoint(self, inp_mean, inp_var):
        polmean, polvar = self.forward(inp_mean, inp_var)
        # probit and normalize
        probs = Phi_var(polmean, polvar)
        m = Categorical(probs)
        act = m.sample()
        self.memory_logprob.append(m.log_prob(act))
        return act

    def get_KL(self):
        for l in [self.fc1, self.fc2]:
            l.alpha = 100
        return sum([l.KL() for l in [self.fc1, self.fc2]])

    def update_policy_grad(self, optimizer):
        losses = []
        rewards = []
        R = 0
        for r in self.memory_rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = th.cat(rewards)

        if len(self.baseline):
            b = th.stack(self.baseline).mean(0)
        else:
            b = 0
        for log_prob, r in zip(self.memory_logprob, rewards):
            losses.append(-(r - b) * log_prob.view(-1))

        optimizer.zero_grad()
        loss = th.cat(losses).sum() + self.get_KL()
        loss.backward()
        optimizer.step()
        self.loss_archive.append(loss.data)

        self.baseline.extend(self.memory_rewards)

        del self.memory_logprob[:]
        del self.memory_rewards[:]
        del self.memory_actions[:]
