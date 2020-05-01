import math
import torch as th
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

def Phi(X):
    return 0.5 * (1 + th.erf(X/math.sqrt(2)))


def Phi_var(mu, var):
    return Phi(mu/(1 + var).sqrt())


def ProbitLoss_var(mu, var, target):
    phi = Phi_var(mu, var)
    loss = target * th.log(phi + 1e-8) + (1 - target) * th.log(1 - phi + 1e-8)
    return -loss.sum(1).mean(0)


def H(p):
    return - p * th.log(p + 1e-8) - (1 - p) * th.log(1 - p + 1e-8)


def get_entr_meanvar(Xmean, Xvar):
    return H(Phi_var(Xmean, Xvar)).sum(1).cpu()


def extract_features(data, net, S=10):
    net.eval()
    with th.no_grad():
        mu_pred, var_pred = net(data.to(device))
        collect = th.stack([F.softmax(mu_pred + var_pred.sqrt() * th.randn_like(mu_pred), 1) for _ in range(S)], 2)
    net.train()
    return collect[:, :, :, None]


def entropy(p):
    "Get entropy in nats with epsilon stability"
    if len(p.shape) == 1:
        return (-p.mul(th.log(p + 1e-8))).sum()
    elif len(p.shape) == 2:
        return (-p.mul(th.log(p + 1e-8))).sum(1)
    else:
        raise NotImplementedError(f"Expected len(p.shape) \leq 2, but got {len(p.shape)}")


def BALD_all(loader, network, S=10):
    network.eval()
    with th.no_grad():
        utility = [BALD(data, target, network, S) for (data, target) in tqdm(loader, total=len(loader),
                                                                             desc='Compute BALD', ncols=80,
                                                                             leave=False)]
    return th.cat(utility)


def expand_samples(X, S=10):
    return th.cat([x[None].expand(S, *X.size()[1:]) for x in X])


def BALD(data, target, network, S=10):
    data_exp = expand_samples(data.to(device), S=S)
    preds = network(data_exp)
    preds_soft = F.softmax(preds, 1)

    # Fst summand
    fst = entropy(th.stack([preds_soft[(i * S):((i + 1) * S)].mean(0).view(-1) for i in range(len(target))]))
    # Snd summand
    snd = th.cat([entropy(preds_soft)[(i * S):((i + 1) * S)].mean().view(-1) for i in range(len(target))])
    return fst - snd




def gen_state(net, n_preds, inact_loader, norm=False):
    with th.no_grad():
        Xdata = [extract_features(data, net, S=n_preds) for data, _ in tqdm(inact_loader, leave=False)]
        Xdata = th.cat(Xdata).transpose(1, 2)
        if norm:
            for i in range(10):
                Xdata[:,i,:] = (Xdata[:,i,:] - Xdata[:,i,:].mean())/ Xdata[:,i,:].std()
    return Xdata


def train(epoch, model, optimizer, data_loader, n_data, verbose=False):
    model.train()
    train_loss = 0.0
    for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader),
                                          desc='Train epoch=%d' % epoch, ncols=80, leave=False):
        data = data.to(device)
        target = th.eye(10)[target].float().to(device)
        optimizer.zero_grad()
        loss = model.loss(data, target, n_data)
        loss.backward()
        optimizer.step()
        train_loss += loss.data.cpu().numpy()
        if np.isnan(float(loss.data)):
            raise ValueError('loss is nan while validating')
    if verbose:
        tqdm.write(f'Train Epoch: {epoch} \tLoss: {train_loss / len(data_loader):.6f}')
    return train_loss/len(data_loader)


def validate(model, data_loader, ret_loss=False, verbose=False):
    with th.no_grad():
        model.eval()
        test_loss = 0.0
        correct = 0
        n_tested = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            if len(target) == 1:
                output = output.view(1,-1)
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().float().sum()
            n_tested += data.size(0)
        if verbose:
            tqdm.write(f'Accuracy: {correct}/{n_tested} ({100. * correct / n_tested:.0f}%)')
        if ret_loss:
            return (1.0 - correct / n_tested).item(), test_loss #.item()
        else:
            return (1.0 - correct / n_tested).item()


from torch.utils.data.sampler import Sampler
class SubsetSeqSampler(Sampler):
    """Return the subset of datapoints in order.
    Arguments:
        indices (list): a list of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices.numpy())

    def __len__(self):
        return len(self.indices)



def lr_linear_to0(epoch, lr, max_epoch):
    return lr * max(0, (max_epoch - epoch) / max_epoch)


def reduce_lr(optimizer, new_lr=None, factor=1.0):
    """
    Change the learning rate of the optimizer_net. An explicit new rate overrules a factor
    """
    if new_lr is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= factor