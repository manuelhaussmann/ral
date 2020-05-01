import torch as th
from torchvision import transforms, datasets
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
from PIL import Image
import fire

from utils import *
from model import LeNet5Closed
from policy import PolicyNet

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

def get_mnist_loader(batch_size=64, kwargs=None):
    if kwargs is None:
        kwargs = {'num_workers': 1, 'pin_memory': True}
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    Dataset = datasets.MNIST
    train_set = Dataset(f"data/mnist", train=True, transform=transform_train, download=True)
    test_set = Dataset(f"data/mnist", train=False, transform=transform_test, download=False)

    train_loader = th.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = th.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader


def actify_loader(train_loader, test_loader, n_init, balanced_init=True, n_subset=0):
    train_set = train_loader.dataset

    if balanced_init:
        if hasattr(train_set, 'train_labels'):
            n_class = len(np.unique(train_set.train_labels.numpy()))
        elif hasattr(train_set, 'labels'):
            n_class = len(np.unique(train_set.labels))
        else:
            raise NotImplementedError("No training label information available.")

        percent = 0.6
        imbalanced = True
        counter = 0
        while imbalanced and counter < 100:
            counter += 1
            if n_subset > 0:
                indices = th.from_numpy(np.random.choice(np.arange(len(train_set.train_labels)), n_subset, replace=False))
            else:
                indices = th.randperm(len(train_set))
            active_ind, inactive_ind = indices[:n_init], indices[n_init:]
            if hasattr(train_set, 'train_labels'):
                imbalanced = np.any(np.bincount(train_set.train_labels[
                                                    active_ind].numpy()) < n_init / n_class * percent)
            elif hasattr(train_set, 'labels'):
                imbalanced = np.any(np.bincount(train_set.labels[
                                                    active_ind]) < n_init / n_class * percent)
            else:
                raise NotImplementedError("No training label information available.")

        if counter == 100:
            print(f"Warning: Possibly imbalanced starting set since I capitulated after {counter} steps.")
        else:
            print(f'It took me {counter} {"step" if counter == 1 else "steps"} to get a balanced init')
    else:
        if n_subset > 0:
            indices = th.from_numpy(np.random.choice(np.arange(len(train_set.train_labels)), n_subset, replace=False))
        else:
            indices = th.randperm(len(train_set))
        active_ind, inactive_ind = indices[:n_init], indices[n_init:]

    batch_size = train_loader.batch_size
    num_workers = train_loader.num_workers
    pin_memory = train_loader.pin_memory
    if n_subset > 0:
        complete_loader = DataLoader(train_set, batch_size=batch_size, sampler=SubsetSeqSampler(indices),
                                     num_workers=num_workers, pin_memory=pin_memory)
    else:
        complete_loader = DataLoader(train_set, batch_size=batch_size, sampler=SequentialSampler(train_set),
                                     num_workers=num_workers, pin_memory=pin_memory)

    active_loader = DataLoader(train_set, batch_size=batch_size, sampler=SubsetRandomSampler(active_ind),
                                 num_workers=num_workers, pin_memory=pin_memory)
    inactive_loader = DataLoader(train_set, batch_size=batch_size, sampler=SubsetSeqSampler(inactive_ind),
                                 num_workers=num_workers, pin_memory=pin_memory)
    return indices, active_ind, inactive_ind, active_loader, inactive_loader, test_loader, complete_loader


def main(n_init = 50, epochs=30, n_labelrounds = 90, n_new = 5):
    n_label = n_init

    # Get the data
    train_loader, test_loader = get_mnist_loader(10)
    indices, active_ind, inactive_ind, active_loader, inactive_loader, test_loader, complete_loader = actify_loader(train_loader, test_loader, n_init, balanced_init=False)

    # Get the model and policy
    trainer = PolicyNet(50).to(device)
    optimizer_trainer = optim.Adam(trainer.parameters())
    model = LeNet5Closed(1, 10, large=False).to(device)
    lrate = 1e-4
    optimizer = th.optim.Adam(model.parameters(), lr=lrate)

    # First training round
    performance = dict()
    rewards = []
    for epoch in tqdm(range(epochs)):
        # lr_linear_to0(epoch, 1e-3, epochs)
        train(epoch, model, optimizer, active_loader, len(active_ind))
    performance[n_init] = validate(model, test_loader, ret_loss=True, verbose=True)
    validate(model, active_loader, ret_loss=True, verbose=True)

    t = tqdm(range(n_labelrounds), desc=f"{n_label}/{n_init + n_new*n_labelrounds}")

    # Start the Labeling rounds
    for round in t:
        n_label += n_new
        t.set_description(f"{n_label}/{n_init + n_new*n_labelrounds}")
        t.refresh() # to show immediately the update

        # Setup the state and choose new points
        Fmean, Fvar = [], []
        entros = []
        for data, target in tqdm(inactive_loader, leave=False):
            with th.no_grad():
                tmp_m, tmp_v = model(data.to(device))
                Fmean.append(tmp_m)
                Fvar.append(tmp_v)
                entros.append(get_entr_meanvar(tmp_m, tmp_v))
        entr = th.cat(entros)
        _, sort_ind = th.sort(entr, descending=True)
        n_top, n_backup = 1000, 1200
        tmp_ind = th.arange(n_backup)
        Fmean = th.cat(Fmean)[sort_ind[:n_backup]]
        Fvar = th.cat(Fvar)[sort_ind[:n_backup]]

        for _ in tqdm(range(n_new)):
            # Thinning of the state
            Fsmean, Fsvar = Fmean[:n_top:20].clone(), Fvar[:n_top:20].clone()
            really_local_act = trainer.get_newpoint(Fsmean, Fsvar)
            local_act = tmp_ind[:n_top:20][really_local_act]
            act_ind = inactive_ind[sort_ind[local_act]].view(-1) # it's a mess, but trust me...

            # change labelled
            active_ind = th.cat([active_ind, act_ind])
            active_loader.sampler.indices = active_ind
            inactive_ind = inactive_ind[inactive_ind != act_ind]
            inactive_loader.sampler.indices = inactive_ind

            Fmean = th.cat((Fmean[:local_act], Fmean[(local_act + 1):]), 0)
            Fvar = th.cat((Fvar[:local_act], Fvar[(local_act + 1):]), 0)
            tmp_ind = th.cat((tmp_ind[:local_act], tmp_ind[(local_act + 1):]))

            trainer.memory_rewards.append(th.zeros(1).to(device))
            trainer.memory_actions.append(act_ind)



        lst_acts = active_ind[-n_new:]
        lst_data = complete_loader.dataset.train_data[lst_acts]
        lst_target = th.LongTensor(complete_loader.dataset.train_labels)[lst_acts]
        #
        with th.no_grad():
            rew_data = []
            for img in lst_data:
                img = Image.fromarray(img.numpy(), mode='L')
                rew_data.append(complete_loader.dataset.transform(img))
            rew_data = th.stack(rew_data)
            model.eval()
            mu, var = model(rew_data.to(device))
            preds = Phi_var(mu, var)
            target = th.eye(10)[lst_target].to(device)
            loglikelihood = target * th.log(preds + 1e-8) + (1 - target) * th.log(1 - preds + 1e-8)
            likelihood_pre = loglikelihood.sum(1).exp()

        # Retrain net
        for epoch in tqdm(range(epochs)):
            train(epoch, model, optimizer, active_loader, len(active_ind))
        performance[n_label] = validate(model, test_loader, ret_loss=True, verbose=True)

        # Finalize rewards and update policy net
        lst_acts = active_ind[-n_new:]
        lst_data = complete_loader.dataset.train_data[lst_acts]
        lst_target = th.LongTensor(complete_loader.dataset.train_labels)[lst_acts]

        with th.no_grad():
            rew_data = []
            for img in lst_data:
                img = Image.fromarray(img.numpy(), mode='L')
                rew_data.append(complete_loader.dataset.transform(img))
            rew_data = th.stack(rew_data)
            model.eval()
            mu, var = model(rew_data.to(device))
            preds = Phi_var(mu, var)
            target = th.eye(10)[lst_target].to(device)
            loglikelihood = target * th.log(preds + 1e-8) + (1 - target) * th.log(1 - preds + 1e-8)
            likelihood_post = loglikelihood.sum(1).exp()

        for ll in range(len(likelihood_post)):
            trainer.memory_rewards[ll] += likelihood_post[ll] - likelihood_pre[ll]

        trainer.memory_rewards[-1] += (len(np.unique(lst_target.numpy()))/n_new)
        rewards.extend(trainer.memory_rewards)
        trainer.update_policy_grad(optimizer_trainer)

        tqdm.write(f"CumRew: {np.sum(np.array(rewards).ravel()).item()}, #Labeled: {len(active_ind)}")
        reduce_lr(optimizer, lr_linear_to0(round, lrate, n_labelrounds))


if __name__ == "__main__":
    # Either use fire or just run main() with the defaults
    # fire.Fire(main)
    main()
