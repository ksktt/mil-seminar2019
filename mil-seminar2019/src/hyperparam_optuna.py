from argparse import Namespace
from torch.nn import Module
import torchvision.transforms as transforms
from torchvision.transforms import Compose
from task import Task
#from trainer import Trainer
from torch.utils.data.dataset import Subset
import optuna

from data import CifarDataset

def wrap_scheduler(initial_lr, first_time, second_time):
    accuracy = 0
    max_accuracy = 0
    first_epoch = int(epochs_num*first_time)
    second_epoch = int(epochs_num*second_time)
    #third_epoch = int(epochs_num*third_time)
    for epoch in range(1, epochs_num+1):
        if epoch <= first_epoch:
            return 10**(-2.0+0.4*epoch)*initial_lr
        x = initial_lr * 5
        if epoch >= second_epoch:
            x /= 5
        #if epoch >= third_epoch:
        #    x /= 5

        optimizer = optim.Adadelta(model.parameters(), lr=x)

        model.train()
        for batch_idx, train_data, target in iter(train_loader):
            train_data, target = train_data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(train_data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for val_data, target in val_loader:
                val_data, target = val_data.to(device), target.to(device)
                output = model(val_data)
                # sum up batch loss
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(val_loader.dataset)

        accuracy = correct / len(val_loader.dataset)

        if accuracy > max_accuracy:
            max_accuracy = accuracy

    return 1 - max_accuracy

def objective(trial):
    initial_lr = trial.suggest_uniform('initial_lr', 0.001, 0.1)
    first_time = trial.suggest_uniform('first_time', 0.1, 0.2)
    second_time = trial.suggest_uniform('second_time', 0.4, 0.6)
    return wrap_scheduler(initial_lr, first_time, second_time)

def wrap_lr(initial_lr, first_time, second_time,epoch):
    if epoch <= first_time:
        return 10**(-2.0+0.4*epoch)*initial_lr
    x = initial_lr * 5
    if epoch >= second_time:
        x /= 5
    return x

class Hyperparams:
    ''' Hyperparameter class

    This class contains hyperparameter values.

    Please add new attributes or methods if needed.

    '''

    def __init__(self):
        #self.hyperparam1 = 256  #warmup_step
        #self.hyperparam2 = 4000 #d_model
        self.hyperparam0 = 0.01 #init_lr
        self.hyperparam1 = 0.02 #first_time
        self.hyperparam2 = 0.03 #second_time

def tune_hyperparams(args: Namespace, task: Task, preprocess_func: Compose, model: Module) -> Hyperparams:
    ''' Tune hyperparameters

    Given task, preprocess function, and model, this method returns tuned hyperparameters.

    '''

    # TODO: Implement hyperparameter tuning
    """
    pbounds = {
        'initial_lr': (0.001, 0.1),
        'first_time': (0.1, 0.25),
        'second_time': (0.35, 0.6),
        'third_time': (0.65, 0.9)
    }
    print("start tuning ...")
    bo = BayesianOptimization(f=wrap_scheduler, pbounds=pbounds)
    # 最大化する
    bo.maximize(init_points=5, n_iter=15)
    # 結果を出力する
    #print(bo.max)
    """

    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    Hyperparams.hyperparam0 = study.best_params['initial_lr']
    Hyperparams.hyperparam1 = study.best_params['first_time']
    Hyperparams.hyperparam2 = study.best_params['second_time']
    #Hyperparams.hyperparam0 = bo.max['params']['initial_lr']
    #Hyperparam.hyperparam1 = bo.max['params']['first_time']
    #Hyperparams.hyperparam2 = bo.max['params']['second_time']
    #Hyperparams.hyperparam3 = bo.max['params']['third_time']

    #hyperparams = Hyperparams()

    param_list = [study.best_params['initial_lr'], study.best_params['first_time'], study.best_params['second_time']]

    return param_list

if __name__ == '__main__':
    '''
    Indivisual test
    Copied from
        https://github.com/pytorch/examples/tree/master/mnist
    '''

    import argparse
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.optim.lr_scheduler import StepLR
    import os

    class Net(Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)
            self.fc1 = nn.Linear(12544, 128)
            self.fc2 = nn.Linear(128, 100)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--task', '-t', type=str, help='Task name')
    parser.add_argument('--gpu', '-g', type=int,
                        nargs='?', help='GPU ids to use')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    epochs_num = args.epochs

    def train(args, model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def test(args, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                # sum up batch loss
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        #return correct/len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    ############################################################
    # Instantiate task object
    task = Task(args.task)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    n_samples = len(train_loader.dataset)
    train_size = int(n_samples / 5)
    subset1_indices = list(range(0,train_size))
    subset2_indices = list(range(train_size,n_samples))

    train_dataset = Subset(train_loader.dataset, subset1_indices)
    val_dataset   = Subset(train_loader.dataset, subset2_indices)

    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Dummy preprocess and model
    preprocess_func = Compose([transforms.ToTensor(), ])
    model = Net().to(device)

    hyperparam = tune_hyperparams(args, task, preprocess_func, model)
    ############################################################

    warmup_lr = wrap_lr(hyperparam.hyperparam0, hyperparam.hyperparam1, hyperparam.hyperparam2, epoch)
    optimizer = optim.Adadelta(model.parameters(), lr=warmup_lr)
    train(args, model, device, train_loader, optimizer, epoch)
    test(args, model, device, test_loader)
