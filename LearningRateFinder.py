import math

import torch
from torch import nn
import matplotlib.pyplot as plt


class LR_Finder(object):
    """Learning rate range test.
    The learning rate range test increases the learning rate in a pre-training run
    between two boundaries in a linear or exponential manner. It provides valuable
    information on how well the network can be trained over a range of learning rates
    and what is the optimal learning rate.
    Arguments:
        model (torch.nn.Module): PyTorch model for learning rate finder.
        optimizer (torch.optim.Optimizer): Targetted optimizer.
        criterion (torch.nn.Module): Targetted loss function.
        trainloader (torch.utils.data.DataLoader) Train dataloader
    Example:
        >>> lr_finder = LR_Finder(model, optimizer, criterion, trainloader)
        >>> lr_finder.find()
        >>> lr_finder.plot_lr()
    Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    Influenced by below implementations
    1.fastai/lr_find: https://github.com/fastai/fastai
    2.How Do You Find A Good Learning Rate:
        https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """

    def __init__(
            self,
            model,
            optimizer,
            criterion,
            trainloader,
            start_lr=1e-6,
            end_lr=5,
            beta=.98):
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, 'model_lr_finder.pth.tar')
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.trainloader = trainloader
        self.step_size = 150
        self.beta = beta
        self.lr_mult = (self.end_lr / self.start_lr)**(1 / self.step_size)
        self.best_loss = 1e9
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def reset(self):
        """Restores the model and optimizer to their initial states."""
        checkpoint = torch.load(
            'model_lr_finder.pth.tar',
            map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.model.to(self.device)
        self.model.train()

    def find(self):
        self.lr_loss_stats = {"lr": [], "loss": []}
        lr = self.start_lr
        self.optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        batch_counter = 0
        self.model.eval()
        iterator = iter(self.trainloader)
        for _ in range(self.step_size):
            try:
                inputs, labels = next(iterator)
            except StopIteration:
                iterator = iter(self.trainloader)
                inputs, labels = next(iterator)
            batch_counter += 1
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            avg_loss = self.beta * avg_loss + \
                (1 - self.beta) * loss.data.item()
            smoothed_loss = avg_loss / (1 - self.beta**batch_counter)
            if smoothed_loss > 5 * self.best_loss:
                return
            if smoothed_loss < self.best_loss:
                self.best_loss = smoothed_loss
            self.lr_loss_stats['lr'].append(math.log10(lr))
            self.lr_loss_stats['loss'].append(smoothed_loss)
            loss.backward()
            self.optimizer.step()
            lr *= self.lr_mult
            self.optimizer.param_groups[0]['lr'] = lr

    def plot_lr(self):
        plt.plot(self.lr_loss_stats['lr'][10:-5],
                 self.lr_loss_stats['loss'][10:-5])
        plt.xlabel("Learning rate (log10 scale)")
        plt.ylabel("Loss")
