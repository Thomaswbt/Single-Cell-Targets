from data_util import RegressionDataset
from model import RegressionModel
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import argparse
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle
from scipy.stats import spearmanr
from torch.utils.tensorboard import SummaryWriter

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train(model, optimizer, loss_fn, train_loader, val_loader, epochs, device):
    # train the model with loss visualization through tensorboard
    # use validation set to early stop
    # save the best model
    early_stopping = EarlyStopping(patience=10, verbose=True)
    writer = SummaryWriter()
    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        # use progress bar to show the training progress and print loss
        progress_bar = tqdm(train_loader, desc='Epoch {}/{}'.format(epoch+1, epochs), ascii=True)
        for batch_idx, (data, target) in enumerate(progress_bar):
            data = data.to(device)
            target = target.to(device).squeeze().float()
            optimizer.zero_grad()
            output = model(data).squeeze()
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())
            writer.add_scalar('RMSE/train', np.sqrt(loss.item()), epoch * len(train_loader) + batch_idx)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device).squeeze().float()
                output = model(data).squeeze()
                val_loss = loss_fn(output, target)
                val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)
        writer.add_scalar('RMSE/val', np.sqrt(val_loss), epoch)
        
        early_stopping(np.sqrt(val_loss), model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    model.load_state_dict(torch.load('checkpoint.pt'))
    return model

def evaluate(model, test_loader, device):
    # evaluate the model with test set
    # return the AUC score and ROC curve
    model = model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device).squeeze().float()
            output = model(data).squeeze()
            y_true.extend(target.cpu().numpy())
            y_pred.extend(output.cpu().numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # calculate RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    # calculate spearman r
    spearman_r = spearmanr(y_true, y_pred)[0]
    # print the idx of highest y_pred
    print('The score of highest y_pred: ', y_pred[np.argpartition(y_pred, -10)[-10:]])
    return rmse, spearman_r, np.argpartition(y_pred, -10)[-10:]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ours', action='store_true', help='use our model')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    # fix seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.ours:
        X_train = np.load('data/dataset/GBM_gene_embed_1e-2_train.npy')
        y_train = np.load('data/dataset/overall_scores_train.npy')
        X_test = np.load('data/dataset/GBM_gene_embed_1e-2_test.npy')
        y_test = np.load('data/dataset/overall_scores_test.npy')
    else:
        X_train = np.load('data/dataset/GBM_gene_embed_0_train.npy')
        y_train = np.load('data/dataset/overall_scores_train.npy')
        X_test = np.load('data/dataset/GBM_gene_embed_0_test.npy')
        y_test = np.load('data/dataset/overall_scores_test.npy')
    dataset = RegressionDataset(X_train, y_train)
    # split the dataset into train, validation set, and test set
    train_size = int(0.9 * len(dataset))
    val_size = int(0.1 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    test_dataset = RegressionDataset(X_test, y_test)
    model = RegressionModel(input_dim=X_train.shape[1], hidden_dim=args.hidden_dim, output_dim=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # mse loss
    loss_fn = nn.MSELoss()
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    model = train(model, optimizer, loss_fn, train_loader, val_loader, args.epochs, args.device)
    rmse, r, idx = evaluate(model, test_loader, args.device)
    print('RMSE: {}'.format(rmse))
    print('Spearman r: {}'.format(r))
    
    gene_test = pickle.load(open('data/dataset/gene_test.pkl', 'rb'))
    print('The gene of highest y_pred: ', np.array(gene_test)[idx])