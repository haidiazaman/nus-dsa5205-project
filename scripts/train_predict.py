import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alive_progress import alive_it

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_selection import SelectKBest, f_regression

from .models import FCN
from .losses import FocalLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import time

def train_val_test_split(X, y, train_size, val_size, test_size):
    """
    split X,y according to sizes
    """
    train_size, val_size, test_size = int(len(X) * train_size), int(len(X) * val_size), int(len(X) * test_size)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
    
    return X_train,y_train,X_val,y_val,X_test,y_test

def shuffle_train_val_test_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    return X_train,y_train,X_val,y_val,X_test,y_test

######################################################
################### PYTORCH CODE ####################
######################################################

def train(
    device,
    lr,
    epochs,
    scheduler_factor,
    scheduler_patience,
    early_stopping_limit,
    class_weights,
    criterion,
    train_loader,
    val_loader
):

    if criterion=="cross":
        criterion = nn.CrossEntropyLoss()
    elif criterion=="focal":
        class_weights = torch.tensor(class_weights, device=device) # change to 1-%of that class
        criterion = FocalLoss(alpha=class_weights) 

    model = FCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=scheduler_factor, patience=scheduler_patience)

    model = model.to(device)
    criterion = criterion.to(device)
    
    early_stop_count = 0
    min_val_loss = float('inf')
    train_losses,train_accs=[],[]
    val_losses,val_accs=[],[]
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f'epoch {epoch}')
        model.train()
    
        train_running_loss = []
        train_running_acc = []
        for batch in train_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch.long())
            train_running_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            train_running_acc.append(get_accuracy(outputs,y_batch))
    
        train_loss = np.mean(train_running_loss)
        train_losses.append(train_loss)
        train_acc = np.mean(train_running_acc)
        train_accs.append(train_acc)
    
        # Validation
        model.eval()
        val_running_loss = []
        val_running_acc = []
        with torch.no_grad():
            for batch in val_loader:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch.long())
                val_running_loss.append(loss.item())
                val_running_acc.append(get_accuracy(outputs,y_batch))
    
        val_loss = np.mean(val_running_loss)
        val_losses.append(val_loss)
        val_acc = np.mean(val_running_acc)
        val_accs.append(val_acc)
    
        scheduler.step(val_loss)
    
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), 'fcn.pt')
            print(f'model epoch {epoch} saved as fcn.pt')
            early_stop_count = 0
        else:
            early_stop_count += 1
    
        if early_stop_count >= early_stopping_limit:
            print("Early stopping!")
            break
    
        time_taken = round(time.time()-epoch_start_time,1)
        print(f"Epoch {epoch + 1}/{epochs}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, time_taken: {time_taken}")
        print(f"train acc: {round(train_acc*100,2)}, val_acc: {round(val_acc*100,2)}")
    
    return model, train_losses, val_losses, train_accs, val_accs


def get_accuracy(outputs, gt):
    softmax_outputs=nn.Softmax(dim=1)(outputs)
    # Convert softmax outputs to predicted classes
    _, predictions = torch.max(softmax_outputs, dim=1)
    # Compare predictions with ground truth
    correct = (predictions == gt).sum().item()
    # Calculate accuracy
    acc = correct / gt.size(0)
    return acc


def plot_loss_acc(train_losses, val_losses, train_accs, val_accs):
    fig,ax = plt.subplots(1,2,figsize=(10,6))

    plt.subplot(1,2,1)
    plt.plot(train_losses,label='train loss')
    plt.plot(val_losses,label='val loss')
    plt.title("loss curves")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_accs,label='train acc')
    plt.plot(val_accs,label='val acc')
    plt.title("acc curves")
    plt.legend()

    plt.savefig("fcn.png")
    plt.show()


def evaluate(X_test,y_test):
    criterion = nn.CrossEntropyLoss()

    # load best pretraining model
    best_model_path = 'fcn.pt'
    model = FCN()
    model.load_state_dict(torch.load(best_model_path,map_location=device))
    model.eval()

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    preds = model(X_test.to(device))
    softmax_preds=nn.Softmax(dim=1)(preds)
    test_loss = criterion(softmax_preds,y_test.long())
    print("test loss: ",test_loss)
    test_acc = round(get_accuracy(softmax_preds,y_test)*100,2)
    print("test acc: ",test_acc)
    
    return preds


# # check lr scheduler
# model = FCN()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# epochs = 100
# scheduler_factor=0.75
# scheduler_patience=3
# early_stopping_limit = 10
# scheduler = ReduceLROnPlateau(optimizer, 'min', factor=scheduler_factor, patience=scheduler_patience)

# lrs=[]
# for _ in range(epochs):
#     lrs.append(scheduler.get_last_lr())
#     scheduler.step(0.1)

# plt.plot(lrs)
# plt.show()


## code to get class weights
# nums = get_numpy_value_counts(y_train)[:,1]
# nums = 1-nums/sum(nums)
# nums