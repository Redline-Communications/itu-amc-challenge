#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import os.path
import time
from torch import nn
import os

from radioml import radioml_18_dataset
from model import model as model

start_time = time.time()

gpu = 0
if torch.cuda.is_available():
    torch.cuda.device(gpu)
    print("Using GPU %d" % gpu)
else:
    gpu = None
    print("Using CPU only")

dataset_path = "/home/ssadmin/GOLD_XYZ_OSC.0001_1024.hdf5"
if not os.path.isfile(dataset_path):
    print ('dataset not found')
    exit(0)

dataset = radioml_18_dataset(dataset_path)

# Setting seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)


from sklearn.metrics import accuracy_score

def train(model, train_loader, optimizer, criterion):
    losses = []
    model.train()    
    for (inputs, target, snr) in tqdm(train_loader, desc="Batches", leave=False):   
        if gpu is not None:
            inputs = inputs.cuda()
            target = target.cuda()
                
        # forward pass
        output = model(inputs)
        loss = criterion(output, target)
        
        # backward pass + run optimizer to update weights
        for param in model.parameters():
            param.grad = None

        loss.backward()
        optimizer.step()
        
        # keep track of loss value
        losses.append(loss.cpu().detach().numpy())
           
    return losses

def test(model, test_loader):    
    # ensure model is in eval mode
    model.eval() 
    y_true = []
    y_pred = []
   
    with torch.no_grad():
        for (inputs, target, snr) in test_loader:
            if gpu is not None:
                inputs = inputs.cuda()
                target = target.cuda()
            output = model(inputs)
            pred = output.argmax(dim=1, keepdim=True)
            y_true.extend(target.tolist()) 
            y_pred.extend(pred.reshape(-1).tolist())
        
    return accuracy_score(y_true, y_pred)

def display_loss_plot(losses, title="Training loss", xlabel="Iterations", ylabel="Loss"):
    plt.figure()
    x_axis = [i for i in range(len(losses))]
    plt.plot(x_axis,losses)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

batch_size = 1024
num_epochs = 20

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

data_loader_train = DataLoader(dataset, batch_size=batch_size, sampler=dataset.train_sampler, num_workers=8, pin_memory=True)
data_loader_test = DataLoader(dataset, batch_size=batch_size, sampler=dataset.test_sampler)

if gpu is not None:
    print ('port model to cuda')
    model = model.cuda()

# loss criterion and optimizer
criterion = nn.CrossEntropyLoss()
if gpu is not None:
    print ('port criterion to cuda')
    criterion = criterion.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)
lr_scheduler =torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 17], gamma=0.1)

running_loss = []
running_test_acc = []

for epoch in tqdm(range(num_epochs), desc="Epochs"):
        loss_epoch = train(model, data_loader_train, optimizer, criterion)
        test_acc = test(model, data_loader_test)
        print("Epoch %d: Training loss = %f, test accuracy = %f" % (epoch, np.mean(loss_epoch), test_acc))
        running_loss.append(loss_epoch)
        running_test_acc.append(test_acc)
        lr_scheduler.step()


# Plot training loss over epochs
loss_per_epoch = [np.mean(loss_per_epoch) for loss_per_epoch in running_loss]
display_loss_plot(loss_per_epoch)


# Plot test accuracy over epochs
acc_per_epoch = [np.mean(acc_per_epoch) for acc_per_epoch in running_test_acc]
display_loss_plot(acc_per_epoch, title="Test accuracy", ylabel="Accuracy [%]")


# Save the trained parameters to disk
torch.save(model.state_dict(), "new_model_trained.pth")

t_elapsed = time.time() - start_time
print("--- %s seconds ---" % str(t_elapsed))
print("--- %s minutes ---" % str(t_elapsed / 60.0))
print("--- %s hours   ---" % str(t_elapsed / 3600.0))

plt.show()