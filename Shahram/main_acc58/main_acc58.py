#!/usr/bin/env python3


# Import some general modules
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import h5py
import os.path


from torch import nn
import brevitas.nn as qnn
from brevitas.quant import IntBias
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.defaults import Int8ActPerTensorFloatMinMaxInit
from brevitas.export.onnx.generic.manager import BrevitasONNXManager
import os
# import netron
from IPython.display import IFrame


# Select which GPU to use (if available)
gpu = 0
if torch.cuda.is_available():
    torch.cuda.device(gpu)
    print("Using GPU %d" % gpu)
else:
    gpu = None
    print("Using CPU only")




# Check if dataset is present

dataset_path = "/home/ssadmin/GOLD_XYZ_OSC.0001_1024.hdf5"
os.path.isfile(dataset_path)


# Prepare data loader


class radioml_18_dataset(Dataset):
    def __init__(self, dataset_path):
        super(radioml_18_dataset, self).__init__()
        h5_file = h5py.File(dataset_path,'r')
        self.data = h5_file['X']
        self.mod = np.argmax(h5_file['Y'], axis=1) # comes in one-hot encoding
        self.snr = h5_file['Z'][:,0]
        self.len = self.data.shape[0]

        self.mod_classes = ['OOK','4ASK','8ASK','BPSK','QPSK','8PSK','16PSK','32PSK',
        '16APSK','32APSK','64APSK','128APSK','16QAM','32QAM','64QAM','128QAM','256QAM',
        'AM-SSB-WC','AM-SSB-SC','AM-DSB-WC','AM-DSB-SC','FM','GMSK','OQPSK']
        self.snr_classes = np.arange(-20., 32., 2) # -20dB to 30dB

        # do not touch this seed to ensure the prescribed train/test split!
        np.random.seed(2018)
        train_indices = []
        test_indices = []
        for mod in range(0, 24): # all modulations (0 to 23)
            for snr_idx in range(0, 26): # all SNRs (0 to 25 = -20dB to +30dB)
                # 'X' holds frames strictly ordered by modulation and SNR
                start_idx = 26*4096*mod + 4096*snr_idx
                indices_subclass = list(range(start_idx, start_idx+4096))
                
                # 90%/10% training/test split, applied evenly for each mod-SNR pair
                split = int(np.ceil(0.1 * 4096)) 
                np.random.shuffle(indices_subclass)
                train_indices_subclass = indices_subclass[split:]
                test_indices_subclass = indices_subclass[:split]
                
                # you could train on a subset of the data, e.g. based on the SNR
                # here we use all available training samples
                if snr_idx >= 4:
                    train_indices.extend(train_indices_subclass)
                test_indices.extend(test_indices_subclass)
                
        self.train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        self.test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    def __getitem__(self, idx):
        # transpose frame into Pytorch channels-first format (NCL = -1,2,1024)
        return self.data[idx].transpose(), self.mod[idx], self.snr[idx]

    def __len__(self):
        return self.len

dataset = radioml_18_dataset(dataset_path)




# Adjustable hyperparameters
input_bits = 8
a_bits = 8
w_bits = 8
filters_conv = 32
filters_dense = 32

# Setting seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

class InputQuantizer(Int8ActPerTensorFloatMinMaxInit):
    bit_width = input_bits
    min_val = -2.0
    max_val = 2.0
    scaling_impl_type = ScalingImplType.CONST # Fix the quantization range to [min_val, max_val]

model = nn.Sequential(
    # Input quantization layer
    qnn.QuantHardTanh(act_quant=InputQuantizer),
    
    qnn.QuantConv1d(2, filters_conv, 3, padding=1, weight_bit_width=w_bits, bias=False),
    nn.BatchNorm1d(filters_conv),
    qnn.QuantReLU(bit_width=a_bits),
    nn.MaxPool1d(2),

    qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w_bits, bias=False),
    nn.BatchNorm1d(filters_conv),
    qnn.QuantReLU(bit_width=a_bits),
    nn.MaxPool1d(2),

    qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w_bits, bias=False),
    nn.BatchNorm1d(filters_conv),
    qnn.QuantReLU(bit_width=a_bits),
    nn.MaxPool1d(2),

    qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w_bits,bias=False),
    nn.BatchNorm1d(filters_conv),
    qnn.QuantReLU(bit_width=a_bits),
    nn.MaxPool1d(2),

    qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w_bits, bias=False),
    nn.BatchNorm1d(filters_conv),
    qnn.QuantReLU(bit_width=a_bits),
    nn.MaxPool1d(2),

    qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w_bits, bias=False),
    nn.BatchNorm1d(filters_conv),
    qnn.QuantReLU(bit_width=a_bits),
    nn.MaxPool1d(2),

    
    qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w_bits, bias=False),
    nn.BatchNorm1d(filters_conv),
    qnn.QuantReLU(bit_width=a_bits),
    nn.MaxPool1d(2),
    
    nn.Flatten(),


    qnn.QuantLinear(filters_conv*8, filters_dense, weight_bit_width=w_bits, bias=False),
    nn.BatchNorm1d(filters_dense),
    qnn.QuantReLU(bit_width=a_bits),

    # nn.Dropout(0.2),
    qnn.QuantLinear(filters_dense, filters_dense, weight_bit_width=w_bits, bias=False),
    nn.BatchNorm1d(filters_dense),
    qnn.QuantReLU(bit_width=a_bits, return_quant_tensor=True),
    
    qnn.QuantLinear(filters_dense, 24, weight_bit_width=w_bits, bias=True, bias_quant=IntBias),
)




from sklearn.metrics import accuracy_score

def train(model, train_loader, optimizer, criterion):
    losses = []
    # ensure model is in training mode
    model.train()    
#     scaler = GradScaler()
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
#         optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        
        # keep track of loss value
        losses.append(loss.cpu().detach().numpy())
        
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
           
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
    x_axis = [i for i in range(len(losses))]
    plt.plot(x_axis,losses)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()




batch_size = 2048
num_epochs = 100

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

data_loader_train = DataLoader(dataset, batch_size=batch_size, sampler=dataset.train_sampler,num_workers=4,pin_memory=True)
data_loader_test = DataLoader(dataset, batch_size=batch_size, sampler=dataset.test_sampler)

if gpu is not None:
    model = model.cuda()

# loss criterion and optimizer
criterion = nn.CrossEntropyLoss()
if gpu is not None:
    criterion = criterion.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)
# lr_scheduler =torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 17], gamma=0.1)

running_loss = []
running_test_acc = []

for epoch in tqdm(range(num_epochs), desc="Epochs"):
        loss_epoch = train(model, data_loader_train, optimizer, criterion)
        test_acc = test(model, data_loader_test)
        print("Epoch %d: Training loss = %f, test accuracy = %f" % (epoch, np.mean(loss_epoch), test_acc))
        # print("Epoch %d: Training loss = %f" %(epoch, np.mean(loss_epoch)))
        running_loss.append(loss_epoch)
        running_test_acc.append(test_acc)
        lr_scheduler.step()

# test_acc = test(model, data_loader_test)
# print("Epoch %d: Training loss = %f, test accuracy = %f" % (epoch, np.mean(loss_epoch), test_acc))
# Plot training loss over epochs
loss_per_epoch = [np.mean(loss_per_epoch) for loss_per_epoch in running_loss]
display_loss_plot(loss_per_epoch)


# Plot test accuracy over epochs
acc_per_epoch = [np.mean(acc_per_epoch) for acc_per_epoch in running_test_acc]
display_loss_plot(acc_per_epoch, title="Test accuracy", ylabel="Accuracy [%]")


# Save the trained parameters to disk
torch.save(model.state_dict(), "model_trained.pth")


# # Load a Trained Model <a id='load_trained_model'></a>
# Alternatively, you can load the provided pre-trained model.
# It was trained for 20 epochs and reaches an overall accuracy of 59.5%.

# Load trained parameters
# savefile = "models/pretrained_VGG10_w8a8_20.pth"
# saved_state = torch.load(savefile, map_location=torch.device("cpu"))
# model.load_state_dict(saved_state)
# if gpu is not None:
#     model = model.cuda()


# # Evaluate the Accuracy <a id='evaluate_accuracy'></a>
# The following cells visualize the test accuracy across different modulations and signal-to-noise ratios. Submissions for this problem statement must reach an overall accuracy of at least **56.0%**, so this should give you an idea what makes up this figure.
# Set up a fresh test data loader
# batch_size = 1024
dataset = radioml_18_dataset(dataset_path)
data_loader_test = DataLoader(dataset, batch_size=batch_size, sampler=dataset.test_sampler)



# Run inference on validation data
y_exp = np.empty((0))
y_snr = np.empty((0))
y_pred = np.empty((0,len(dataset.mod_classes)))
model.eval()
with torch.no_grad():
    for data in tqdm(data_loader_test, desc="Batches"):
        inputs, target, snr = data
        if gpu is not None:
            inputs = inputs.cuda()
        output = model(inputs)
        y_pred = np.concatenate((y_pred,output.cpu()))
        y_exp = np.concatenate((y_exp,target))
        y_snr = np.concatenate((y_snr,snr))


# # Plot overall confusion matrix
# def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(labels))
#     plt.xticks(tick_marks, labels, rotation=90)
#     plt.yticks(tick_marks, labels)
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')

# conf = np.zeros([len(dataset.mod_classes),len(dataset.mod_classes)])
# confnorm = np.zeros([len(dataset.mod_classes),len(dataset.mod_classes)])
# for i in range(len(y_exp)):
#     j = int(y_exp[i])
#     k = int(np.argmax(y_pred[i,:]))
#     conf[j,k] = conf[j,k] + 1
# for i in range(0,len(dataset.mod_classes)):
#     confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])

# plt.figure(figsize=(12,8))
# plot_confusion_matrix(confnorm, labels=dataset.mod_classes)

# cor = np.sum(np.diag(conf))
# ncor = np.sum(conf) - cor
# print("Overall Accuracy across all SNRs: %f"%(cor / (cor+ncor)))


# # Plot confusion matrices at 4 different SNRs
# snr_to_plot = [-20,-4,+4,+30]
# plt.figure(figsize=(16,10))
# acc = []
# for snr in dataset.snr_classes:
#     # extract classes @ SNR
#     indices_snr = (y_snr == snr).nonzero()
#     y_exp_i = y_exp[indices_snr]
#     y_pred_i = y_pred[indices_snr]
 
#     conf = np.zeros([len(dataset.mod_classes),len(dataset.mod_classes)])
#     confnorm = np.zeros([len(dataset.mod_classes),len(dataset.mod_classes)])
#     for i in range(len(y_exp_i)):
#         j = int(y_exp_i[i])
#         k = int(np.argmax(y_pred_i[i,:]))
#         conf[j,k] = conf[j,k] + 1
#     for i in range(0,len(dataset.mod_classes)):
#         confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
 
#     if snr in snr_to_plot:
#         plot, = np.where(snr_to_plot == snr)[0]
#         plt.subplot(221+plot)
#         plot_confusion_matrix(confnorm, labels=dataset.mod_classes, title="Confusion Matrix @ %d dB"%(snr))
 
#     cor = np.sum(np.diag(conf))
#     ncor = np.sum(conf) - cor
#     acc.append(cor/(cor+ncor))


# # Plot accuracy over SNR
# plt.figure(figsize=(10,6))
# plt.plot(dataset.snr_classes, acc, marker='o')
# plt.xlabel("SNR [dB]")
# plt.xlim([-20, 30])
# plt.ylabel("Classification Accuracy")
# plt.yticks(np.arange(0, 1.1, 0.1))
# plt.title("Classification Accuracy over SNR")
# plt.grid()
# plt.title("Classification Accuracy over SNR");

# print("Accuracy @ highest SNR (+30 dB): %f"%(acc[-1]))
# print("Accuracy overall: %f"%(np.mean(acc)))


# # Plot accuracy per modulation
# accs = []
# for mod in range(24):
#     accs.append([])
#     for snr in dataset.snr_classes:
#         indices = ((y_exp == mod) & (y_snr == snr)).nonzero()
#         y_exp_i = y_exp[indices]
#         y_pred_i = y_pred[indices]
#         cor = np.count_nonzero(y_exp_i == np.argmax(y_pred_i, axis=1))
#         accs[mod].append(cor/len(y_exp_i))
        
# # Plot accuracy-over-SNR curve
# plt.figure(figsize=(12,8))
# for mod in range(24):
#     if accs[mod][25] < 0.95 or accs[mod][0] > 0.1:
#         color = None
#     else:
#         color = "black"
#     plt.plot(dataset.snr_classes, accs[mod], label=str(mod) + ": " + dataset.mod_classes[mod], color=color)
# plt.xlabel("SNR [dB]")
# plt.xlim([-20, 30])
# plt.ylabel("Classification Accuracy")
# plt.yticks(np.arange(0, 1.1, 0.1))
# plt.title("Accuracy breakdown")
# plt.grid()
# plt.legend();


# # # Evaluate the Inference Cost <a id='evaluate_inference_cost'></a>
# # 
# # First, we have to export the model to Brevita's quantized variant of the ONNX interchange format. **All submissions must correctly pass through this export flow and provide the resulting .onnx file**. Any `TracerWarning` can be safely ignored.


# export_onnx_path = "models/model_export.onnx"
# final_onnx_path = "models/model_final.onnx"
# cost_dict_path = "models/model_cost.json"

# BrevitasONNXManager.export(model.cpu(), input_t=torch.randn(1, 2, 1024), export_path=export_onnx_path);


# # Now we use our analysis tool, which is part of [finn-base](https://github.com/Xilinx/finn-base), to determine the inference cost. It reports the number of output activation variables (`mem_o`), weight parameters (`mem_w`), and multiply-accumulate operations (`op_mac`) for each data type. These are used to calculate the total number of activation bits, weight bits, and bit-operations (BOPS).
# # 
# # If the report shows any unsupported operations, for instance because you implemented custom layers, you should check with the rules on the problem statement [website](http://bit.ly/brevitas-radioml-challenge-21) and consider to contact the organizers.

# # In[33]:


# from finn.util.inference_cost import inference_cost
# import json

# inference_cost(export_onnx_path, output_json=cost_dict_path, output_onnx=final_onnx_path,
#                preprocess=True, discount_sparsity=True)


# # The call to `??nference_cost()` cleans up the model by inferring shapes and datatypes, folding constants, etc. We visualize the pre-processed ONNX model using [Netron](https://netron.app/).

# # In[12]:




# def showInNetron(model_filename):
#     localhost_url = os.getenv("LOCALHOST_URL")
#     netron_port = os.getenv("NETRON_PORT")
#     netron.start(model_filename, address=("0.0.0.0", int(netron_port)))
#     return IFrame(src="http://%s:%s/" % (localhost_url, netron_port), width="100%", height=400)

# showInNetron(final_onnx_path)


# # Finally, we compute the inference cost score, normalized to the baseline 8-bit VGG10 defined in this notebook. **Submissions will be judged based on this score.**


# with open(cost_dict_path, 'r') as f:
#     inference_cost_dict = json.load(f)

# bops = int(inference_cost_dict["total_bops"])
# w_bits = int(inference_cost_dict["total_mem_w_bits"])

# bops_baseline = 807699904
# w_bits_baseline = 1244936

# score = 0.5*(bops/bops_baseline) + 0.5*(w_bits/w_bits_baseline)
# print("Normalized inference cost score: %f" % score)


(self, iÑÆ
