#!/usr/bin/env python


import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from PIL import Image
from stanford_online_products import StanfordOnlineProducts
from torch.utils.data import DataLoader
from torchvision.transforms import Resize,RandomResizedCrop,RandomHorizontalFlip,ToTensor,Normalize, CenterCrop
from pytorch_metric_learning import samplers
from transforms import ConvertToBGR, Multiplier
from torchvision.transforms import InterpolationMode
import pretrainedmodels
from pytorch_metric_learning.utils import common_functions as c_f


if not torch.cuda.is_available():
    print("Torch cuda acceleration is NOT available")
    device = torch.device('cpu')
for i in range(torch.cuda.device_count()):
    name=torch.cuda.get_device_name(i)
    print("I am GPU number {i}, but you can call me {name}")
    

path_dataset='/home/andrea/Scrivania/thesis/datasets'

transf_train = transforms.Compose([
                                    ConvertToBGR(),
                                    Resize(size=256, interpolation=InterpolationMode.BILINEAR),
                                    RandomResizedCrop(size=(227, 227), scale=(0.16, 1), ratio=(0.75, 1.33), interpolation=InterpolationMode.BILINEAR),
                                    RandomHorizontalFlip(p=0.5),
                                    ToTensor(),
                                    Multiplier(multiple=255),
                                    Normalize(mean=[104, 117, 128], std=[1, 1, 1])
                                    ])


transf_eval = transforms.Compose([
                                    ConvertToBGR(),
                                    Resize(size=256, interpolation=InterpolationMode.BILINEAR),
                                    CenterCrop(size=(227, 227)),
                                    ToTensor(),
                                    Multiplier(multiple=255),
                                    Normalize(mean=[104, 117, 128], std=[1, 1, 1])
                                    ])


train_set = StanfordOnlineProducts(root=path_dataset,
                                   train=True,
                                   transform=transf_train)
                                   
valid_set = StanfordOnlineProducts(root=path_dataset,
			           train=True,
			           transform=transf_eval)

test_set  = StanfordOnlineProducts(root=path_dataset,
                                   train=False,
                                   transform=transf_eval)
                                   
                                   
valid_labels= np.loadtxt('/home/andrea/Scrivania/thesis/src/valid_labels.txt', dtype=int)
valid_set.labels= valid_labels
train_set.labels = [i for i in train_set.labels if i not in valid_labels]

print("---DATASET INFO---")
print("len train_set:",len(train_set))
print("labels train_set:",len(train_set.labels)) 
print("") 
print("len valid_set:",len(valid_set))
print("labels valid_set:",len(valid_set.labels)) 
print("") 
print("len test_set:",len(test_set))
print("labels test_set:",test_set.labels.shape)
print("")


#batch_size=32
n_classes = 8
n_samples = 4
print("dim batch:", n_classes*n_samples)
print("-num classes:", n_classes)
print("-num sample per class:", n_samples)
print("")


train_sampler = samplers.MPerClassSampler(train_set.labels, m=4, batch_size=32, length_before_new_iter=len(train_set))
test_sampler = samplers.MPerClassSampler(test_set.labels, m=4, batch_size=32, length_before_new_iter=len(test_set))
valid_sampler = samplers.MPerClassSampler(valid_set.labels, m=4, batch_size=32, length_before_new_iter=len(valid_set))

train_dataloader = DataLoader(train_set, batch_size=32,sampler=train_sampler, num_workers=4)
test_dataloader= DataLoader(test_set, batch_size=32,sampler=test_sampler, num_workers=4)
valid_dataloader = DataLoader(valid_set, batch_size=32,sampler=valid_sampler, num_workers=4)

print("num batches train:",len(train_dataloader))
print("num batches valid:",len(valid_dataloader))
print("num batches test:",len(test_dataloader))
print("")


print("----EMBED MODEL------")
model_name = 'bninception' # could be fbresnet152 or inceptionresnetv2
model = pretrainedmodels.__dict__[model_name](pretrained='imagenet')
model.last_linear = nn.Linear(1024, 128)

optimizer = optim.RMSprop(model.parameters(), lr=1e-6, weight_decay=0.0001, momentum=0.9)

checkpoint = torch.load("/home/andrea/Scrivania/thesis/models/checkpoint_10epoch+valid.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']


print("Restore state: epoch={}, loss={}".format(epoch,loss))
print("")



print("----DISTANCE MODULE TRAIN------")
from net import Net_residuals, Net_fc
from loss import DistanceLoss
from train import trainDistanceModule
from mining_pairs import get_pairs

distNet = Net_fc()
distNet.to(device)
loss_func=DistanceLoss(margin=0.0, distance=2)
#optimizer = optim.RMSprop(distNet.parameters(), lr=1e-5, momentum=0.9)
#optimizer = optim.Adam(distNet.parameters(), lr=5e-3, weight_decay=0.0001)
optimizer = optim.Adam(distNet.parameters(), lr=1e-4, weight_decay=0.0001)
#optimizer = optim.SGD(distNet.parameters(), lr=1e-4, momentum=0.9)

scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

loss_history_train=[]
loss_history_valid=[]

num_epochs = 20
for epoch in range(1, num_epochs+1):
    trainDistanceModule(model, distNet, loss_func, device, train_dataloader,valid_dataloader, optimizer, epoch, loss_history_train,loss_history_valid, scheduler)


print("----------------------------")

