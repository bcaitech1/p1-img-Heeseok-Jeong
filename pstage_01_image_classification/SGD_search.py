import argparse
import glob
import json
import os
import random
import re
from importlib import import_module
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset, MaskSplitByProfileDataset
from loss import create_criterion

from efficientnet_pytorch import EfficientNet

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def load_model(saved_model, model_name, name, num_classes, device):
    print(f"model_name : {model_name}")
    model_cls = getattr(import_module("model"), model_name)
    model = model_cls(
        num_classes=num_classes
    )

    model_path = os.path.join(saved_model, name)
    model_path = os.path.join(model_path, 'best.pth')
    print(f"Model loading success from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

seed = 42
seed_everything(seed)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")    

# dataset
dataset_name = 'MaskBaseDataset'
dataset_module = getattr(import_module("dataset"), dataset_name)  # default: BaseAugmentation
data_dir = '/opt/ml/input/data/train/images'
dataset = dataset_module(
    data_dir=data_dir,
)

augmentation_name = 'BaseAugmentation'
transform_module = getattr(import_module("dataset"), augmentation_name)  # default: BaseAugmentation
transform = transform_module(
    resize=[244, 244],
    mean=dataset.mean,
    std=dataset.std,
)
dataset.set_transform(transform)

valid = 0
train_set, _ = dataset.split_dataset(valid)
batch_size = 28
log_interval = 20
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    num_workers=8,
    shuffle=True,
    pin_memory=use_cuda,
    drop_last=True,
)

# model
model_dir = './model'
model_name = 'EfficientNet_B4'
# model_name = 'ResNet50'
name = 'Other_EfficientNet_B4_3'
# name = 'Other_ResNet50_3'
num_classes = dataset.num_classes  # 18
output_dir = 'SGD_EfficientNet_B4_4'

save_dir = os.path.join(model_dir, output_dir)
os.makedirs(save_dir, exist_ok=True)

model = load_model(model_dir, model_name, name, num_classes, device).to(device)
model.train()

criteron_name = 'cross_entropy'
criterion = create_criterion(criteron_name)

lr = 1e-6
optimizer = SGD(model.parameters(), lr=lr, weight_decay=0.01)
lr_decay_step = 10
scheduler = StepLR(optimizer, lr_decay_step, gamma=0.5)

# SGD train
epochs = 4

for epoch in range(epochs):
    # train loop
    
    loss_value = 0
    matches = 0
    for idx, train_batch in enumerate(train_loader):
        inputs, labels = train_batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outs = model(inputs)
        preds = torch.argmax(outs, dim=-1)
        loss = criterion(outs, labels)

        loss.backward()
        optimizer.step()

        loss_value += loss.item()
        matches += (preds == labels).sum().item()
        if (idx + 1) % log_interval == 0:
            train_loss = loss_value / log_interval
            train_acc = matches / batch_size / log_interval
            current_lr = get_lr(optimizer)
            print(
                f"Epoch[{epoch+1}/{epochs}]({idx + 1}/{len(train_loader)}) || "
                f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
            )

            loss_value = 0
            matches = 0

    torch.save(model.state_dict(), f"{save_dir}/best.pth")
    
print("SGD Done.")
