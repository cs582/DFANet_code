from  imagenet  import ILSVRC
import argparse
from torch.utils.data import  DataLoader
from pathlib import Path
import yaml
from trainer import Trainer
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.optim import lr_scheduler
#from model.dfanet import XceptionA
from model.backbone import XceptionA
from config import Config
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import torch

if __name__=='__main__':

    cfg=Config()
    #create dataset
    torch.cuda.empty_cache()

    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        normalize,
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        normalize,
    ])

    train_dataset = ILSVRC(ilsvrc_data_path='datasets/imagenet',meta_path='datasets/imagenet/meta.mat',
                            transform = transform_train)
    val_dataset =  ILSVRC(ilsvrc_data_path='datasets/imagenet',meta_path='datasets/imagenet/meta.mat',
                            transform = transform_val, val=True)

    train_loader = DataLoader(dataset=train_dataset,
                                           batch_size=64, shuffle=True,
                                           num_workers=4)
    val_loader = DataLoader(dataset=val_dataset,
                                         batch_size=64, shuffle=False,
                                         num_workers=4)

    #net = XceptionA(cfg.CH_CFG[0],num_classes=1000)
    net = XceptionA()
    #load loss
    criterion = CrossEntropyLoss()
    optimizer = optim.SGD(
    net.parameters(), lr=0.3, momentum=0.9,weight_decay=4e-5, nesterov=True)  #select the optimizer

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer,30,gamma=0.1)
    trainer = Trainer('training', optimizer,exp_lr_scheduler, net, cfg, './log')
    trainer.load_weights(trainer.find_last())
    trainer.train(train_loader, val_loader, criterion, 60)
    #trainer.evaluate(valid_loader)
    print('Finished Training')
