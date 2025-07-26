#Done: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
# import torch.nn.functional as F
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


import argparse

def test(model, test_loader,criterion):
    '''
    Done: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )
    

def train(model, train_loader,val_loader,criterion, optimizer,args):
    '''
    Done: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    for epoch in range(int(args.epochs)):
        model.train()
        for batch_idx, (inputs, target) in enumerate(train_loader, 1):
        # for inputs, labels in train_loader:
            optimizer.zero_grad()
            # output = model(input)
            # loss = F.nll_loss(output, target)
            # loss.backward()
            # optimizer.step()
            logps = model(inputs)
            loss = criterion(logps, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(inputs),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        #test(model, test_loader)
        # ----- Validation after each epoch -----
        # if hook:
        #     hook.set_mode(modes.EVAL)
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                logps = model(inputs)
                loss = criterion(logps, targets)
                val_loss += loss.item()
                _, preds = torch.max(logps, 1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        val_loss /= len(val_loader)
        val_accuracy = 100. * correct / total
        logger.info(
            f"Validation Epoch: {epoch}\tLoss: {val_loss:.4f}\tAccuracy: {val_accuracy:.2f}%"
        )
    # save_model(model, args.model_dir)
    return model

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)
    
def net(num_classes):
    '''
    Done: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)
    
    # Freeze feature extractor layers (optional)
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer to match the number of classes
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, num_classes)
    )
    
    return model
    

def create_data_loaders(data_dir, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    image_size = 224
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    valid_dataset = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=eval_transforms)
    test_dataset  = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=eval_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    class_names = train_dataset.classes

    return train_loader, valid_loader, test_loader, class_names


def main(args):
    print(vars(args))
    data_dir = args.data_dir
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir  = os.path.join(data_dir, 'test')

    train_loader, valid_loader, test_loader, class_names = create_data_loaders(
    data_dir=data_dir,
    batch_size=64
    )
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net(133)
    
    '''
    TODO: Create your loss and optimizer
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, train_loader,valid_loader, criterion, optimizer,args)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, criterion)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
    "--dropout", type=float, default=0.3, help="Dropout probability (default: 0.3)"
    )
    default=os.environ.get('SM_CHANNEL_TRAINING', './data')
    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS", '["localhost"]')))
    parser.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST", "localhost"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"))
    parser.add_argument("--data-dir", type=str, default=os.environ.get('SM_CHANNEL_TRAINING', './dogImages'))
    parser.add_argument("--num-gpus", type=int, default=os.environ.get("SM_NUM_GPUS"))
    
    args=parser.parse_args()
    print(vars(args))
    main(args)
