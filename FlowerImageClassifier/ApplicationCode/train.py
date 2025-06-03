#!/usr/bin/env python3
import argparse
import os
import torch
import json
from torch import nn, optim
from torchvision import datasets, models, transforms
from collections import OrderedDict
#from PIL import Image


def get_data():
    print("retreiving data")
    train_dir = args.data_directory + '/train'
    test_dir = args.data_directory + '/test'
    valid_dir = args.data_directory + '/valid'
    print("processing data into iterators")
    

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)


    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    print(len(trainloader))
    print(len(validationloader))
    print(len(testloader))
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    loaders = {'train':trainloader,'valid':validationloader,'test':testloader,'labels':cat_to_name,'class_to_idx': train_data.class_to_idx}
    return loaders
    

def validationCLI():
    print("validating parameters")
    if (args.gpu and not (torch.cuda.is_available() or torch.mps.is_available())):
        raise Exception("--gpu option enabled...but no GPU detected")
    if(not os.path.isdir(args.data_directory)):
        raise Exception('directory does not exist!')
    data_dir = (os.listdir(args.data_directory))

    expected_subdirs = ['train', 'valid', 'test']
    
    for i in expected_subdirs:
        if i not in data_dir:
            raise Exception('Missing one or more required subdirectories: train, valid, test')       
    print("Validation successfully done")   



def build_model():
    print("building model object")
    arch_type=args.arch
    if (arch_type == 'vgg'):
        model = models.vgg13(pretrained=True)
        input_node=25088
    elif (arch_type == 'densenet'):
        model = models.densenet121(pretrained=True)
        input_node=1024
    if (args.hidden_units is None):
        hidden_units = 4096
    else:
        hidden_units = args.hidden_units
    for param in model.parameters():
        param.requires_grad = False
    hidden_units = int(hidden_units)
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_node, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout',nn.Dropout(p=0.5)),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    model.classifier = classifier
    return model,arch_type

def train(model,arch,data):
    print("training model")
    
    if (args.learning_rate is None):
        learn_rate = 0.001
    else:
        learn_rate = args.learning_rate

    if (args.gpu):
        device = 'cuda'
    else:
        device = 'mps'
    
    learn_rate = float(learn_rate)
    epochs = int(args.epochs)
    
    trainloader=data['train']
    validloader=data['valid']
    testloader=data['test']
    class_to_idx=data['class_to_idx']
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    steps = 0
    running_loss = 0
    print_every = 10

    model.to(device)
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        validation_loss += batch_loss.item()
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {validation_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    print("Model Test Accuracy")
    test_model(model, testloader, criterion,'mps')
    save_model(arch,epochs,learn_rate,model.state_dict(),optimizer.state_dict(),model.classifier,class_to_idx)

def save_model(arch,epochs,learnRate,modelState,optimizerState,modelClassifier,class_to_idx):
    print("saving model")
    if (args.save_dir is None):
        save_dir = 'check.pth'
    else:
        save_dir = args.save_dir
    
    checkpoint = {
        'arch':arch,
        'epochs': epochs,
        'learning_rate': learnRate,
        'model_state_dict': modelState,
        'optimizer_state_dict': optimizerState,
        'classifier': modelClassifier,
        'class_to_idx': class_to_idx  
        }
    torch.save(checkpoint, save_dir)

def test_model(model, testloader, criterion, device="mps"):
    model.to(device)
    model.eval()
    test_loss = 0
    accuracy = 0

    with torch.no_grad():  # No gradient calculation
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            log_ps = model(images)  # Get log-probabilities
            loss = criterion(log_ps, labels)
            test_loss += loss.item()

            ps = torch.exp(log_ps)  # Convert log-prob to probabilities
            top_p, top_class = ps.topk(1, dim=1)  # Get predicted class
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    avg_loss = test_loss / len(testloader)
    avg_accuracy = accuracy / len(testloader)

    print(f"Test Loss: {avg_loss:.3f}")
    print(f"Test Accuracy: {avg_accuracy*100:.2f}%")


def commandLine():
    global args
    parser = argparse.ArgumentParser(description='Train a pretrained neural network!')
    parser.add_argument('--data_directory',required=True,help='data directory to be used for images')
    parser.add_argument('--save_dir', help='directory to save a neural network.')
    parser.add_argument('--arch', default='vgg13',help='models to use like [vgg13,vgg19 etc], default=VGG')
    parser.add_argument('--learning_rate',default=0.001, help='learning rate')
    parser.add_argument('--hidden_units', help='number of hidden units')
    parser.add_argument('--epochs', default=3,help='no of epochs, default=3')
    parser.add_argument('--gpu',action='store_true', help='if you want to use gpu')
    args = parser.parse_args()
    return args

def main():
    global args
    print("Lets create a image classifier model")  
    args=commandLine()
    validationCLI()
    loaders=get_data()
    # print(loaders['class_to_idx'])
    model,arch=build_model()
    print(model)
    model = train(model,arch,loaders)
    print("model Training finished!")

main()