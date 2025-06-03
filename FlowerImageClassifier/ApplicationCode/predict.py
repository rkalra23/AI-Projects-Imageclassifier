#!/usr/bin/env python3
import argparse 
import time
import torch 
import numpy as np
import json
import sys
import os

from torch import nn, optim
from torchvision import datasets, models, transforms
from PIL import Image

def load_checkpoint(filepath):
    # Load checkpoint file from disk
    checkpoint = torch.load(filepath) 
    arch=checkpoint['arch']
    if arch == 'vgg':
        model = models.vgg13(pretrained=True)
    elif arch == 'densenet':
        model = models.densenet121(pretrained=True)
    else:
        print("there is an error in model architecture")
        exit
    # model = models.vgg13(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=checkpoint['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs = checkpoint['epochs']
    
    return model

def process_image(image):
    image = Image.open(image).convert("RGB")
    image_transform=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # Converts to float and scales [0,255] → [0,1]
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
    ])
    return image_transform(image)

def predict(image_path, model,topk=5, device='cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    model.eval()
    img = process_image(image_path)
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        ps = torch.exp(output)  
        top_p, top_class = ps.topk(topk, dim=1)
    return top_p.cpu().numpy().squeeze().tolist(),top_class.cpu().numpy().squeeze().tolist()


def read_categories():
    if (args.category_names is not None):
        cat_file = args.category_names 
        jfile = json.loads(open(cat_file).read())
        return jfile
    return None

    

def commandLine():
    global args
    parser = argparse.ArgumentParser(description='using a model created classify an image!')
    parser.add_argument('--image_input', help='image file to classifiy (required)')
    parser.add_argument('--check', help='model used for classification (required)')
    parser.add_argument('--top_k', default=5,help='how many prediction categories to show [default 5].')
    parser.add_argument('--category_names', help='file for category names')
    parser.add_argument('--gpu', action='store_true', help='gpu option')
    args = parser.parse_args()
    return args

def main():
    global args,device
    args = commandLine() 

    print(args)
    if args.gpu:
        if torch.cuda.is_available():
            device='cuda'
        elif torch.mps.is_available():
            device='mps'
        else:
            raise Exception("--gpu option enabled...but no GPU detected")
    top_k=int(args.top_k)
    model=load_checkpoint(args.check)
    print(model)
    prob,classes = predict(args.image_input,model,top_k,device)
   
    actual_class = os.path.basename(os.path.dirname(args.image_input))  # returns '3'
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    actual_label_name = cat_to_name[actual_class]
    print('###################')
    print("Actual class is : ",actual_label_name)
    print('###################')
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    labels = [cat_to_name[idx_to_class[c]] for c in classes]
    confidence = prob[0]
    print("class predicted by model: ",labels[0])
    print("confidence level: ",confidence)
    print('###################')
    print(f"Top {args.top_k} Predictions:")
    for i, (name, prob) in enumerate(zip(labels,prob), start=1):
        print(f"{i}. {name}: {prob:.2f}")

main()