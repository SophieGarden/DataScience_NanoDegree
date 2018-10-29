'''
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint

Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu

e.g.: python predict.py 'flowers/test/1/image_06760.jpg' 'checkpoint.pth' 5 'gpu'
'''

import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim, topk
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models, utils
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.optim import lr_scheduler
from collections import OrderedDict
from PIL import Image
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Predict flower name from an image with a pre-trained model: pass a single image, return the flower name and class probability")
    parser.add_argument('--image_path', default='flowers/test/1/image_06760.jpg', type=str, help='choose the image path')
    parser.add_argument('--checkpoint', default='checkpoint.pth', type=str, help='give the checkpoint')
    parser.add_argument('--top_k', default=5, type=int, help='num of top predicted classes')
    parser.add_argument('--gpu_mode', default=True, type=bool, help='set the gpu mode')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    image_path = args.image_path
    gpu_mode = args.gpu_mode
    checkpoint = args.checkpoint
    top_k = args.top_k
    print('='*10+'Params'+'='*10)
    print('Image path:      {}'.format(image_path))


    model_l = load_checkpoint('checkpoint.pth')

    #Sanity checkpoint# Display an image along with the top 5 classes
    im_path = image_path
    im = Image.open(im_path)
    pi = process_image(im)



    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # Set the device
    device = torch.device("cuda:0" if gpu_mode==True else "cpu")

    probs, classes = predict(im_path, model_l,top_k, device)
    print(probs, classes)
    names = [cat_to_name[c] for c in classes]
    print(names)



#Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']

    model_name = checkpoint['model_name']
    classifier = checkpoint['classifier']
    class_to_idx = checkpoint['class_to_idx']
    model_state_dict = checkpoint['state_dict']

    if model_name == 'resnet':
        print('resnet18')
        model = models.resnet18(pretrained=True)
        model.fc = classifier

    elif model_name == 'vgg':
        print('vgg11_bn')
        model = models.vgg11_bn(pretrained=True)
        model.classifier = classifier

    model.class_to_idx = class_to_idx
    model.load_state_dict(model_state_dict)
    # lr = checkpoint['learning_rate']
    # epochs = checkpoint['epochs']
    # optimizer = optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # exp_lr_scheduler.load_state_dict(checkpoint['exp_lr_scheduler'])

    return model




def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''


    # TODO: Process a PIL image for use in a PyTorch model
    #resize
    size = (256, 256)
    image = image.resize(size)

    #crop
    width, height = image.size   # Get dimensions
    new_width = new_height = 224

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    image = image.crop(box = (left, top, right, bottom))
#     image.crop(box = (16, 16, 240, 240))


    #normalize
    np_image = np.array(image)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    np_image = (np_image - mean) / std
    np_image = np_image.transpose(2,0,1)
    return np_image



def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    im = Image.open(image_path)
    pi_im = process_image(im)

    model.to(device)
    model.eval()
    with torch.no_grad():
        images = torch.from_numpy(pi_im).type(torch.FloatTensor)
        images = images.unsqueeze(0)
        images = images.to(device)

        outputs = model.forward(images)

        #
        outputs = torch.exp(outputs)
        probs, labels = outputs.topk(topk)

        probs = probs.cpu().numpy()[0]
        labels = labels.cpu().numpy()[0]


        inv_map = {v: k for k, v in model.class_to_idx.items()}
        classes = [inv_map[l] for l in labels]

    return probs, classes

if __name__ == '__main__':
    main()
