'''
Train a new network on a data set with train.py

Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg13"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu
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

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model(neural network) on a dataset of images and saves it to a checkpoint for future predicitons")
    parser.add_argument('--data_root', default='flowers', type=str, help='set the data dir')
    parser.add_argument('--model', default='vgg11', type=str, help='choose the model architecture')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--hidden_units', default=1000, nargs='+', type=int, help='list of integers, the sizes of the hidden layers')
    parser.add_argument('--epochs', default=3, type=int, help='num of training epochs')
    parser.add_argument('--gpu_mode', default=True, type=bool, help='set the gpu mode')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    data_dir = args.data_root
    gpu_mode = args.gpu_mode
    model_name = args.model
    lr = args.lr
    hidden_units = args.hidden_units
    epochs = args.epochs

    print('='*10+'Params'+'='*10)
    print('Data dir:      {}'.format(data_dir))
    print('Model:         {}'.format(model_name))
    print('Hidden units: {}'.format(hidden_units))
    print('Learning rate: {}'.format(lr))
    print('Epochs:        {}'.format(epochs))

    # load the datase
    # Load the pretrained model from pytorch
    if model_name == 'vgg11':
        model = models.vgg11_bn(pretrained=True)
        # model.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
        # print(model.classifier[6].out_features) # 1000

        # Freeze training for all layers,# Newly created modules have require_grad=True by default
        for param in model.features.parameters():
            param.require_grad = False
        classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout1', nn.Dropout(0.5)),
                              ('output', nn.Linear(hidden_units, 102))
                              ]))

        model.classifier = classifier
        criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # load data
    image_datasets, dataset_sizes, dataloaders = load_data(data_dir)
    #train model
    model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler,
                       epochs, dataset_sizes, dataloaders)
    # test model: Do validation on the test set
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():

        for images, labels in dataloaders['test']:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    #print('Accuracy of the network on the 819 test images: %d %%' % (100 * correct / total))

    print("Accuracy of the network on the {} test images: {}%".format(dataset_sizes['test'], round(100 * correct / total, 1)))

    # Save the checkpoint
    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {'number_of_epochs': epochs,
                  'mapping_classes_to_indices':image_datasets['train'].class_to_idx,
                  'model_classifier': model.classifier,
                  'model_state_dict': model.state_dict(),
                  'model_class_to_inx': model.class_to_idx,
                  'optimizer_state_dict':optimizer.state_dict(),
                  }

    torch.save(checkpoint, 'checkpoint.pth')


def load_data(data_dir):
    data_dir = data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'

    data_transforms = {
    TRAIN: transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),
    VALID: transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])]),
    TEST: transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x),
            transform=data_transforms[x]
        )
        for x in [TRAIN, VALID, TEST]
    }


    #  Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=24,
            shuffle=True
        )
        for x in [TRAIN, VALID, TEST]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VALID, TEST]}

    for x in [TRAIN, VALID, TEST]:
        print("Loaded {} images under {}".format(dataset_sizes[x], x))

    print("Classes: ")
    class_names = image_datasets[TRAIN].classes
    print(image_datasets[TRAIN].classes)
    print('number of classes:', len(image_datasets[TRAIN].classes))

    return image_datasets, dataset_sizes, dataloaders


def train_model(model, criterion, optimizer, scheduler, num_epochs, dataset_sizes, dataloaders):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    model.to('cuda')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



if __name__ == '__main__':
    main()
