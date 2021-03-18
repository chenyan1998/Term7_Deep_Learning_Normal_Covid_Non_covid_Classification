#!/usr/bin/env python
# coding: utf-8

# Import library
#Numpy, Matplotlib,Pillow,Torch

# Here is the normal-infected dataset for 3(normal, infectedcovid, infectednoncovid)classes
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torchvision import transforms 

class Lung_Train_Dataset(Dataset):
    
    def __init__(self):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.
        """
        
        # All images are of size 150 x 150
        self.img_size = (150, 150)
        
        # Only two classes will be considered here (normal and infected)
        self.classes = {0: 'normal', 1: 'infectednon',2:'infectedcovid'}
        
        # The dataset consists only of training images
        self.groups = 'train'
        
        # Number of images in each part of the dataset
        self.dataset_numbers = {'train_normal' : 1341,                   'train_infectednon' : 2530,                   'train_infectedcovid' : 1345 }
        
        # Path to images for different parts of the dataset
        self.dataset_paths = {'train_normal': './dataset/train/normal/',                         'train_infectednon': './dataset/train/infected/non-covid/',                         'train_infectedcovid': './dataset/train/infected/covid'}
        
        
    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """
        
        # Generate description
        msg = "This is the training dataset of the Lung Dataset"
        msg += " used for the Small Project Demo in the 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
        msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        for key, val in self.dataset_paths.items():
            msg += " - {}, in folder {}: {} images.\n".format(key, val, self.dataset_numbers[key])
        print(msg)
        
    
    def open_img(self, group_val, class_val, index_val):
        """
        Opens image with specified parameters.
        
        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infectednon', 'infectedcovid'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        
        Returns loaded image as a normalized Numpy array.
        """
        
        # Asserts checking for consistency in passed parameters
        err_msg = "Error - group_val variable should be set to 'train', 'test' or 'val'."
        assert group_val in self.groups, err_msg
        
        err_msg = "Error - class_val variable should be set to 'normal' or 'infected''infectedcovid'."
        assert class_val in self.classes.values(), err_msg
        
        max_val = self.dataset_numbers['{}_{}'.format(group_val, class_val)]
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
        err_msg += "\n(In {}/{}, you have {} images.)".format(group_val, class_val, max_val)
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val <= max_val, err_msg
        
        # Open file as before
        path_to_file = '{}/{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)
        with open(path_to_file, 'rb') as f:
            im = np.asarray(Image.open(f))/255
        f.close()
        return im
    
    
    def show_img(self, group_val, class_val, index_val):
        """
        Opens, then displays image with specified parameters.
        
        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected','infectedcovid'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """
        
        # Open image
        im = self.open_img(group_val, class_val, index_val)
        
        # Display
        plt.imshow(im)
        
        
    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """
        
        # Length function
        return sum(self.dataset_numbers.values())
    
    
    def __getitem__(self, index):
            """
            Getitem special method.

            Expects an integer value index, between 0 and len(self) - 1.

            Returns the image and its label as a one hot vector, both
            in torch tensor format in dataset.
            """

            # Get item special method
            first_val = int(list(self.dataset_numbers.values())[0])
            second_val = int(list(self.dataset_numbers.values())[1])
            #print('first_val: ',first_val)
            #print('second_val: ',second_val)
            if index < first_val:
                class_val = 'normal'
                label = torch.Tensor([1, 0])
                #print('choice1', index)
            elif index> first_val  and index < first_val +second_val:
                #print('choice2', index)
                index = index- first_val
                class_val = 'infectednon'
                label = torch.Tensor([0, 1])

            elif index> second_val and index < 5216:
                #print('choice3', index)
                class_val = 'infectedcovid'
                index = index - (first_val + second_val)
                label = torch.Tensor([0, 1])
            im = self.open_img(self.groups, class_val, index)
            im = transforms.functional.to_tensor(np.array(im)).float()
            return im, label
        
# Test code 

ld_train = Lung_Train_Dataset()
ld_train.describe()
#print('len: ', len(ld_train))
#im, class_oh = ld_train[5011]
#print("im.shape: " ,im.shape)
#print('im: ',im)
#print('class_oh: ',class_oh)

class Lung_Test_Dataset(Dataset):
    
    def __init__(self):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.
        """
        
        # All images are of size 150 x 150
        self.img_size = (150, 150)
        
        # Only two classes will be considered here (normal and infected)
        self.classes = {0: 'normal', 1: 'infectednon', 2:'infectedcovid'}
        
        # The dataset consists only of training images
        self.groups = 'test'
        
        # Number of images in each part of the dataset
        self.dataset_numbers = {'test_normal' : 234,                   'test_infectednon' : 242,                   'test_infectedcovid' : 138 }
        
        # Path to images for different parts of the dataset
        self.dataset_paths = { 'test_normal': './dataset/test/normal/',                         'test_infectednon': './dataset/test/infected/non-covid',                         'test_infectedcovid': './dataset/test/infected/covid'}
        
        
    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """
        
        # Generate description
        msg = "This is the test dataset of the Lung Dataset"
        msg += " used for the Small Project Demo in the 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
        msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        for key, val in self.dataset_paths.items():
            msg += " - {}, in folder {}: {} images.\n".format(key, val, self.dataset_numbers[key])
        print(msg)
        
    
    def open_img(self, group_val, class_val, index_val):
        """
        Opens image with specified parameters.
        
        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        
        Returns loaded image as a normalized Numpy array.
        """
        
        # Asserts checking for consistency in passed parameters
        err_msg = "Error - group_val variable should be set to 'train', 'test' or 'val'."
        assert group_val in self.groups, err_msg
        
        err_msg = "Error - class_val variable should be set to 'normal' or 'infected'."
        assert class_val in self.classes.values(), err_msg
        
        max_val = self.dataset_numbers['{}_{}'.format(group_val, class_val)]
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
        err_msg += "\n(In {}/{}, you have {} images.)".format(group_val, class_val, max_val)
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val <= max_val, err_msg
        
        # Open file as before
        path_to_file = '{}/{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)
        with open(path_to_file, 'rb') as f:
            im = np.asarray(Image.open(f))/255
        f.close()
        return im
    
    
    def show_img(self, group_val, class_val, index_val):
        """
        Opens, then displays image with specified parameters.
        
        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """
        
        # Open image
        im = self.open_img(group_val, class_val, index_val)
        
        # Display
        plt.imshow(im)
        
        
    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """
        
        # Length function
        return sum(self.dataset_numbers.values())
    
    
    def __getitem__(self, index):
        """
        Getitem special method.
        
        Expects an integer value index, between 0 and len(self) - 1.
        
        Returns the image and its label as a one hot vector, both
        in torch tensor format in dataset.
        """
        
        # Get item special method
        first_val = int(list(self.dataset_numbers.values())[0])
        second_val = int(list(self.dataset_numbers.values())[1])
        #print('first_val: ',first_val)
        #print('second_val: ',second_val)
        if index < first_val:
            class_val = 'normal'
            label = torch.Tensor([1, 0])
        elif index >= first_val and index < first_val + second_val:
            class_val = 'infectednon'
            label = torch.Tensor([0, 1])
        elif index> second_val and index < 614:
            class_val = 'infectedcovid'
            index = index - (first_val+ second_val)
            label = torch.Tensor([0, 1])
        im = self.open_img(self.groups, class_val, index)
        im = transforms.functional.to_tensor(np.array(im)).float()
        return im, label
# Test code 

ld_test = Lung_Test_Dataset()
ld_test.describe()
#print('len: ', len(ld_test))
#im, class_oh = ld_test[500]
#print("im.shape: " , im.shape)
#print('im: ',im)
#print('class_oh: ', class_oh)

class Lung_Val_Dataset(Dataset):
    
    def __init__(self):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.
        """
        
        # All images are of size 150 x 150
        self.img_size = (150, 150)
        
        # Only two classes will be considered here (normal and infected)
        self.classes = {0: 'normal', 1: 'infectednon', 2:'infectedcovid'}
        
        # The dataset consists only of training images
        self.groups = 'val'
        
        # Number of images in each part of the dataset
        self.dataset_numbers = {'val_normal' : 8,                   'val_infectednon' : 8,                   'val_infectedcovid': 8}
        
        # Path to images for different parts of the dataset
        self.dataset_paths = { 'val_normal': './dataset/val/normal/',                         'val_infectednon': './dataset/val/infected/non-covid/',                         'val_infectedcovid': './dataset/val/infected/covid/'}
        
        
    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """
        
        # Generate description
        msg = "This is the test dataset of the Lung Dataset"
        msg += " used for the Small Project Demo in the 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
        msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        for key, val in self.dataset_paths.items():
            msg += " - {}, in folder {}: {} images.\n".format(key, val, self.dataset_numbers[key])
        print(msg)
        
    
    def open_img(self, group_val, class_val, index_val):
        """
        Opens image with specified parameters.
        
        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        
        Returns loaded image as a normalized Numpy array.
        """
        
        # Asserts checking for consistency in passed parameters
        err_msg = "Error - group_val variable should be set to 'train', 'test' or 'val'."
        assert group_val in self.groups, err_msg
        
        err_msg = "Error - class_val variable should be set to 'normal' or 'infected'."
        assert class_val in self.classes.values(), err_msg
        
        max_val = self.dataset_numbers['{}_{}'.format(group_val, class_val)]
        #print('max_val: ', max_val)
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
        err_msg += "\n(In {}/{}, you have {} images.)".format(group_val, class_val, max_val)
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val <= max_val, err_msg
        
        # Open file as before
        path_to_file = '{}/{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)
        with open(path_to_file, 'rb') as f:
            im = np.asarray(Image.open(f))/255
        f.close()
        return im
    
    
    def show_img(self, group_val, class_val, index_val):
        """
        Opens, then displays image with specified parameters.
        
        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """
        
        # Open image
        im = self.open_img(group_val, class_val, index_val)
        
        # Display
        plt.imshow(im)
        
        
    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """
        
        # Length function
        return sum(self.dataset_numbers.values())
    

    def __getitem__(self, index):
            """
            Getitem special method.

            Expects an integer value index, between 0 and len(self) - 1.

            Returns the image and its label as a one hot vector, both
            in torch tensor format in dataset.
            """

            # Get item special method
            first_val = int(list(self.dataset_numbers.values())[0])
            second_val = int(list(self.dataset_numbers.values())[1])
            class_val = ""
            #print('first_val: ',first_val)
            #print('second_val: ',second_val)
            if index < first_val:
                class_val = 'normal'
                label = torch.Tensor([1, 0])
                #print('choice1', index)
            elif index> first_val  and index < first_val +second_val:
                #print('choice2', index)
                index = index- first_val
                class_val = 'infected'
                label = torch.Tensor([0, 1])

            elif index> second_val and index < 24:
                #print('choice3', index)
                class_val = 'infected'
                index = index - (first_val + second_val)
                label = torch.Tensor([0, 1])
            im = self.open_img(self.groups, class_val, index)
            im = transforms.functional.to_tensor(np.array(im)).float()
            return im, label

        # Validation set Test code 

ld_val = Lung_Val_Dataset()
ld_val.describe()
#print('len: ', len(ld_val))
#im, class_oh = ld_val[22]
#print("im.shape: " , im.shape)
#print('im: ',im)
#print('class_oh: ', class_oh)

# Parameter 
bs_val = 4 
# Dataloader (train)(test)(val)
train_loader = DataLoader(ld_train, batch_size = bs_val, shuffle = True)
test_loader = DataLoader(ld_test, batch_size = bs_val, shuffle = True)
val_loader = DataLoader(ld_val, batch_size = bs_val, shuffle = True)

def get_data_obj():
    ld_train = Lung_Train_Dataset()
    ld_test = Lung_Test_Dataset()
    ld_val = Lung_Val_Dataset()
    return ld_train, ld_test, ld_val

