import torch
import random
import numpy as np
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

class GAN_attack:
    def __init__(self, dataset, ratio_adv, device='cuda', random_state=42):
        self.dataset = dataset
        self.ratio_adv = ratio_adv
        self.device = device
        self.random_state = random_state
        self.generator_path = ''

        self.subsample = None
        self.adv_index = []
        self.original_labels = None
        self.xadv = None
        self.original = None
        
        random.seed(random_state)
        torch.manual_seed(random_state)
        
    def load_generator(self):
        if self.dataset == 'MNIST':
            self.generator_path = './Generator_Models/MNIST.pth'
            image_nc = 1
            gen_input_nc = 1
        if self.dataset == 'FMNIST':
            self.generator_path = './Generator_Models/FMNIST.pth'
            image_nc = 1
            gen_input_nc = 1
        
        self.generator = Generator(gen_input_nc, image_nc).to(self.device)
        self.generator.load_state_dict(torch.load(self.generator_path))
        self.generator.eval()
        
    def load_dataset(self):
        if self.dataset == 'MNIST':
            dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
        if self.dataset == 'FMNIST':
            dataset = torchvision.datasets.FashionMNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
        
        #have b.s as 1 to ease the process.
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
        
    def reset(self):
        self.adv_index = []
        self.original_labels = None
        self.samples = None
        
    def generate_samples(self, length = 2000, clamp=0.1, subsample = [0,1,2,3,4,5,6,7,8,9]):
        """
        returns: data in the form of 2-D numpy array, if 2000 samples and MNIST, then shape would be (2000, 784)
        """
        self.load_generator()
        self.load_dataset()
        self.subsample = subsample
       
        no_adv_samples = int(self.ratio_adv*length)
        self.adv_index = [random.randint(0, length) for _ in range(no_adv_samples)] #randomly generate adv_indexes
    
        #generating samples
        for i, data in enumerate(self.dataloader):
            if self.xadv is not None and self.xadv.shape[0] >= length:
                break
            img, label = data
            
            if label in self.subsample:
                img, label = img.to(self.device), label.to(self.device)
                img_orig = torch.clone(img)

                if i in self.adv_index:
                    perturbation = self.generator(img)
                    perturbation = torch.clamp(perturbation, -clamp, clamp)
                    adv_img = perturbation + img
                    img = torch.clamp(adv_img, 0, 1)

                npimg = img.detach().cpu().numpy()
                nplabel = label.detach().cpu().numpy()
                npimg = npimg.reshape(npimg.shape[0], npimg.shape[1]*npimg.shape[2]*npimg.shape[3]) #1,1,28,28 -> 1, 1*28*28 (MNIST)
                #for keeping track of original samples
                npimg_o = img_orig.detach().cpu().numpy()
                npimg_o = npimg_o.reshape(npimg_o.shape[0], npimg_o.shape[1]*npimg_o.shape[2]*npimg_o.shape[3]) #1,1,28,28 -> 1, 1*28*28 (MNIST)

                if self.xadv is None:
                    self.xadv = npimg
                    self.original = npimg_o
                    self.original_labels = nplabel
                else:
                    self.xadv = np.concatenate((self.xadv, npimg))
                    self.original = np.concatenate((self.original, npimg_o))
                    self.original_labels = np.concatenate((self.original_labels, nplabel))
                
        return self.xadv
    
#--------------------------------- Do not touch below this line ----------------------------------------------

class Generator(nn.Module):
    def __init__(self,
                 gen_input_nc,
                 image_nc,
                 ):
        super(Generator, self).__init__()

        encoder_lis = [
            # MNIST:1*28*28
            nn.Conv2d(gen_input_nc, 8, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # 8*26*26
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # 16*12*12
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # 32*5*5
        ]

        bottle_neck_lis = [ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),]

        decoder_lis = [
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # state size. 16 x 11 x 11
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # state size. 8 x 23 x 23
            nn.ConvTranspose2d(8, image_nc, kernel_size=6, stride=1, padding=0, bias=False),
            nn.Tanh()
            # state size. image_nc x 28 x 28
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x


# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out  