import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import os.path as osp
import nibabel as nib
import numpy as np

class MyDataset(Dataset):       
    def __init__(self, imageSamples, gtSamples, transform = None, target_transform = None):
        
        self.Patches = imageSamples
        self.LabelPatches = gtSamples
        self.transform = transform
        self.target_transform = target_transform
       
    def __getitem__(self, index):
        Patch = self.Patches[index,:,:,:,:]
        LabelPatch = self.LabelPatches[index,:,:,:,:]
        Patch = torch.Tensor(Patch)
        LabelPatch = np.int64(LabelPatch)
        LabelPatch = torch.LongTensor(LabelPatch)

        return Patch, LabelPatch
    
    def __len__(self):
        return len(self.Patches)

class MyDatasetTest(Dataset):       
    def __init__(self, imageSamples, transform = None, target_transform = None):
        
        self.Patches = imageSamples
        self.transform = transform
        self.target_transform = target_transform
       
    def __getitem__(self, index):
        Patch = self.Patches[index,:,:,:,:]
        Patch = torch.Tensor(Patch)

        return Patch
    
    def __len__(self):
        return len(self.Patches)


class DiceLoss(nn.Module):
	def __init__(self):
		super(DiceLoss, self).__init__()
 
	def	forward(self, input, target):
		N = target.size(0)
		smooth = 1
 
		input_flat = input.view(N, -1)
		target_flat = target.view(N, -1)
 
		intersection = input_flat * target_flat
 
		loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
		loss = 1 - loss.sum() / N
 
		return loss

class FCN(nn.Module):
    def __init__(self, n_classes):
        super(FCN, self).__init__()
        self.ec0 = self.encoder(1, 32, bias=True, padding=1, batchnorm=True)
        self.ec1 = self.encoder(32, 64, bias=True, padding=1, batchnorm=True)
        self.ec2 = self.encoder(64, 64, bias=True, padding=1, batchnorm=True)
        self.ec3 = self.encoder(64, 128, bias=True, batchnorm=True)
        self.ec4 = self.encoder(128, 128, bias=True, batchnorm=True)
        self.ec5 = self.encoder(128, 256, bias=True, batchnorm=True)
        self.ec6 = self.encoder(256, 256, bias=True, batchnorm=True)
        self.ec7 = self.encoder(256, 512, bias=True, batchnorm=True)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        self.dc9 = self.decoder(512, 512, kernel_size=2, stride=2, bias=True)
        self.dc8 = self.decoder(256+512, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.dc7 = self.decoder(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.dc6 = self.decoder(256, 256, kernel_size=2, stride=2, bias=True)
        self.dc5 = self.decoder(128+256, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dc4 = self.decoder(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dc3 = self.decoder(128, 128, kernel_size=2, stride=2, bias=True)
        self.dc2 = self.decoder(64+128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.dc1 = self.decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.dc0 = self.decoder(64, n_classes, kernel_size=1, stride=1, bias=True)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=True):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Dropout3d(0.1),
                nn.ReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU(inplace=True))
        return layer


    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.Dropout3d(0.1),
            nn.ReLU(inplace=True))
        return layer

    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
           c2 = (bypass.size()[2] - upsampled.size()[2])//2
           c3 = (bypass.size()[3] - upsampled.size()[3])//2
           c4 = (bypass.size()[4] - upsampled.size()[4])//2
           bypass = F.pad(bypass, (-c4, -c4, -c3, -c3, -c2, -c2))
           if bypass.size()[2]!=upsampled.size()[2] or bypass.size()[3]!=upsampled.size()[3] or bypass.size()[4]!=upsampled.size()[4] :
               coord_size1 = upsampled.size()[2]
               coord_size2 = upsampled.size()[3]
               coord_size3 = upsampled.size()[4]
               bypass = bypass[:,:,0:coord_size1, 0:coord_size2, 0:coord_size3]
           return torch.cat((upsampled, bypass), 1)


    def forward(self, x):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)

        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)
        del e3, e4

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)
        del e5, e6

        d9 = self.crop_and_concat(syn2, self.dc9(e7), True)
        del e7, syn2

        d8 = self.dc8(d9)
        d7 = self.dc7(d8)
        del d9, d8
        
        d6 = self.crop_and_concat(syn1, self.dc6(d7), True)
        del d7, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        del d6, d5

        d3 = self.crop_and_concat(syn0, self.dc3(d4), True)
        del d4, syn0

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2
          
        d0 = self.dc0(d1)
        return d0
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def batch_whole_dice(truth, predict, n_class):
    All_and = 0
    All_P1 = 0
    All_T1 = 0
    for label in range(1, n_class):
        predict_l = predict==label
        truth_l = truth==label
        P1 = np.count_nonzero(predict_l)
        T1 = np.count_nonzero(truth_l)
        TP = np.logical_and(truth_l, predict_l)
        TP_count = np.count_nonzero(TP)
        All_and = All_and + TP_count
        All_P1 = All_P1 + P1
        All_T1 = All_T1 + T1
    return (All_P1, All_T1, All_and)



def train(model, train_loader, optimizer, n_class, epoch):
    model.train()
    LossFunc = DiceLoss()

    for batch_idx, (data, target) in enumerate(train_loader):
        correct = 0
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output_score = model(data)
        n, c, h, w, d = output_score.size()
        output = output_score.transpose(1,2).transpose(2,3).transpose(3,4).contiguous().view(-1, c)
        m = torch.nn.LogSoftmax()
        log_p = m(output)
       
        mask = target >= 0
        target = target[mask]

        loss = F.nll_loss(log_p, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t score_max: {:.6f}\t score_min: {:.6f}'.format(
                   epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss, output_score[0,0,:,:,22].data.max(), output_score[0,0,:,:,22].data.min()))

        pred = log_p.data.max(1, keepdim=True)[1]
        target = target.data.view_as(pred)
        pred = pred.cpu()
        target = target.cpu() 
        (T1, P1, inter) = batch_whole_dice(pred, target, n_class)
        if T1+P1 == 0:
            Dice = 0
        else:
            Dice = 100.*float(inter)*2/(float(T1)+float(P1))
            print('the total Dice: {:.4f}%'.format(Dice))

def valid(args, model, validate_loader):
    model.eval()
    sumDice = 0
    for batch_idx, (valdata, valtarget) in enumerate(validate_loader):
        valdata, valtarget = Variable(valdata.cuda(), requires_grad=False), Variable(valtarget.cuda(), requires_grad=False)
        with torch.no_grad():
            output_score = model(valdata)
            nn, cc, hh, ww, dd = output_score.size()
            output = output_score.transpose(1,2).transpose(2,3).transpose(3,4).contiguous().view(-1,cc)
            mm = torch.nn.LogSoftmax()
            log_pp = mm(output)
            valmask = valtarget >= 0
            valtarget = valtarget[valmask]
            valpred = log_pp.data.max(1, keepdim=True)[1]

            (T1, P1, inter) = batch_whole_dice(valpred, valtarget.data.view_as(valpred),1)
            if T1+P1 == 0:
                Dice = 0
            else:
                Dice = 100.*float(inter)*2/(float(T1)+float(P1))
            print('the total Dice: {:.4f}%'.format(Dice))
            sumDice += Dice
    return sumDice/len(validate_loader.dataset)

def test(model, test_loader):
    model.eval()
    progg = 0
    for batch_idx, (testdata) in enumerate(test_loader):
            testdata = Variable(testdata.cuda())
            with torch.no_grad():
                output_score = model(testdata)
                nn, cc, hh, ww, dd = output_score.size()
                output = output_score.transpose(1,2).transpose(2,3).transpose(3,4).contiguous().view(-1,cc)
                mm = torch.nn.LogSoftmax()
                log_pp = mm(output)
                probmm = torch.nn.Softmax()
                pp = probmm(output)
            
                cubic_pp = pp.contiguous().view(nn, hh, ww,  dd, cc)
                cubic_pp = cubic_pp.transpose(3,4).transpose(2,3).transpose(1,2)
                pro_1 = cubic_pp[:,:,:,:,:]
                pro_cpu = pro_1.cpu()
                progg = pro_cpu.detach().numpy()
    return progg
