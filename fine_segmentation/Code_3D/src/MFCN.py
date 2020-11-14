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
    def __init__(self, imageSamples, gtSamples, similarImageSamples, gtsimilarSamples, transform = None, target_transform = None):
        
        self.Patches = imageSamples
        self.LabelPatches = gtSamples
        self.n1Patches = similarImageSamples[:,0,:,:,:]
        self.n1LabelPatches = gtsimilarSamples[:,0,:,:,:]
        self.n2Patches = similarImageSamples[:,1,:,:,:]
        self.n2LabelPatches = gtsimilarSamples[:,1,:,:,:]
        self.n3Patches = similarImageSamples[:,2,:,:,:]
        self.n3LabelPatches = gtsimilarSamples[:,2,:,:,:]
        self.transform = transform
        self.target_transform = target_transform
       
    def __getitem__(self, index):
        Patch = self.Patches[index,:,:,:,:]
        LabelPatch = self.LabelPatches[index,:,:,:,:]
        Patch = torch.Tensor(Patch)
        LabelPatch = np.int64(LabelPatch)
        LabelPatch = torch.LongTensor(LabelPatch)

        n1Patch = self.n1Patches[index,:,:,:]
        n1LabelPatch = self.n1LabelPatches[index,:,:,:]
        n1Patch = torch.Tensor(n1Patch)
        n1Patch = n1Patch.unsqueeze(0)
        n1LabelPatch = np.int64(n1LabelPatch)
        n1LabelPatch = torch.Tensor(n1LabelPatch)
        n1LabelPatch = n1LabelPatch.unsqueeze(0)

        n2Patch = self.n2Patches[index,:,:,:]
        n2LabelPatch = self.n2LabelPatches[index,:,:,:]
        n2Patch = torch.Tensor(n2Patch)
        n2Patch = n2Patch.unsqueeze(0)
        n2LabelPatch = np.int64(n2LabelPatch)
        n2LabelPatch = torch.Tensor(n2LabelPatch)
        n2LabelPatch = n2LabelPatch.unsqueeze(0)
        
        n3Patch = self.n3Patches[index,:,:,:]
        n3LabelPatch = self.n3LabelPatches[index,:,:,:]
        n3Patch = torch.Tensor(n3Patch)
        n3Patch = n3Patch.unsqueeze(0)
        n3LabelPatch = np.int64(n3LabelPatch)
        n3LabelPatch = torch.Tensor(n3LabelPatch)
        n3LabelPatch = n3LabelPatch.unsqueeze(0)

        return Patch, LabelPatch, n1Patch, n1LabelPatch, n2Patch, n2LabelPatch, n3Patch, n3LabelPatch
    
    def __len__(self):
        return len(self.Patches)

class MyDatasetTest(Dataset):       
    def __init__(self, imageSamples, similarImageSamples, gtsimilarSamples, transform = None, target_transform = None):
        
        self.Patches = imageSamples
        self.n1Patches = similarImageSamples[:,0,:,:,:]
        self.n1LabelPatches = gtsimilarSamples[:,0,:,:,:]
        self.n2Patches = similarImageSamples[:,1,:,:,:]
        self.n2LabelPatches = gtsimilarSamples[:,1,:,:,:]
        self.n3Patches = similarImageSamples[:,2,:,:,:]
        self.n3LabelPatches = gtsimilarSamples[:,2,:,:,:]
        self.transform = transform
        self.target_transform = target_transform
       
    def __getitem__(self, index):
        Patch = self.Patches[index,:,:,:,:]
        Patch = torch.Tensor(Patch)

        n1Patch = self.n1Patches[index,:,:,:]
        n1LabelPatch = self.n1LabelPatches[index,:,:,:]
        n1Patch = torch.Tensor(n1Patch)
        n1Patch = n1Patch.unsqueeze(0)
        n1LabelPatch = np.int64(n1LabelPatch)
        n1LabelPatch = torch.Tensor(n1LabelPatch)
        n1LabelPatch = n1LabelPatch.unsqueeze(0)

        n2Patch = self.n2Patches[index,:,:,:]
        n2LabelPatch = self.n2LabelPatches[index,:,:,:]
        n2Patch = torch.Tensor(n2Patch)
        n2Patch = n2Patch.unsqueeze(0)
        n2LabelPatch = np.int64(n2LabelPatch)
        n2LabelPatch = torch.Tensor(n2LabelPatch)
        n2LabelPatch = n2LabelPatch.unsqueeze(0)
        
        n3Patch = self.n3Patches[index,:,:,:]
        n3LabelPatch = self.n3LabelPatches[index,:,:,:]
        n3Patch = torch.Tensor(n3Patch)
        n3Patch = n3Patch.unsqueeze(0)
        n3LabelPatch = np.int64(n3LabelPatch)
        n3LabelPatch = torch.Tensor(n3LabelPatch)
        n3LabelPatch = n3LabelPatch.unsqueeze(0)

        return Patch, n1Patch, n1LabelPatch, n2Patch, n2LabelPatch, n3Patch, n3LabelPatch
    
    def __len__(self):
        return len(self.Patches)

class FCN(nn.Module):
    def __init__(self, n_classes):
        super(FCN, self).__init__()
        self.ec01 = self.encoder(2, 32, bias=True, padding=1, batchnorm=True)
        self.ec02 = self.encoder(2, 32, bias=True, padding=1, batchnorm=True)
        self.ec03 = self.encoder(2, 32, bias=True, padding=1, batchnorm=True)
        self.ec0 = self.encoder(1, 32, bias=True, padding=1, batchnorm=True)
        self.ec11 = self.encoder(32, 64, bias=True, padding=1, batchnorm=True)
        self.ec12 = self.encoder(32, 64, bias=True, padding=1, batchnorm=True)
        self.ec13 = self.encoder(32, 64, bias=True, padding=1, batchnorm=True)
        self.ec1 = self.encoder(32, 64, bias=True, padding=1, batchnorm=True)
        self.ec2 = self.encoder(256, 64, bias=True, padding=1, batchnorm=True)
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
        print(n_classes)
        #self._initialize_weights()

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=True):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Dropout3d(0.1),
                #nn.Tanh())
                #nn.ReLU(inplace=True))
                nn.PReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                #nn.Tanh())
                #nn.ReLU(inplace=True))
                nn.PReLU())
        return layer


    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.Dropout3d(0.1),
            #nn.Tanh())
            nn.PReLU())
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


    def forward(self, x, n1, nl1, n2, nl2, n3, nl3):
        e0 = self.ec0(x)
        N1 = torch.cat((nl1, n1), 1)
        N2 = torch.cat((nl2, n2), 1)
        N3 = torch.cat((nl3, n3), 1)
        e01 = self.ec01(N1)
        e02 = self.ec01(N2)
        e03 = self.ec01(N3)
        syn1 = self.ec11(e01)
        syn2 = self.ec12(e02)
        syn3 = self.ec13(e03)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e11 = self.pool0(syn1)
        e12 = self.pool0(syn2)
        e13 = self.pool0(syn3)

        concate = torch.cat((e1, e13, e12, e11), 1)
        e2 = self.ec2(concate)
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

"""
class ImbLossFunc(nn.Module):
    def __init__(self):
       super(ImbLossFunc, self).__init__()
       #self.t1 = t1
       #self.t2 = t2
       return

    def forward(self, predict, target, dice, n_class):
        Diff = torch.zeros(predict.shape[0])
        for i in xrange(predict.shape[0]):
            Diff[i] = -predict[i][target[i]]
        return torch.sum(Diff)
"""


def batch_whole_dice(truth, predict, n_class):
    All_and = 0
    All_P1 = 0
    All_T1 = 0
    #for label in xrange(1, n_class):
    for label in xrange(2, 3):
        predict = predict==label
        truth = truth==label
        P1 = np.count_nonzero(predict)
        T1 = np.count_nonzero(truth)
        TP = np.logical_and(truth, predict)
        TP_count = np.count_nonzero(TP)
        All_and = All_and + TP_count
        All_P1 = All_P1 + P1
        All_T1 = All_T1 + T1
    return (All_P1, All_T1, All_and)



def train(model, train_loader, optimizer, n_class, epoch, Dice):
    model.train()
    #LossFunc = ImbLossFunc()
    #dice = (1-Dice)/np.sum(1-Dice)
    #dice = torch.FloatTensor(dice)
    #dice = Variable(dice.cuda())
    for batch_idx, (data, target, n1, nl1, n2, nl2, n3, nl3) in enumerate(train_loader):
        correct = 0
        data, target = Variable(data.cuda()), Variable(target.cuda())
        n1, nl1 = Variable(n1.cuda()), Variable(nl1.cuda())
        n2, nl2 = Variable(n2.cuda()), Variable(nl2.cuda())
        n3, nl3 = Variable(n3.cuda()), Variable(nl3.cuda())
        output_score = model(data, n1, nl1, n2, nl2, n3, nl3)
        n, c, h, w, d = output_score.size()
        output = output_score.transpose(1,2).transpose(2,3).transpose(3,4).contiguous().view(-1, c)
        m = torch.nn.LogSoftmax()
        log_p = m(output)
        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(log_p, target)
        #print(log_p.shape,target.shape)
        #loss = LossFunc(log_p, target, dice, n_class)
        #print("_+_+_+_",loss)
        #print(target.shape)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t score_max: {:.6f}\t score_min: {:.6f}'.format(
                   epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss, output_score[0,0,:,:,22].data.max(), output_score[0,0,:,:,22].data.min()))

        pred = log_p.data.max(1, keepdim=True)[1]
        (T1, P1, inter) = batch_whole_dice(pred, target.data.view_as(pred), n_class)
        if T1+P1 == 0:
            Dice = 0
        else:
            Dice = 100.*float(inter)*2/(float(T1)+float(P1))
            print('the total Dice: {:.4f}%'.format(Dice))

def valid(args, model, validate_loader):
    model.eval()
    sumDice = 0
    for batch_idx, (valdata, valtarget, valn1, valnl1, valn2, valnl2, valn3, valnl3) in enumerate(validate_loader):
        valdata, valtarget = Variable(valdata.cuda(), requires_grad=False), Variable(valtarget.cuda(), requires_grad=False)
        valn1, valnl1 = Variable(valn1.cuda(), requires_grad=False), Variable(valnl1.cuda(), requires_grad=False)
        valn2, valnl2 = Variable(valn2.cuda(), requires_grad=False), Variable(valnl2.cuda(), requires_grad=False)
        valn3, valnl3 = Variable(valn3.cuda(), requires_grad=False), Variable(valnl3.cuda(), requires_grad=False)
        with torch.no_grad():
            output_score = model(valdata, valn1, valnl1, valn2, valnl2, valn3, valnl3)
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
    for batch_idx, (testdata, testn1, testnl1, testn2, testnl2, testn3, testnl3) in enumerate(test_loader):
            testdata = Variable(testdata.cuda())
            testn1, testnl1 = Variable(testn1.cuda()), Variable(testnl1.cuda())
            testn2, testnl2 = Variable(testn2.cuda()), Variable(testnl2.cuda())
            testn3, testnl3 = Variable(testn3.cuda()), Variable(testnl3.cuda())
            with torch.no_grad():
                output_score = model(testdata, testn1, testnl1, testn2, testnl2, testn3, testnl3)
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
