import argparse, os
import torch 
import math, random 
from model.srresnet import NetG
from torchvision import models
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader
from dataset import BasicDataset
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import numpy as np
from torchvision import transforms, utils

dir_img = 'Data/input/'
dir_mask = 'Data/output/'

parser = argparse.ArgumentParser(description="PyTorch Version of SRResNet")
parser.add_argument("--lr",             type=float, default=1e-4, help="Learning rate")
parser.add_argument("--start-epoch",    type=int,   default=1,    help="Manual epoch number")
parser.add_argument("--nEpochs",        type=int,   default=500,  help="Number of epochs to train for")
parser.add_argument("--gpus",           type=str,   default=0,    help="gpu ids (default: 0)")
parser.add_argument("--threads",        type=int,   default=0,    help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--batchSize",      type=int,   default=1,    help="training batch size")
parser.add_argument("--step",           type=int,   default=200,  help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--cuda",           action="store_true",      help="Use cuda")
parser.add_argument("--vgg_loss",       action="store_true",      help="Use content loss?")


def main():
    
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpus)
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong GPU id, please run without --cuda")

    print("===> Loading datasets")

    dataset = BasicDataset(dir_img, dir_mask,transform= transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225])
                                                                            ]))
    training_data_loader = DataLoader(dataset=dataset, num_workers=opt.threads, \
        batch_size=opt.batchSize, shuffle=True)

    if opt.vgg_loss:
        print("===> Loading VGG model")
        netVGG = models.vgg19()
        netVGG.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'))
        class _content_model(nn.Module):
            def __init__(self):
                super(_content_model, self).__init__()
                self.feature = nn.Sequential(*list(netVGG.features.children())[:-1])
                
            def forward(self, x):
                out = self.feature(x)
                return out

        netContent = _content_model()
    print("===> Building model")
    model = NetG()
    criterion = nn.MSELoss()

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs +1):
        try:
            os.makedirs('Output/{}'.format(epoch))
        except Exception as e:
            print(e)

        train(training_data_loader, optimizer, model, criterion, epoch, opt, netContent)

def adjust_learning_rate(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr 
    
def train(training_data_loader, optimizer, model, criterion, epoch, opt, netContent):

    lr = adjust_learning_rate(optimizer, epoch-1, opt)
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):

        input, target = Variable(batch['image']), Variable(batch['mask'], requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()
            netContent = netContent.cuda()

        output = model(input)
        loss = criterion(output, target)

        if opt.vgg_loss:
            content_input = netContent(output)
            content_target = netContent(target)
            content_target = content_target.detach()
            content_loss = criterion(content_input, content_target)

        optimizer.zero_grad()

        if opt.vgg_loss:
            netContent.zero_grad()
            content_loss.backward(retain_graph=True)

        loss.backward()

        optimizer.step()

        if iteration%20 == 0:
            if opt.vgg_loss:
                print("===> Epoch[{}]({}/{}): Loss: {:.5} Content_loss {:.5}".format(epoch, iteration, len(training_data_loader), loss.detach().cpu().numpy(), content_loss.detach().cpu().numpy()))
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                fig.suptitle('Horizontally stacked subplots')
                ax1.imshow(np.array(input.detach().cpu()[0]).transpose((1,2,0)).astype(float))
                ax2.imshow(np.array(output.detach().cpu()[0]).transpose((1,2,0)).astype(float))
                ax3.imshow(np.array(target.detach().cpu()[0]).transpose((1,2,0)).astype(float))
                plt.savefig('Output/{}/{}.png'.format(epoch, iteration))
            else:
                print("===> Epoch[{}]({}/{}): Loss: {:.5}".format(epoch, iteration, len(training_data_loader), loss.data[0]))


if __name__ == "__main__":
    main()