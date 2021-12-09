import argparse
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from models import API_Net
from datasets import TestDataset, RandomDataset
from utils import accuracy, AverageMeter, save_checkpoint
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch ImageNet Testing')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size (default: 10)')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    global args
    args = parser.parse_args()
    # create model
    model = API_Net()
    model = model.to(device)
    model.conv = nn.DataParallel(model.conv)
    model.load_state_dict(dict(torch.load('checkpoint.pth'))['state_dict'])

    cudnn.benchmark = True
    # Data loading code
    test_dataset = TestDataset(transform=transforms.Compose([
                                            transforms.Resize([512, 512]), transforms.CenterCrop([448, 448]),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                mean=(0.485, 0.456, 0.406),
                                                std=(0.229, 0.224, 0.225)
                                            )]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    print('START TIME:', time.asctime(time.localtime(time.time())))
    test(test_loader, model)
    #test(val_loader, model)


def test(test_loader, model):
    f = open('test_result2.txt','w')
    predict_list = []
    model.eval()
    tqdm_gen = tqdm(test_loader)
    with torch.no_grad():
        for i, (img,label) in enumerate(tqdm_gen):
            img = img.to(device)
            logits = model(img, targets = None, flag = 'val')
            _, predict = logits.topk(1, 0, True, True)
            predict = int(predict.squeeze())
            predict_list.append(str(predict)+'\n')

    f.writelines(predict_list)
    f.close()

def eval_with_txt():
    f1 = open('datasets/stanford_cars/test_list.txt', 'r')
    f2 = open('test_result2.txt','r')
    f1 = f1.readlines()
    f2 = f2.readlines()
    summ = 0
    for i in range(8041):
        predict = int(f2[i])
        label = int(f1[i].split(' ')[1])
        summ += 1 if predict == label else 0

    print(summ/8041)

if __name__ == '__main__':
    main()
    eval_with_txt()
