import argparse
import datetime
import logging
import os
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import TFIDFDataset
from model import MLP
from loss import cross_entropy
from utils import AverageMeter

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
torch.manual_seed(0)

logging.basicConfig(filename='logs.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)
# logging.basicConfig(level=logging.INFO)

# configurations
parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str,
                    default='/media/vutrungnghia/New Volume/DSLab/dataset/train_tf_idf.txt')
parser.add_argument('--valid_path', type=str,
                    default='/media/vutrungnghia/New Volume/DSLab/dataset/test_tf_idf.txt')
parser.add_argument('--word_idfs_path', type=str,
                    default='/media/vutrungnghia/New Volume/DSLab/dataset/words_idfs.txt')
parser.add_argument('--weight', type=str,
                    default='')
parser.add_argument('--models_folder', type=str,
                    default='/media/vutrungnghia/New Volume/DSLab/models')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_epochs', type=int, default=10)
args = parser.parse_args()
logging.info('\n=========== MLP =============\n')
logging.info(args._get_kwargs())

# create train/valid dataset, dataloader
train_dataset = TFIDFDataset(args.train_path, args.word_idfs_path)
train_dataloader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
valid_dataset = TFIDFDataset(args.valid_path, args.word_idfs_path)
valid_dataloader = DataLoader(valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)

# create and optionally load model, create optimizer
model = MLP()
start_epoch = 1
if args.weight:
    checkpoint = torch.load(args.weight)
    model.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['epoch']
    logging.info(f'Load model at: {args.weight}')
if torch.cuda.is_available():
    model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.001,
                             betas=(0.9, 0.999))

# training and evaluate
for epoch in range(start_epoch, args.n_epochs):
    logging.info(f'EPOCH: {epoch}/{args.n_epochs}')

    # ******************* TRANING PHASE ********************
    model.train()
    avg_loss = AverageMeter()
    logging.info(f'  TRAINING PHASE:')
    for X, labels in tqdm(train_dataloader):
        if torch.cuda.is_available():
            X = X.cuda().type(torch.cuda.FloatTensor)
            labels = labels.cuda().type(torch.cuda.FloatTensor)
        else:
            X = X.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)
        preds = model(X)
        loss = cross_entropy(preds, labels)
        avg_loss.update(loss.item(), X.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logging.info(f'    avg loss: {avg_loss.avg}')
    x = datetime.datetime.now()
    time = x.strftime("%y-%m-%d_%H-%M-%S")
    model_checkpoint = os.path.join(args.models_folder, f'checkpoint_{time}_{epoch}.pth')
    torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, model_checkpoint)
    logging.info(f'    {model_checkpoint}')

    # ****************** VALIDATE PHASE ********************
    model.eval()
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()
    logging.info(f'  VALIDATING PHASE:')
    for X, labels in tqdm(valid_dataloader):
        labels = labels.type(torch.IntTensor)
        if torch.cuda.is_available():
            X = X.cuda().type(torch.cuda.FloatTensor)
        else:
            X = X.type(torch.FloatTensor)

        with torch.no_grad():
            preds = model(X).cpu()
        loss = cross_entropy(preds, labels.type(torch.FloatTensor))
        avg_loss.update(loss.item(), X.shape[0])

        labels = np.argmax(labels, axis=1)
        preds = np.argmax(preds, axis=1)
        acc = (sum(labels == preds)) * 1.0 / X.shape[0]
        avg_acc.update(acc, X.shape[0])
    logging.info(f'val_loss: {avg_loss.avg} - val_acc: {avg_acc.avg}')
