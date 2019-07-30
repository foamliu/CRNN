import random

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm

import utils
from config import device, grad_clip, print_freq, num_workers, imgH, nc, nclass, nh, alphabet
from data_gen import MJSynthDataset
from models import CRNN

converter = utils.strLabelConverter(alphabet)


def train_net(args):
    manual_seed = 7
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    writer = SummaryWriter()
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        model = CRNN(imgH, nc, nclass, nh)
        model = nn.DataParallel(model)

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom,
                                        weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    logger = utils.get_logger()

    # Move to GPU, if available
    model = model.to(device)

    # Loss function
    criterion = nn.CTCLoss()

    # Custom dataloaders
    train_dataset = MJSynthDataset('train')
    train_loader = torch.utils.data.DataLoader() # train_dataset, batch_size=args.batch_size, shuffle=True,                                               num_workers=num_workers
    valid_dataset = MJSynthDataset('val')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                               num_workers=num_workers)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger)
        effective_lr = utils.get_learning_rate(optimizer)
        print('\nCurrent effective learning rate: {}\n'.format(effective_lr))

        writer.add_scalar('Train_Loss', train_loss, epoch)
        writer.add_scalar('Learning_Rate', effective_lr, epoch)

        # One epoch's validation
        valid_loss = valid(valid_loader=valid_loader,
                           model=model,
                           criterion=criterion,
                           logger=logger)
        writer.add_scalar('hmean', valid_loss, epoch)

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = max(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        utils.save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)


def train(train_loader, model, criterion, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = utils.AverageMeter()

    # Batches
    for i, (image, text) in enumerate(train_loader):
        # Move to GPU, if available
        image = image.to(device)
        text = text.to(device)
        batch_size = image.size(0)

        utils.loadData(image, image)
        t, l = converter.encode(text)
        utils.loadData(text, t)
        utils.loadData(length, l)

        print('text.size(): ' + str(text.size()))
        print('length.size(): ' + str(length.size()))

        # Forward prop.
        preds = model(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

        # Calculate loss
        loss = criterion(preds, text, preds_size, length)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        utils.clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item(), batch_size)

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), loss=losses))

    return losses.avg


def valid(valid_loader, model, criterion, logger):
    model.eval()  # train mode (dropout and batchnorm is used)

    losses = utils.AverageMeter()

    # Batches
    for image, text, length in tqdm(valid_loader):
        # Move to GPU, if available
        image = image.to(device)
        text = text.to(device)
        length = length.to(device)
        batch_size = image.size(0)

        # Forward prop.
        preds = model(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

        # Calculate loss
        loss = criterion(preds, text, preds_size, length)

        # Keep track of metrics
        losses.update(loss.item(), batch_size)

    # Print status
    logger.info('TEST Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(loss=losses))

    return losses.avg


def main():
    global args
    args = utils.parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
