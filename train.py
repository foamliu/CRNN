import random

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from warpctc_pytorch import CTCLoss

from config import device, imgH, imgW, keep_ratio, grad_clip, print_freq, num_workers
from data_gen import lmdbDataset, alignCollate
from models import CRNN
from utils import parse_args, save_checkpoint, AverageMeter, clip_gradient, get_logger, get_learning_rate


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
        model = CRNN(args)
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

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)

    # Loss function
    criterion = CTCLoss()

    collate_fn = alignCollate(imgH=imgH, imgW=imgW, keep_ratio=keep_ratio)

    # Custom dataloaders
    train_dataset = lmdbDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=num_workers)
    valid_dataset = lmdbDataset('test')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                               shuffle=False, num_workers=num_workers)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger)
        effective_lr = get_learning_rate(optimizer)
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
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)


def train(train_loader, model, criterion, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()

    # Batches
    for i, (images, texts) in enumerate(train_loader):
        # Move to GPU, if available
        images = images.to(device)
        texts = texts.to(device)
        batch_size = images.size(0)

        # Forward prop.
        preds = model(images)

        # Calculate loss
        loss = criterion(preds, text, preds_size, length)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

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

    losses = AverageMeter()

    # Batches
    for img, score_map, geo_map, training_mask in tqdm(valid_loader):
        # Move to GPU, if available
        img = img.to(device)
        score_map = score_map.to(device)
        geo_map = geo_map.to(device)
        training_mask = training_mask.to(device)

        # Forward prop.
        f_score, f_geometry = model(img)

        # Calculate loss
        loss = criterion(score_map, f_score, geo_map, f_geometry, training_mask)

        # Keep track of metrics
        losses.update(loss.item(), img.size(0))

    # Print status
    logger.info('TEST Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(loss=losses))

    return losses.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
