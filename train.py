import random

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm

import data_gen
import utils
from config import device, print_freq, num_workers, imgH, num_channels, num_classes, num_hidden, max_target_len
from models import CRNN


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

    # custom weights initialization called on crnn
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    # Initialize / load checkpoint
    if checkpoint is None:
        model = CRNN(imgH, num_channels, num_classes, num_hidden)
        model.apply(weights_init)
        # model = nn.DataParallel(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09)
        # optimizer = CRNNOptimizer(
        #     torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        #     args.k,
        #     num_hidden,
        #     args.warmup_steps)

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
    criterion = nn.CTCLoss(reduction='mean').to(device)

    # Custom dataloaders
    train_dataset = data_gen.MJSynthDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=num_workers)
    valid_dataset = data_gen.MJSynthDataset('val')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                               num_workers=num_workers)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        # One epoch's training
        train_loss, train_acc = train(train_loader=train_loader,
                                      model=model,
                                      criterion=criterion,
                                      optimizer=optimizer,
                                      epoch=epoch,
                                      logger=logger)
        writer.add_scalar('Train_Loss', train_loss, epoch)
        writer.add_scalar('Train_Accuracy', train_acc, epoch)

        # One epoch's validation
        valid_loss, valid_acc = valid(valid_loader=valid_loader,
                                      model=model,
                                      criterion=criterion,
                                      logger=logger)
        writer.add_scalar('Validation_Loss', valid_loss, epoch)
        writer.add_scalar('Validation_Accuracy', valid_acc, epoch)

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
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
    accs = utils.AverageMeter()

    # Batches
    for i, (images, labels) in enumerate(train_loader):
        # Move to GPU, if available
        images = images.to(device)
        batch_size = images.size(0)

        target_lengths = [min(max_target_len, len(t)) for t in labels]
        target_lengths = torch.LongTensor(target_lengths)  # size (batch size)

        targets = [utils.encode_target(t[:max_target_len]) for t in labels]
        targets = torch.LongTensor(targets).to(device)  # size (batch size, max target length)

        # Forward prop.
        inputs = model(images)  # size (input length, batch size, number of classes)
        input_lengths = Variable(torch.IntTensor([inputs.size(0)] * batch_size))  # size (batch size)

        # Calculate loss
        loss = criterion(inputs, targets, input_lengths, target_lengths)
        acc = utils.accuracy(inputs, input_lengths, labels, batch_size)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        # utils.clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item(), batch_size)
        accs.update(acc, batch_size)

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Accuracy {acc.val:.4f} ({acc.avg:.4f})'.format(epoch, i, len(train_loader), loss=losses,
                                                                        acc=accs))

    return losses.avg, accs.avg


def valid(valid_loader, model, criterion, logger):
    model.eval()  # eval mode (dropout and batchnorm is not effective)

    losses = utils.AverageMeter()
    accs = utils.AverageMeter()

    # Batches
    for images, labels in tqdm(valid_loader):
        # Move to GPU, if available
        images = images.to(device)
        batch_size = images.size(0)

        target_lengths = [min(max_target_len, len(t)) for t in labels]
        target_lengths = torch.LongTensor(target_lengths)  # size (batch size)

        targets = [utils.encode_target(t[:max_target_len]) for t in labels]
        targets = torch.LongTensor(targets).to(device)  # size (batch size, max target length)

        # Forward prop.
        with torch.no_grad():
            inputs = model(images)  # size (input length, batch size, number of classes)
            input_lengths = Variable(torch.IntTensor([inputs.size(0)] * batch_size))  # size (batch size)

        # Calculate loss
        loss = criterion(inputs, targets, input_lengths, target_lengths)
        acc = utils.accuracy(inputs, input_lengths, labels, batch_size)

        # Keep track of metrics
        losses.update(loss.item(), batch_size)
        accs.update(acc, batch_size)

    # Print status
    logger.info('Validation Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {acc.val:.4f} ({acc.avg:.4f})\n'.format(loss=losses, acc=accs))

    return losses.avg, accs.avg


def main():
    global args
    args = utils.parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
