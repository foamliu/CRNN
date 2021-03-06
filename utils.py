import argparse
import logging
import os

import cv2 as cv
import torch

from config import max_target_len, dict, converter


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, hmean, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'hmean': hmean,
             'model': model,
             'optimizer': optimizer}
    # filename = 'checkpoint_' + str(epoch) + '_' + str(loss) + '.tar'
    filename = 'checkpoint.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_checkpoint.tar')


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']


def accuracy(inputs, input_lengths, labels, batch_size):
    n_correct = 0
    # print(preds.size())
    _, inputs = inputs.max(2)
    # print(preds.size())
    # preds = preds.squeeze(2)
    inputs = inputs.transpose(1, 0).contiguous().view(-1)
    sim_preds = converter.decode(inputs.data, input_lengths.data, raw=False)
    for pred, target in zip(sim_preds, labels):
        if pred == target:
            n_correct += 1
    accuracy = n_correct / float(batch_size)
    return accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Train CRNN network')
    # general
    parser.add_argument('--optimizer', default='adam', help='optimizer')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='start learning rate')
    parser.add_argument('--end-epoch', type=int, default=1000, help='training epoch size.')

    # optimizer
    parser.add_argument('--k', default=0.2, type=float,
                        help='tunable scalar multiply to learning rate')
    parser.add_argument('--warmup_steps', default=4000, type=int,
                        help='warmup steps')

    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    args = parser.parse_args()
    return args


def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def encode_target(target):
    return [dict[c] for c in target] + [0] * (max_target_len - len(target))


def get_images_for_test():
    from config import annotation_files
    split = 'test'
    print('loading {} annotation data...'.format('test'))
    annotation_file = annotation_files[split]
    with open(annotation_file, 'r') as file:
        lines = file.readlines()

    image_paths = [line.split(' ')[0] for line in lines]
    return image_paths
