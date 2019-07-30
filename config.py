import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

imgH = 32
imgW = 100
keep_ratio = True

nc = 1
nh = 256

alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
nclass = len(alphabet) + 1

IMG_FOLDER = 'mnt/ramdisk/max/90kDICT32px/'

annotation_files = {'train': 'mnt/ramdisk/max/90kDICT32px/annotation_train.txt',
                    'val': 'mnt/ramdisk/max/90kDICT32px/annotation_val.txt',
                    'test': 'mnt/ramdisk/max/90kDICT32px/annotation_test.txt'}

# Training parameters
num_workers = 4  # for data-loading
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 10  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none
