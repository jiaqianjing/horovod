import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
import horovod.torch as hvd
import os
import math

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-dir', default=os.path.expanduser('~/imagenet/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('~/imagenet/validation'),
                    help='path to validation data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
allreduce_batch_size = args.batch_size * args.batches_per_allreduce
hvd.init()
torch.manual_seed(args.seed)
if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)

cudnn.benchmark = True

# Horovod: limit # of CPU threads to be used per worker.
torch.set_num_threads(1)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_dataset = \
    datasets.ImageFolder(args.train_dir,
                         transform=transforms.Compose([
                             transforms.RandomResizedCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                         ]))
# Horovod: use DistributedSampler to partition data among workers. Manually specify
# `num_replicas=hvd.size()` and `rank=hvd.rank()`.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=allreduce_batch_size,
    sampler=train_sampler, **kwargs)

val_dataset = \
    datasets.ImageFolder(args.val_dir,
                         transform=transforms.Compose([
                             transforms.Resize(256),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                         ]))
val_sampler = torch.utils.data.distributed.DistributedSampler(
    val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                         sampler=val_sampler, **kwargs)
# Set up standard ResNet-50 model.
model = models.resnet50()
# By default, Adasum doesn't need scaling up learning rate.
# For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
lr_scaler = args.batches_per_allreduce * hvd.size() if not args.use_adasum else 1

if args.cuda:
    # Move model to GPU.
    model.cuda()
    # If using GPU Adasum allreduce, scale learning rate by local_size.
    if args.use_adasum and hvd.nccl_built():
        lr_scaler = args.batches_per_allreduce * hvd.local_size()
# Horovod: scale learning rate by the number of GPUs.
optimizer = optim.SGD(model.parameters(),
                      lr=(args.base_lr * lr_scaler),
                      momentum=args.momentum,
                      weight_decay=args.wd)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(
    optimizer, named_parameters=model.named_parameters(),
    compression=compression,
    backward_passes_per_step=args.batches_per_allreduce,
    op=hvd.Adasum if args.use_adasum else hvd.Average)


# adjust learning rate on reset
def on_state_reset():
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * hvd.size()


def check_rank(epoch):
    if epoch == 2 and int(os.environ.get('HOROVOD_RANK')) == 0:
        print('exit rank {}'.format(hvd.rank()))
        raise RuntimeError('check_rank and exit')

def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()

@hvd.elastic.run
def train(state):
    # post synchronization event (worker added, worker removed) init ...
    for state.epoch in range(state.epoch, args.epochs + 1):
        state.model.train()
        train_sampler.set_epoch(state.epoch)
        steps_remaining = len(train_loader) - state.batch
        for state.batch, (data, target) in enumerate(train_loader):
            if state.batch >= steps_remaining:
                break
            check_rank(state.epoch)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            state.optimizer.zero_grad()
            output = state.model(data)
            acc = accuracy(output, target)
            loss = F.cross_entropy(output, target)
            loss.backward()
            # Gradient is applied across all ranks
            state.optimizer.step()
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tACC: {:.2f}%".format(
                state.epoch,
                state.batch * len(data),
                len(train_sampler),
                100.0 * state.batch / len(train_loader),
                loss.item(),
                100. * acc.item()
            ))
            state.commit()
        state.batch = 0


def test():
    pass


# Horovod: print logs on the first worker.
verbose = 1 if hvd.rank() == 0 else 0
state = hvd.elastic.TorchState(model, optimizer, epoch=1, batch=0)
state.register_reset_callbacks([on_state_reset])
train(state)
test()

