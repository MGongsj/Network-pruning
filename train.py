import os
import sys
import torch.utils
import torch.utils.data.distributed
import torchvision
sys.path.append("../../")
from utils.utils import *
from torchvision import datasets, transforms
from resnet import resnet

parser = argparse.ArgumentParser("ResNet")
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--print_freq', type=float, default=1, help='report frequency')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
args = parser.parse_args()

CLASSES = 200

if not os.path.exists('log'):
    os.mkdir('log')

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join('log/log00.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)



def main():
    if not torch.cuda.is_available():
        sys.exit(1)
    start_t = time.time()

    cudnn.benchmark = True
    cudnn.enabled=True

    model = resnet()
    logging.info(model)

    model = nn.DataParallel(model).cuda()
    para = list(model.parameters())
    #logging.info(para)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()
    # BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
    # BCEWithLogitsLoss = BCEWithLogitsLoss.cuda()




    optimizer = torch.optim.SGD(
        # [{'params' : other_parameters},
        # {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        )

    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/args.epochs), last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs//4, args.epochs//2, args.epochs//4*3], gamma=0.1)
    start_epoch = 0
    best_top1_acc= 0
    checkpoint_tar = os.path.join(args.save, 'checkpoint_cut.pth.tar')
    if os.path.exists(checkpoint_tar):
        logging.info('loading checkpoint {} ..........'.format(checkpoint_tar))
        checkpoint = torch.load(checkpoint_tar)
        start_epoch = checkpoint['epoch']
        best_top1_acc = checkpoint['best_top1_acc']
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint {} epoch = {}" .format(checkpoint_tar, checkpoint['epoch']))
    for epoch in range(start_epoch):
        scheduler.step()

    # Data loading code
    traindir = os.path.join('D:/MGongsj/0929\MetaPruning-master/resnet/training/data/tiny-imagenet-200/train')
    valdir = os.path.join('D:/MGongsj/0929\MetaPruning-master/resnet/training/data/tiny-imagenet-200/val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    crop_scale = 0.08
    lighting_param = 0.1
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
        Lighting(lighting_param),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transforms)
    trainset = datasets.ImageFolder(
        traindir,
        transform=train_transforms)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    valset = datasets.ImageFolder(
        valdir,
        transform=train_transforms)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=train_transforms)
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    epoch = start_epoch
    while epoch < args.epochs:
      train_obj, train_top1_acc,  train_top5_acc, epoch = train(epoch,  train_loader, model, criterion_smooth, optimizer, scheduler)
      valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model, criterion, args)

      is_best = False
      if valid_obj > best_top1_acc:
        best_top1_acc = valid_obj,
        is_best = True

      save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'best_top1_acc': best_top1_acc,
        'optimizer' : optimizer.state_dict(),
        }, is_best, args.save)

      epoch += 1

    training_time = (time.time() - start_t) / 36000
    print('total training time = {} hours'.format(training_time))


def train(epoch, train_loader, model, criterion, optimizer, scheduler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    scheduler.step()

    for i, (images, target) in enumerate(train_loader):
      data_time.update(time.time() - end)
      images = images.cuda()
      target = target.cuda()
      # compute output
      logits = model(images)#, overall_scale_ids, mid_scale_ids
      loss = criterion(logits, target)
      # measure accuracy and record loss
      prec1, prec5 = accuracy(logits, target, topk=(1,5))
      n = images.size(0)
      losses.update(loss.item(), n)   #accumulated loss
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      # compute gradient and do SGD step
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % args.print_freq == 0:
          progress.display(i)

    return losses.avg, top1.avg, top5.avg, epoch


def validate(epoch, val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')


    model.eval()


    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
  main()
