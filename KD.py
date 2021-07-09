#知识蒸馏
from resnet32_10 import resnet_32
from resnet32_cut_10 import resnet_32_cut
import pickle
from utils.utils import *
from torchvision import datasets, transforms
import torchvision
import argparse

parser = argparse.ArgumentParser("ResNet50")
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
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

CLASSES = 10

def main():
    if not torch.cuda.is_available():
        sys.exit(1)
    start_t = time.time()
    cudnn.benchmark = True
    cudnn.enabled=True

    filename = './searching_snapshot_res32_10_04.pkl'
    if os.path.exists(filename):
        data = pickle.load(open(filename, 'rb'))
        candidates = data['candidates']
        keep_top_k = data['keep_top_k']
        keep_top_50 = data['keep_top_50']
        start_iter = data['iter'] + 1
    model_1 = resnet_32()
    model_1 = nn.DataParallel(model_1.cuda())

    model_2 = resnet_32_cut(keep_top_k[0],[3,3,6,3])
    model_2 = nn.DataParallel(model_2.cuda())


    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()


    optimizer = torch.optim.SGD(
        # [{'params' : other_parameters},
        # {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
        model_2.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        )

    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/args.epochs), last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs//4, args.epochs//2, args.epochs//4*3], gamma=0.1)
    start_epoch = 0
    best_top1_acc= 0
    checkpoint_tar = 'D:/MGongsj/0929/MetaPruning-master/resnet/searching/models/resnet32_cifar10/model_best.pth.tar' #os.path.join(args.save, 'checkpoint.pth.tar')
    if os.path.exists(checkpoint_tar):
        # logging.info('loading checkpoint {} ..........'.format(checkpoint_tar))
        checkpoint = torch.load(checkpoint_tar)
        # start_epoch = checkpoint['epoch']
        # best_top1_acc = checkpoint['best_top1_acc']
        model_1.load_state_dict(checkpoint['state_dict'])
        # logging.info("loaded checkpoint {} epoch = {}" .format(checkpoint_tar, checkpoint['epoch']))
    for epoch in range(start_epoch):
        scheduler.step()

    # Data loading code

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


    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=train_transforms)


    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    epoch = start_epoch
    while epoch < args.epochs:
      train_obj, train_top1_acc,  train_top5_acc, epoch = train(epoch,  train_loader, model_1, model_2, criterion_smooth, optimizer, scheduler)
      valid_obj, valid_top1_acc, valid_top5_acc = validate(val_loader, model_2, criterion, args)

      is_best = False
      if valid_top1_acc > best_top1_acc:
        best_top1_acc = valid_top1_acc
        is_best = True

      save_checkpoint({
        'epoch': epoch,
        'state_dict': model_2.state_dict(),
        'best_top1_acc': best_top1_acc,
        'optimizer' : optimizer.state_dict(),
        }, is_best, args.save)

      epoch += 1

    training_time = (time.time() - start_t) / 36000
    print('total training time = {} hours'.format(training_time))


def train(epoch, train_loader, model_1, model_2, criterion, optimizer, scheduler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model_2.train()
    end = time.time()
    scheduler.step()

    for i, (images, target) in enumerate(train_loader):

      data_time.update(time.time() - end)
      images = images.cuda()
      target = target.cuda()
      # compute output
      logits, a1, a2, a3, a4 = model_1(images)
      logits_, a_1, a_2, a_3, a_4 = model_2(images)

      pi = 0.5
      loss0 = criterion(logits_, target)
      loss1 = nn.functional.mse_loss(logits,logits_)#nn.functional.nll_loss(logits_, target)  # 学生网络的标签
      # loss1 = nn.functional.mse_loss(a1, a_1) * pi
      # loss2 = nn.functional.mse_loss(a2, a_2) * (pi ** 2)
      # loss3 = nn.functional.mse_loss(a3, a_3) * (pi ** 3)
      # loss4 = nn.functional.mse_loss(a4, a_4) * (pi ** 4)
      loss = 0.7*loss0 + 0.3*loss1 #+ loss2 + loss3 + loss4  # 学生网络使用标签
      # loss = loss1 + loss2 + loss3 + loss4  # 学生网络不使用标签
      # logits = model(images)

      # loss = criterion(logits, target)
      # measure accuracy and record loss
      prec1, prec5 = accuracy(logits_, target, topk=(1,5))
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

def validate(val_loader, model, criterion, args):
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
            # loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            # losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg
if __name__ == '__main__':
    main()
# parser = argparse.ArgumentParser("ResNet50")
# parser.add_argument('--max_iters', type=int, default=30)
# parser.add_argument('--net_cache', type=str, default='../training/models/checkpoint.pth.tar', help='model to be loaded')
# parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
# parser.add_argument('--batch_size', type=int, default=10, help='batch size')
# parser.add_argument('--save_dict_name',type=str, default='save_dict_3.txt')
# parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
#                     help='number of data loading workers (default: 4)')
# args = parser.parse_args()
#
#
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#
# # data augmentation
# crop_scale = 0.08
# lighting_param = 0.1
# train_transforms = transforms.Compose([
#     transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
#     Lighting(lighting_param),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     normalize])
#
# trainset = torchvision.datasets.CIFAR100(root='./data', train=False,
#                                             download=True, transform=train_transforms)
#
#
# train_loader = torch.utils.data.DataLoader(
#     trainset, batch_size=2, shuffle=True,
#     num_workers=args.workers, pin_memory=True)
#
#
# filename = './searching_snapshot.pkl'
# if os.path.exists(filename):
#     data = pickle.load(open(filename, 'rb'))
#     candidates = data['candidates']
#     keep_top_k = data['keep_top_k']
#     keep_top_50 = data['keep_top_50']
#     start_iter = data['iter'] + 1
#
#
#
# # print(len(keep_top_k[0]))
#
#
# model_1 = resnet_50()
# model_1 = nn.DataParallel(model_1.cuda())
#
# model_2 = resnet_50_cut(keep_top_k[0],[3,4,6,3])
# model_2 = nn.DataParallel(model_2.cuda())
#
# if os.path.exists(args.net_cache):
#     print('loading checkpoint {} ..........'.format(args.net_cache))  # 加载模型参数
#     checkpoint = torch.load(args.net_cache)  # 加载模型参数
#     best_top1_acc = checkpoint['best_top1_acc']
#     model_1.load_state_dict(checkpoint['state_dict'])  # 模型加载权值
#     # para=list(model.parameters())
#     # logging.info(para)
#     print("loaded checkpoint {} epoch = {}".format(args.net_cache, checkpoint['epoch']))
#
# batch_time = AverageMeter('Time', ':6.3f')
# data_time = AverageMeter('Data', ':6.3f')
# losses = AverageMeter('Loss', ':.4e')
# top1 = AverageMeter('Acc@1', ':6.2f')
# top5 = AverageMeter('Acc@5', ':6.2f')
#
# progress = ProgressMeter(
#     len(train_loader),
#     [batch_time, data_time, losses, top1, top5],
#     prefix="Epoch: [{}]".format(epoch))
#
# optimizer = torch.optim.SGD(
#         # [{'params' : other_parameters},
#         # {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
#         model_2.parameters(),
#         args.learning_rate,
#         momentum=args.momentum,
#         )
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs//4, args.epochs//2, args.epochs//4*3], gamma=0.1)
# model_2.train()
# end = time.time()
# scheduler.step()
#
# for i, (images, target) in enumerate(train_loader):
#
#     images = images.cuda()
#     target = target.cuda()
#     logits, a1, a2, a3, a4 = model_1(images)
#     logits_, a_1, a_2, a_3, a_4 = model_2(images)
#
#     pi = 0.5
#     loss0 = nn.functional.nll_loss(logits_, target)  # 学生网络的标签
#     loss1 = nn.functional.mse_loss(a1, a_1) * pi
#     loss2 = nn.functional.mse_loss(a2, a_2) * (pi ** 2)
#     loss3 = nn.functional.mse_loss(a3, a_3) * (pi ** 3)
#     loss4 = nn.functional.mse_loss(a4, a_4) * (pi ** 4)
#     loss = loss0 + loss1 + loss2 + loss3 + loss4  # 学生网络使用标签
#     # loss = loss1 + loss2 + loss3 + loss4  # 学生网络不使用标签
