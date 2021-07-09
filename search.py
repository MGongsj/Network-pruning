import sys
import pickle
import torch.utils
import torch.utils.data.distributed
sys.path.append("../../")
from utils.utils import *
from resnet50 import resnet_50
from cut_test import resnet_50_cut



import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms

import pywt
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from PIL import Image
from numpy import average, dot, linalg

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join('log/log51.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# 对于低频分量，计算两图的权重比
def varianceWeight(img1, img2):
    mean1, var1 = cv2.meanStdDev(img1)
    mean2, var2 = cv2.meanStdDev(img2)
    weight1 = var1 / (var1 + var2)
    weight2 = var2 / (var1 + var2)
    return weight1, weight2

def varianceWeight_1(img):
    var = []
    weight = []
    n = 0
    for i in range(0,len(img)):
        # print(img[i])
        mean1, var1 = cv2.meanStdDev(img[i][0])
        n = n+var1
        var.append(var1)
    for j in range(0,len(var)):
        weight1 = var[j]/n
        weight.append(weight1)

    return weight


# 实测这个函数效果非常好！！！
def getVarianceImg(array):
    row, col = array.shape
    varImg = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            up = i - 5 if i - 5 > 0 else 0
            down = i + 5 if i + 5 < row else row
            left = j - 5 if j - 5 > 0 else 0
            right = j + 5 if j + 5 < col else col
            window = array[up:down, left:right]
            mean, var = cv2.meanStdDev(window)
            varImg[i, j] = var
    return varImg


# 不会写canny，暂时先用Sobel算子代替
def calcGradient(img):
    xDiff = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    yDiff = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    stdXdiff = cv2.convertScaleAbs(xDiff)
    stdYdiff = cv2.convertScaleAbs(yDiff)
    gradient = np.sqrt(stdXdiff ** 2 + stdYdiff ** 2)
    return gradient


def testWave1(a1):
    transf = []
    for p in range(0,len(a1)):
        a = a1[p].cpu().detach().numpy()
        transf1 = pywt.wavedec2(a, 'haar', level=4)
        transf.append(transf1)#把小波变换后的图片存在列表里

    # assert len(transf1) == len(transf2)
    recWave = []
    for k in range(len(transf[0])):
        # 处理低频分量
        if k == 0:
            loWeight = varianceWeight_1(transf)
            lowFreq = np.zeros(transf[0][0].shape)
            row, col = transf[0][0].shape
            for i in range(row):
                for j in range(col):
                    for m in range(0,len(loWeight)):
                        lowFreq[i, j] = lowFreq[i, j] + loWeight[m] * transf[m][0][i, j]
            recWave.append(lowFreq)
            continue
        # 处理高频分量
        cvtArray = []
        cvtArray_ = []
        array = []


        for s in range(0,int(len(transf)/2)):
            for array1, array2 in zip(transf[s*2][k], transf[s*2+1][k]):
                array.append(array1)
                array.append(array2)
        for array1, array2 in zip(transf[0][k], transf[1][k]):

            tmp_row, tmp_col = array1.shape
            highFreq = np.zeros((tmp_row, tmp_col))
            var = []
            for h in range(0,len(array)):
                var1 = getVarianceImg(array[h])
                var.append(var1)
            for i in range(tmp_row):
                for j in range(tmp_col):
                    s = 0
                    max = var[0][i, j]
                    for h in range(1,len(var)):
                        if var[h][i,j] > max:
                            max = var[h][i,j]
                            s = h
                    highFreq[i, j] = array[s][i, j]
            # cvtArray_.append(highFreq)
            cvtArray_.append(highFreq)

        recWave.append(tuple(cvtArray_))
    return pywt.waverec2(recWave, 'haar')

def testWave2(a1):
    transf = []
    for p in range(0, len(a1)):
        a = a1[p].cpu().detach().numpy()
        transf1 = pywt.wavedec2(a, 'haar', level=3)
        transf.append(transf1)  # 把小波变换后的图片存在列表里

    # assert len(transf1) == len(transf2)
    recWave = []
    for k in range(len(transf[0])):
        # 处理低频分量
        if k == 0:
            loWeight = varianceWeight_1(transf)
            lowFreq = np.zeros(transf[0][0].shape)
            row, col = transf[0][0].shape
            for i in range(row):
                for j in range(col):
                    for m in range(0, len(loWeight)):
                        lowFreq[i, j] = lowFreq[i, j] + loWeight[m] * transf[m][0][i, j]
            recWave.append(lowFreq)
            continue
        # 处理高频分量
        cvtArray = []
        cvtArray_ = []
        array = []

        for s in range(0, int(len(transf) / 2)):
            for array1, array2 in zip(transf[s * 2][k], transf[s * 2 + 1][k]):
                array.append(array1)
                array.append(array2)
        for array1, array2 in zip(transf[0][k], transf[1][k]):

            tmp_row, tmp_col = array1.shape
            highFreq = np.zeros((tmp_row, tmp_col))
            var = []
            for h in range(0, len(array)):
                var1 = getVarianceImg(array[h])
                var.append(var1)
            for i in range(tmp_row):
                for j in range(tmp_col):
                    s = 0
                    max = var[0][i, j]
                    for h in range(1, len(var)):
                        if var[h][i, j] > max:
                            max = var[h][i, j]
                            s = h
                    highFreq[i, j] = array[s][i, j]
            # cvtArray_.append(highFreq)
            cvtArray_.append(highFreq)

        recWave.append(tuple(cvtArray_))
    return pywt.waverec2(recWave, 'haar')

def testWave3(a1):
    transf = []
    for p in range(0, len(a1)):
        a = a1[p].cpu().detach().numpy()
        transf1 = pywt.wavedec2(a, 'haar', level=2)
        transf.append(transf1)  # 把小波变换后的图片存在列表里

    # assert len(transf1) == len(transf2)
    recWave = []
    for k in range(len(transf[0])):
        # 处理低频分量
        if k == 0:
            loWeight = varianceWeight_1(transf)
            lowFreq = np.zeros(transf[0][0].shape)
            row, col = transf[0][0].shape
            for i in range(row):
                for j in range(col):
                    for m in range(0, len(loWeight)):
                        lowFreq[i, j] = lowFreq[i, j] + loWeight[m] * transf[m][0][i, j]
            recWave.append(lowFreq)
            continue
        # 处理高频分量
        cvtArray = []
        cvtArray_ = []
        array = []

        for s in range(0, int(len(transf) / 2)):
            for array1, array2 in zip(transf[s * 2][k], transf[s * 2 + 1][k]):
                array.append(array1)
                array.append(array2)
        for array1, array2 in zip(transf[0][k], transf[1][k]):

            tmp_row, tmp_col = array1.shape
            highFreq = np.zeros((tmp_row, tmp_col))
            var = []
            for h in range(0, len(array)):
                var1 = getVarianceImg(array[h])
                var.append(var1)
            for i in range(tmp_row):
                for j in range(tmp_col):
                    s = 0
                    max = var[0][i, j]
                    for h in range(1, len(var)):
                        if var[h][i, j] > max:
                            max = var[h][i, j]
                            s = h
                    highFreq[i, j] = array[s][i, j]
            # cvtArray_.append(highFreq)
            cvtArray_.append(highFreq)

        recWave.append(tuple(cvtArray_))
    return pywt.waverec2(recWave, 'haar')

def get_thum(image, size=(64, 64), greyscale=False):
    # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        # 将图片转换为L模式，其为灰度图，其每个像素用8个bit表示
        image = image.convert('L')
    return image


# 计算图片的余弦距离
def image_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        # linalg=linear（线性）+algebra（代数），norm则表示范数
        # 求图片的范数？？
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = dot(a / a_norm, b / b_norm)
    return res


sys.setrecursionlimit(10000)

if not os.path.exists('log'):
    os.mkdir('log')

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join('log/log03.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


parser = argparse.ArgumentParser("ResNet50")
parser.add_argument('--max_iters', type=int, default=30)
parser.add_argument('--net_cache', type=str, default='../training/models/checkpoint.pth.tar', help='model to be loaded')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=10, help='batch size')
parser.add_argument('--save_dict_name',type=str, default='save_dict_3.txt')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
args = parser.parse_args()

stage_repeat = [3, 4, 6, 3] #模型中maker_layer里的block的个数

max_FLOPs = 2050
min_FLOPs = 1950

# file for save the intermediate searched results
save_dict = {}#全局变量
if os.path.exists(args.save_dict_name):
    f = open(args.save_dict_name, 'rb')
    save_dict = pickle.load(f)
    f.close()
    print(save_dict, flush=True)

# load training data
traindir = os.path.join(args.data)
valdir = os.path.join(args.data)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# data augmentation
crop_scale = 0.08
lighting_param = 0.1
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
    Lighting(lighting_param),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize])

trainset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=train_transforms)

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=2, shuffle=True,
    num_workers=args.workers, pin_memory=True)



# infer the accuracy of a selected pruned net (identidyed with ids)
def infer(model, criterion, layer, n):#评估剪枝的网络就是把它训练
    model.eval()

    model_2 = resnet_50_cut(criterion,layer)
    model_2 = nn.DataParallel(model_2.cuda())
    model_2.eval()

    for i, (images, target) in enumerate(train_loader):
        if i == 1:
            break
        logits,a1,a2,a3,a4 = model(images)
        logits_,a_1, a_2, a_3, a_4 = model_2(images)

        sil_1 = 0
        sil_2 = 0
        sil_3 = 0
        sil_4 = 0

        for j in range(0,len(a1)):#32...
            # a = a1[0].cpu().detach().numpy()

            img1 = Image.fromarray(testWave1(a1[j]))
            img_1 = Image.fromarray(testWave1(a_1[j]))
            img2 = Image.fromarray(testWave1(a2[j]))
            img_2 = Image.fromarray(testWave1(a_2[j]))
            img3 = Image.fromarray(testWave2(a3[j]))
            img_3 = Image.fromarray(testWave2(a_3[j]))
            img4 = Image.fromarray(testWave3(a4[j]))
            img_4 = Image.fromarray(testWave3(a_4[j]))
            sil1 = image_similarity_vectors_via_numpy(img1, img_1)
            sil_1 = sil_1+sil1
            sil2 = image_similarity_vectors_via_numpy(img2, img_2)
            sil_2 = sil_2 + sil2
            sil3 = image_similarity_vectors_via_numpy(img3, img_3)
            sil_3 = sil_3 + sil3
            sil4 = image_similarity_vectors_via_numpy(img4, img_4)
            sil_4 = sil_4 + sil4
        if n == 0:
            return sil_1/2
        elif n == 1:
            return sil_2/2
        elif n == 2:
            return sil_3/2
        elif n == 3:
            return sil_4/2

    #     sil = sil + (sil1+sil2+sil3+sil4)/4
    # return(sil/10)


# prepare ids for testing
def test_candidates_model(model, candidates, cnt, layer, n):#n:是第几次迭代
    candidate = []
    for can in candidates:
        print('test {}th model'.format(cnt), flush=True)
        # print(list(can[:-1].astype(np.int32)))
        # print(list(can))
        # print('FLOPs = {:.2f}M'.format(can[-1]), flush=True)
        t_can = tuple(can[:-1])
        similar = infer(model, can[:-1], layer, n)#就是整个can除了最后一个元素的意思，规定了网络结构的各参数，这里应该指的是各层通道数
        print('similar = {:.2f}'.format(similar), flush=True)
        save_dict[t_can] = similar
        cnt += 1
        can[-1] = similar
        candidate.append(can)
    return candidate, cnt  #感觉cnt没有用，就是连续计数的

# mutation operation in evolution algorithm
def get_mutation(keep_top_k, mutation_num, length, n):
    print('mutation ......', flush=True)
    res = []
    k = len(keep_top_k)#0
    iter = 0
    max_iters = 10
    while len(res)<mutation_num and iter<max_iters:
        ids = np.random.choice(k, mutation_num)
        select_seed = np.array([keep_top_k[id] for id in ids])#(0,22721)
        is_m_ = np.zeros((mutation_num, 22721))
        is_m = np.random.choice(np.arange(-1, 1), (mutation_num, length), p=[0.1, 0.9])
        if n == 0:
            is_m_[:,0:1216] = is_m
            select_list = (abs(select_seed + is_m_))
            print(select_list.shape)
            iter += 1
        if n == 1:
            is_m_[:, 1216:4288] = is_m
            select_list = (abs(select_seed + is_m_))
            iter += 1
        if n == 2:
            is_m_[:, 4288:13504] = is_m
            select_list = (abs(select_seed + is_m_))
            iter += 1
        if n == 3:
            is_m_[:, 13504:22720] = is_m
            select_list = (abs(select_seed + is_m_))
            iter += 1
        for can in select_list:
            res.append(can)
            if len(res)==mutation_num:
                break

    print('mutation_num = {}'.format(len(res)), flush=True)
    return res

# crossover operation in evolution algorithm
def get_crossover(keep_top_k, crossover_num):#交叉
    print('crossover ......', flush=True)
    res = []
    k = len(keep_top_k)
    iter = 0
    max_iters = 10 * crossover_num
    while len(res)<crossover_num and iter<max_iters:
        id1, id2 = np.random.choice(k, 2, replace=False)#从种群里随机选取两个个体（不重复）
        p1 = keep_top_k[id1]
        p2 = keep_top_k[id2]
        mask = np.random.randint(low=0, high=2, size=(22721)).astype(np.float32)
        can = p1*mask + p2*(1.0-mask)
        iter += 1
        res.append(can)
        if len(res)==crossover_num:
            break
    print('crossover_num = {}'.format(len(res)), flush=True)
    return res

# random operation in evolution algorithm
def random_can(num, length):
    print('random select ........', flush=True)
    candidates = []
    while(len(candidates))<num:#20
        can = np.random.randint(0, 2, (length+1)).astype(np.float32)
        can[-1] = 0.5
        candidates.append(can)
    print('random_num = {}'.format(len(candidates)), flush=True)
    return candidates#返回的是一个多维数组，我的理解：输出50个候选人，也就是50个可行解组成第一个种群，然后去评估，再进行遗传算法的优化

# select topk
def select(candidates, keep_top_k, select_num):
    print('select ......', flush=True)
    keep_top_k.extend(candidates)
    keep_top_k = sorted(keep_top_k, key=lambda can:can[-1], reverse=True)#排序，can：Top1_err，按照错误率，默认为升序
    return keep_top_k[:select_num]

def search(model):

    cnt = 1
    select_num = 50
    population_num = 50
    mutation_num = 10
    crossover_num = 10
    random_num = population_num - mutation_num - crossover_num


    keep_top_k = []
    keep_top_50 = []
    keep_top = []
    print('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_iters = {}'.format(population_num, select_num, mutation_num, crossover_num, random_num, args.max_iters))

    # first 20 candidates are generated randomly

    filename = './searching_snapshot_res50_imagenet.pkl'
    if os.path.exists(filename):
        data = pickle.load(open(filename, 'rb'))
        candidates = data['candidates']
        keep_top_k = data['keep_top_k']
        keep_top_50 = data['keep_top_50']
        start_iter = data['iter'] + 1
    candidates = random_can(population_num, 22720)  # 50,4+3+6+4+3=20，test_dict = {}，untest_dict = {}
    for j in range(0,4):
        if j==0:
            layer = [3, 0, 0, 0]
            max_iters = 10
            length = 1216
        elif j == 1:
            layer = [3, 4, 0, 0]
            max_iters = 10
            length = 3072
        elif j == 2:
            layer = [3, 4, 6, 0]
            max_iters = 5
            length = 9216
        elif j == 3:
            layer = [3, 4, 6, 3]
            max_iters = 10
            length = 9216

        start_iter = 0
        for iter in range(start_iter, max_iters):#0,5

            candidates, cnt = test_candidates_model(model, candidates, cnt, layer, j)#resnet50,loss,候选人（那50个）,1,test_dict={}

            keep_top_50 = select(candidates, keep_top_50, select_num)#给50个个体排序
            keep_top_k = keep_top_50[0:20]#抽出50个个体的前20个
            if iter == max_iters-1:
                keep_top = keep_top_50
                keep_top_50 = []
            print('iter = {} : top {} result'.format(iter, select_num), flush=True)
            for i in range(20):
                res = keep_top_k[i]
                print('No.{} {} similar = {}'.format(i+1, res[:-1], res[-1]))

            mutation = get_mutation(keep_top_k, mutation_num, length, j) #突变
            crossover = get_crossover(keep_top_k, crossover_num) #交叉
            random_num = population_num - len(mutation) -len(crossover)#需要重新生成几个随机数组
            rand = random_can(random_num, 22720)

            candidates = []#重新定义种群
            candidates.extend(mutation)
            candidates.extend(crossover)
            candidates.extend(rand)


            if iter == max_iters-1:
                candidates = np.array(candidates)
                keep_top = np.array(keep_top)
                if j == 0:
                    candidates[20:50, 0:1216] = keep_top[0:30, 0:1216]
                if j == 1:
                    candidates[20:50:, 0:4288] = keep_top[0:30, 0:4288]
                if j == 2:
                    candidates[20:50:, 0:13504] = keep_top[0:30, 0:13504]
            if j == 1 and iter != max_iters-1:
                candidates = np.array(candidates)
                keep_top = np.array(keep_top)
                candidates[20:50,0:1216] = keep_top[0:30,0:1216]
            if j == 2 and iter != max_iters-1:
                candidates = np.array(candidates)
                keep_top = np.array(keep_top)
                candidates[20:50:,0:4288] = keep_top[0:30,0:4288]
            if j == 3 and iter != max_iters-1:
                candidates = np.array(candidates)
                keep_top = np.array(keep_top)
                candidates[20:50:,0:13504] = keep_top[0:30,0:13504]

    print('saving tested_dict ........', flush=True)
    f = open(args.save_dict_name, 'wb')
    pickle.dump(save_dict, f)
    f.close()
    snap = {'candidates':candidates, 'keep_top_k':keep_top_k, 'keep_top_50':keep_top_50, 'iter':iter}
    pickle.dump(snap, open(filename, 'wb'))

    logging.info(keep_top_k)
    print('finish!')

def run():
    t = time.time()


    model_1 = resnet_50()
    model_1 = nn.DataParallel(model_1.cuda())

    if os.path.exists('D:/MGongsj/0929/MetaPruning-master/resnet/searching/models/resnet50_imagenet/model_best.pth.tar'):
        # print('loading checkpoint {} ..........'.format(args.net_cache))#加载模型参数
        checkpoint = torch.load('D:/MGongsj/0929/MetaPruning-master/resnet/searching/models/resnet50_imagenet/model_best.pth.tar')#加载模型参数
        best_top1_acc = checkpoint['best_top1_acc']
        model_1.load_state_dict(checkpoint['state_dict'])#模型加载权值

    else:
        print('can not find {} ')
        return

    # num_states = len(stage_repeat) + sum(stage_repeat)
    search(model_1)

    total_searching_time = time.time() - t
    print('total searching time = {:.2f} hours'.format(total_searching_time/3600), flush=True)


if __name__ == '__main__':
    run()


