import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5,6'
from sklearn.model_selection import KFold
import traceback, sys
import numpy as np
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader
from stage_dataset import STAGE_sub3_dataset
from cnn_model import *
from tqdm import tqdm
import datetime
import transforms as trans

import warnings
warnings.filterwarnings('ignore')

kfold = 5
random_state = 42 # DataSplit Random Seed
num_workers = 8

oct_img_size = [512, 512]
fundus_img_size = [512, 512]
backbone = 'resnext50'
if backbone == 'resnet34':
    batchsize = 12 * torch.cuda.device_count()
elif backbone == 'resnet50' or backbone == 'resnext50':
    batchsize = 8 * torch.cuda.device_count()
else:
    batchsize = 8 * torch.cuda.device_count()
init_lr = 1e-2
optimizer_type = "sgd"
scheduler = None
iters = 100 # Run Epochs

root = "/media/user/"
trainset_root = root + "MICCAI2024-STAGE2/STAGE_training/training_images"
model_root = "./model_checkpoints"
logs_root = "./logs"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.log = open(filename, 'w')
        self.hook = sys.excepthook
        sys.excepthook = self.kill

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def kill(self, ttype, tvalue, ttraceback):
        for trace in traceback.format_exception(ttype, tvalue, ttraceback):
            print(trace)
        os.remove(self.filename)

    def flush(self):
        pass


def train(model, iters, train_dataloader, val_dataloader, optimizer, criterion, log_interval, eval_interval, fold):
    model_save_path = os.path.join(model_root, time_now, str(fold))
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    print(time_now, "Total Nums: {}, train: {}, val: {}".format(len(filelists), len(train_filelists), len(val_filelists)))

    scheduler = EpochLR(optimizer, epochs=iters, gamma=0.9)
    model.train()
    avg_loss3_list = []
    avg_ma_f1_list = []
    avg_mi_f1_list = []
    avg_score3_list = []
    best_score = 0
    best_epoch = 0
    for iter in tqdm(range(1, iters+1)):
        for batch, data in enumerate(train_dataloader):
            oct_imgs = (data[0] / 255.).to(dtype=torch.float32).to(device)
            fundus_imgs = (data[1] / 255.).to(dtype=torch.float32).to(device)
            label_task3 = (data[2]).to(dtype=torch.uint8).to(device)
            info = data[3].to(dtype=torch.float32).to(device)

            task3_logits = model(oct_imgs, fundus_imgs, info, deep_sup=True)
            total_loss, task3_likehood = None, None
            for i in range(len(task3_logits)):
                loss, likehood = criterion(task3_logits[i].reshape(-1, 1), label_task3.reshape(-1, 1))
                if task3_likehood is None:
                    task3_likehood = likehood
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss += loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            ma_f1 = f1_score(task3_likehood.detach().cpu().numpy().argmax(1), label_task3.view(-1).cpu().numpy(), labels=[0, 1, 2, 3, 4], average='macro')
            mi_f1 = f1_score(task3_likehood.detach().cpu().numpy().argmax(1), label_task3.view(-1).cpu().numpy(), labels=[0, 1, 2, 3, 4], average='micro')

            avg_loss3_list.append(total_loss.item())
            avg_ma_f1_list.append(ma_f1)
            avg_mi_f1_list.append(mi_f1)
            avg_score3_list.append(ma_f1 * 10 * 0.5 + mi_f1 * 10 * 0.5)

        if scheduler is not None:
            scheduler.step()
            
        if iter % log_interval == 0:
            avg_loss3 = np.array(avg_loss3_list).mean()
            avg_ma_f1 = np.array(avg_ma_f1_list).mean()
            avg_mi_f1 = np.array(avg_mi_f1_list).mean()
            avg_score3 = np.array(avg_score3_list).mean()
            avg_loss3_list = []
            avg_ma_f1_list = []
            avg_mi_f1_list = []
            avg_score3_list = []
            print("[TRAIN] iter={}/{} avg_loss={:.4f} avg_score={:.4f} avg_ma_f1={:.4f} avg_mi_f1={:.4f}".format(
                iter, iters, avg_loss3, avg_score3, avg_ma_f1, avg_mi_f1))

        if iter % eval_interval == 0:
            avg_loss3, avg_ma_f1, avg_mi_f1, avg_score3 = val(model, val_dataloader, criterion)
            print("[EVAL] iter={}/{} avg_loss={:.4f} avg_score={:.4f}avg_ma_f1={:.4f} avg_mi_f1={:.4f} ".format(
                iter, iters, avg_loss3, avg_score3, avg_ma_f1, avg_mi_f1))
            if avg_score3 >= best_score:
                best_score = avg_score3
                best_epoch = iter
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), os.path.join(model_save_path, 'best_model.pth'))
                else:
                    torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))
            model.train()

    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), os.path.join(model_save_path, "last_model.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(model_save_path, "last_model.pth"))
    print('Best Epoch:{} Best Score:{:.4f}'.format(best_epoch, best_score))


def val(model, val_dataloader, criterion):
    model.eval()
    avg_loss3_list = []
    avg_ma_f1_list = []
    avg_mi_f1_list = []
    avg_score3_list = []
    with torch.no_grad():
        for batch, data in enumerate(val_dataloader):
            oct_imgs = (data[0] / 255.).to(dtype=torch.float32).to(device)
            fundus_imgs = (data[1] / 255.).to(dtype=torch.float32).to(device)
            label_task3 = (data[2]).to(dtype=torch.uint8).to(device)
            info = data[3].to(dtype=torch.float32).to(device)

            task3_logits = model(oct_imgs, fundus_imgs, info)

            loss, task3_likehood = criterion(task3_logits.reshape(-1, 1), label_task3.reshape(-1, 1))

            ma_f1 = f1_score(task3_likehood.detach().cpu().numpy().argmax(1), label_task3.view(-1).cpu().numpy(), labels=[0, 1, 2, 3, 4], average='macro')
            mi_f1 = f1_score(task3_likehood.detach().cpu().numpy().argmax(1), label_task3.view(-1).cpu().numpy(), labels=[0, 1, 2, 3, 4], average='micro')

            avg_loss3_list.append(loss.item())
            avg_ma_f1_list.append(ma_f1)
            avg_mi_f1_list.append(mi_f1)
            avg_score3_list.append(ma_f1 * 10 * 0.5 + mi_f1 * 10 * 0.5)

    avg_loss3 = np.array(avg_loss3_list).mean()
    avg_ma_f1 = np.array(avg_ma_f1_list).mean()
    avg_mi_f1 = np.array(avg_mi_f1_list).mean()
    avg_score3 = np.array(avg_score3_list).mean()
    return avg_loss3, avg_ma_f1, avg_mi_f1, avg_score3


def main(train_filelists, val_filelists, fold):
    oct_train_transforms = trans.Compose([
        trans.CenterRandomCrop([256] + oct_img_size),
        trans.RandomHorizontalFlip(),
        trans.GaussianBlurTransform(),
        trans.GaussianNoiseTransform()
    ])

    oct_val_transforms = trans.Compose([
        trans.CenterCrop([256] + oct_img_size)
    ])

    fundus_train_transforms = trans.Compose([
        trans.Resize(fundus_img_size),
        trans.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1, hue=0.1),
        trans.RandomHorizontalFlip(),
        trans.RandomVerticalFlip(),
        trans.GaussianBlurTransform(p=0.25, blur_sigma=(0.25, 0.5))
    ])

    fundus_val_transforms = trans.Compose([
        trans.Resize(fundus_img_size)
    ])

    train_dataset = STAGE_sub3_dataset(dataset_root=trainset_root,
                            oct_transforms=oct_train_transforms,
                            fundus_transforms=fundus_train_transforms,
                            filelists=train_filelists,
                            label_file=root + 'MICCAI2024-STAGE2/STAGE_training/training_GT/task3_GT_training.csv',
                            info_file=root + 'MICCAI2024-STAGE2/STAGE_training/data_info_training.xlsx')

    val_dataset = STAGE_sub3_dataset(dataset_root=trainset_root,
                            oct_transforms=oct_val_transforms,
                            fundus_transforms=fundus_val_transforms,
                            filelists=val_filelists,
                            label_file=root + 'MICCAI2024-STAGE2/STAGE_training/training_GT/task3_GT_training.csv',
                            info_file=root + 'MICCAI2024-STAGE2/STAGE_training/data_info_training.xlsx')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batchsize,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batchsize,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )

    model = Model(type=backbone).to(device)
    criterion = OrdinalRegressionLoss(num_class=5)

    if torch.cuda.device_count() > 1:
        device_ids = list(range(0, torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.99))
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.99, weight_decay=0.005, nesterov=True)

    train(model, iters, train_loader, val_loader, optimizer, criterion, log_interval=1, eval_interval=5, fold=fold)


if not os.path.exists(logs_root):
    os.makedirs(logs_root)
log_path = os.path.join(logs_root, time_now + '.log')
sys.stdout = Logger(log_path, sys.stdout)

kf = KFold(n_splits=kfold, shuffle=True, random_state=random_state)
filelists = os.listdir(trainset_root)
for fold, (train_index, val_index) in enumerate(kf.split(filelists)):
    print(f"Fold {fold + 1}")
    train_filelists = [filelists[i] for i in train_index]
    val_filelists = [filelists[i] for i in val_index]
    main(train_filelists, val_filelists, fold)

