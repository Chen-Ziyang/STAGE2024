import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import numpy as np
from sklearn.model_selection import KFold
import datetime
from sklearn.metrics import r2_score
import torch, sys, traceback
from torch.utils.data import DataLoader
from stage_dataset import STAGE_sub1_dataset
from cnn_model import *
from tqdm import tqdm
import transforms as trans

import warnings
warnings.filterwarnings('ignore')

torch.set_num_threads(1)

kfold = 5
random_state = 42 # DataSplit Random Seed
num_workers = 8

fundus_img_size = [512, 512]
oct_img_size = [512, 512]
image_size = 256
backbone = 'resnet34'
if backbone == 'resnet18':
    batchsize = 24 * torch.cuda.device_count()
elif backbone == 'resnet34':
    batchsize = 16 * torch.cuda.device_count()
elif backbone == 'resnet50':
    batchsize = 8 * torch.cuda.device_count()
else:
    batchsize = 16 * torch.cuda.device_count()
init_lr = 1e-2
optimizer_type = "sgd"
scheduler = None
iters = 50 # Run Epochs

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

    avg_loss_list = []
    avg_score_list = []
    avg_smape_list = []
    avg_r2_list = []
    best_score = 0
    best_epoch = 0
    loss_weights = np.array([1.0, 0.5, 0.5])
    loss_weights = loss_weights / loss_weights.sum()
    for iter in tqdm(range(1, iters+1)):
        for batch, data in enumerate(train_dataloader):
            oct_imgs = (data[0] / 255.).to(dtype=torch.float32).to(device)
            fundus_imgs = (data[1] / 255.).to(dtype=torch.float32).to(device)
            labels = (data[2]).to(dtype=torch.float32).to(device)
            info = (data[3]).to(dtype=torch.float32).to(device)

            logit_outputs = model(oct_imgs, fundus_imgs, info, deep_sup=True)

            loss = None
            for i in range(len(logit_outputs)):
                if loss is None:
                    loss = loss_weights[i] * criterion(labels, logit_outputs[i])
                else:
                    loss += loss_weights[i] * criterion(labels, logit_outputs[i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            smape = Smape_(labels.cpu(), logit_outputs[0].detach().cpu()).numpy()
            r2 = r2_score(labels.cpu(), logit_outputs[0].detach().cpu())

            avg_loss_list.append(loss.item())
            avg_score_list.append(Score(smape, r2))
            avg_smape_list.append(smape)
            avg_r2_list.append(r2)

        if scheduler is not None:
            scheduler.step()
            
        if iter % log_interval == 0:
            avg_loss = np.array(avg_loss_list).mean()
            avg_score = np.array(avg_score_list).mean()
            smape_ = np.array(avg_smape_list).mean()
            r2_ = np.array(avg_r2_list).mean()
            avg_loss_list = []
            avg_score_list = []
            avg_smape_list = []
            avg_r2_list = []
            print("[TRAIN] iter={}/{} avg_loss={:.4f} avg_score={:.4f} SMAPE={:.4f} R2={:.4f}".format(iter, iters, avg_loss, avg_score, smape_, r2_))
            print(model.para.data)

        if iter % eval_interval == 0:
            avg_loss, avg_score, smape_, r2_ = val(model, val_dataloader, criterion)
            print("[EVAL] iter={}/{} avg_loss={:.4f} score={:.4f} SMAPE={:.4f} R2={:.4f}".format(iter, iters, avg_loss, avg_score, smape_, r2_))
            if avg_score >= best_score:
                best_score = avg_score
                best_epoch = iter
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(),
                               os.path.join(model_save_path, 'best_model.pth'))
                else:
                    torch.save(model.state_dict(),
                               os.path.join(model_save_path, 'best_model.pth'))
            model.train()

    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), os.path.join(model_save_path, "last_model.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(model_save_path, "last_model.pth"))
    print('Best Epoch:{} Best Score:{:.4f}'.format(best_epoch, best_score))


def val(model, val_dataloader, criterion):
    model.eval()
    avg_loss_list = []
    avg_score_list = []
    avg_smape_list = []
    avg_r2_list = []
    with torch.no_grad():
        for batch, data in enumerate(val_dataloader):
            oct_imgs = (data[0] / 255.).to(dtype=torch.float32).to(device)
            fundus_imgs = (data[1] / 255.).to(dtype=torch.float32).to(device)
            labels = (data[2]).to(dtype=torch.float32).to(device)
            info = (data[3]).to(dtype=torch.float32).to(device)

            logits = model(oct_imgs, fundus_imgs, info)

            loss = criterion(labels, logits)

            smape = Smape_(labels.cpu(), logits.detach().cpu()).numpy()
            r2 = r2_score(labels.cpu(), logits.detach().cpu())

            avg_loss_list.append(loss.item())
            avg_score_list.append(Score(smape, r2))
            avg_smape_list.append(smape)
            avg_r2_list.append(r2)

    avg_score = np.array(avg_score_list).mean()
    avg_loss = np.array(avg_loss_list).mean()
    smape_ = np.array(avg_smape_list).mean()
    r2_ = np.array(avg_r2_list).mean()
    return avg_loss, avg_score, smape_, r2_


def main(train_filelists, val_filelists, fold):
    oct_train_transforms = trans.Compose([
        trans.CenterRandomCrop([256] + oct_img_size),
        trans.RandomHorizontalFlip(),
        trans.RandomVerticalFlip(),
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

    train_dataset = STAGE_sub1_dataset(dataset_root=trainset_root,
                            oct_transforms=oct_train_transforms,
                            fundus_transforms=fundus_train_transforms,
                            filelists=train_filelists,
                            label_file=root + 'MICCAI2024-STAGE2/STAGE_training/training_GT/task1_GT_training.csv',
                            info_file=root + 'MICCAI2024-STAGE2/STAGE_training/data_info_training.xlsx')

    val_dataset = STAGE_sub1_dataset(dataset_root=trainset_root,
                            oct_transforms=oct_val_transforms,
                            fundus_transforms=fundus_val_transforms,
                            filelists=val_filelists,
                            label_file=root + 'MICCAI2024-STAGE2/STAGE_training/training_GT/task1_GT_training.csv',
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

    model = Model(type=backbone, mixstyle_layer=['layer1', 'layer2']).to(device)
    if torch.cuda.device_count() > 1:
        device_ids = list(range(0, torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.99))
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.99, weight_decay=0.005, nesterov=True)

    criterion = MixLoss()

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
    print(len(train_filelists), len(val_filelists))
    main(train_filelists, val_filelists, fold)
