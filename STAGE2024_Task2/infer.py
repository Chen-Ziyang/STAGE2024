import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from stage_dataset import STAGE_sub2_dataset
from cnn_model import *

import transforms as trans

import warnings
warnings.filterwarnings('ignore')

torch.set_num_threads(1)


def test(loader, model):
    model.eval()
    cache = []
    idxs = []
    with torch.no_grad():
        for batch, (oct_img, fundus_img, idx, info) in enumerate(loader):
            print(idx)
            if len(oct_img.size()) == 5:    # TTA delete the first dimension
                oct_img = oct_img.squeeze(0)
            if len(fundus_img.size()) == 5:    # TTA delete the first dimension
                fundus_img = fundus_img.squeeze(0)
            oct_img = (oct_img / 255.).to(dtype=torch.float32).to(device)
            fundus_img = (fundus_img / 255.).to(dtype=torch.float32).to(device)
            info = info.repeat(oct_img.shape[0], 1).to(dtype=torch.float32).to(device)

            logits = model(oct_img, fundus_img, info) * 40.

            if logits.shape[0] == 1:
                cache.append(list(logits.detach().cpu().numpy()[0]))        # w/o TTA
            else:
                cache.append(list(logits.detach().cpu().numpy().mean(0)))   # w TTA
            idxs.append(idx[0])
    return np.stack(idxs, 0), np.stack(cache, 0)


root = '/media/user/'
save_root = "./results/Sensitivity_map_Results.csv"
test_root = root + "MICCAI2024-STAGE2/STAGE_validation/validation_images"
oct_img_size = [512, 512]
fundus_img_size = [512, 512]
device = 'cuda:0'

oct_test_transforms = trans.Compose([
    trans.CenterCrop([256] + oct_img_size)
])

fundus_test_transforms = trans.Compose([
    trans.Resize(fundus_img_size)
])

oct_tta_transforms = trans.Compose([
    trans.CenterRandomCrop([256] + oct_img_size),
    trans.RandomHorizontalFlip()
])

fundus_tta_transforms = trans.Compose([
    trans.Resize(fundus_img_size),
    trans.RandomHorizontalFlip(),
    trans.RandomVerticalFlip(),
])

test_dataset = STAGE_sub2_dataset(dataset_root=test_root,
                                  oct_transforms=oct_test_transforms,
                                  fundus_transforms=fundus_test_transforms,
                                  info_file=root + 'MICCAI2024-STAGE2/STAGE_validation/data_info_val.xlsx',
                                  mode='test', TTA=True,
                                  oct_TTA_transforms=oct_tta_transforms, fundus_TTA_transforms=fundus_tta_transforms)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    num_workers=2,
    pin_memory=True
)

best_model_path = [
    "./model_checkpoints/20240728_220910_797630", 
]
kfold = 5

columns = ['ID'] + ['point'+str(i) for i in range(1, 52+1)]
caches = []
for i in range(len(best_model_path)):
    for fold in range(kfold):
        print('Model {}/{}: '.format(i+1, fold), best_model_path[i], fold)
        model = Model().to(device)
        para_state_dict = torch.load(os.path.join(best_model_path[i], str(fold), 'best_model.pth'))
        model.load_state_dict(para_state_dict)

        ids, cache = test(test_loader, model)
        caches.append(cache)

caches = np.stack(caches, 0)
print(caches.shape)
caches = np.mean(caches, 0)  # calculate the average value along the first dimension
caches[caches < 1] = 0
caches = np.concatenate((ids[:, np.newaxis], caches), axis=-1)
submission_result = pd.DataFrame(list(caches), columns=columns)
submission_result.to_csv(save_root, index=False)
print('Finish: ', save_root)
