import numpy as np
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, \
    BrightnessTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform


def get_train_transform():
    tr_transforms = []
    tr_transforms.append(BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=0.15, data_key='data'))
    tr_transforms.append(BrightnessTransform(0.0, 0.1, True, p_per_sample=0.15, p_per_channel=0.5, data_key='data'))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15, data_key='data'))
    tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.), different_sigma_per_channel=True,
                                               p_per_channel=0.5, p_per_sample=0.2, data_key='data'))
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1, data_key='data'))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def collate_fn_w_transform(batch):
    oct_imgs, label, info = zip(*batch)
    oct_imgs = np.stack(oct_imgs, 0)
    label = np.stack(label, 0)
    info = np.stack(info, 0)

    stride = oct_imgs.shape[0] // 2
    for i in range(0, oct_imgs.shape[0], stride):    # To avoid 4GB limit
        data_dict = {'data': oct_imgs[i:i+stride].astype(np.float64)}
        tr_transforms = get_train_transform()
        data_dict = tr_transforms(**data_dict)
        oct_imgs[i:i+stride] = data_dict['data']
    return oct_imgs, label, info

