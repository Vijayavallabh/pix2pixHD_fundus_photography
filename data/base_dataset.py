import torch.utils.data as data
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import random

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize            
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))
    
    flip = random.random() > 0.5
    vflip = random.random() > 0.5
    angle = 0.0
    if opt.isTrain and not opt.no_augment and opt.aug_rotate > 0:
        angle = random.uniform(-opt.aug_rotate, opt.aug_rotate)
    contrast_factor = 1.0
    if opt.isTrain and not opt.no_augment and opt.aug_contrast > 0:
        low = max(0.0, 1.0 - opt.aug_contrast)
        high = 1.0 + opt.aug_contrast
        contrast_factor = random.uniform(low, high)

    noise_std = 0.0
    if opt.isTrain and not opt.no_augment and opt.aug_noise_std > 0:
        noise_std = opt.aug_noise_std

    return {
        'crop_pos': (x, y),
        'flip': flip,
        'vflip': vflip,
        'angle': angle,
        'contrast_factor': contrast_factor,
        'noise_std': noise_std,
    }

def get_transform(opt, params, method=Image.BICUBIC, normalize=True, apply_intensity_augment=False):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, method))   
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))

    if opt.isTrain and not opt.no_augment and params.get('angle', 0) != 0:
        transform_list.append(transforms.Lambda(lambda img: __rotate(img, params['angle'], method)))
        
    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
        transform_list.append(transforms.Lambda(lambda img: __vflip(img, params['vflip'])))

    transform_list += [transforms.ToTensor()]

    if apply_intensity_augment and opt.isTrain and not opt.no_augment:
        if params.get('contrast_factor', 1.0) != 1.0:
            transform_list.append(transforms.Lambda(
                lambda tensor: __adjust_contrast(tensor, params['contrast_factor'])
            ))
        if params.get('noise_std', 0.0) > 0:
            transform_list.append(transforms.Lambda(
                lambda tensor: __add_gaussian_noise(tensor, params['noise_std'])
            ))

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def normalize():    
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size        
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)

def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img    
    w = target_width
    h = int(target_width * oh / ow)    
    return img.resize((w, h), method)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):        
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def __vflip(img, vflip):
    if vflip:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    return img

def __rotate(img, angle, method=Image.BICUBIC):
    if angle == 0:
        return img
    return img.rotate(angle, resample=method)

def __adjust_contrast(tensor, factor):
    return F.adjust_contrast(tensor, factor)

def __add_gaussian_noise(tensor, std):
    noise = torch.randn_like(tensor) * std
    return torch.clamp(tensor + noise, 0.0, 1.0)
