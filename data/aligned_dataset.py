import os.path
import torch
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _pair_identity(path):
    stem = _stem(path)
    return _pair_identity_from_stem(stem)


def _pair_identity_from_stem(stem):
    parts = stem.split('_')
    if len(parts) >= 3:
        return '{}_{}'.format(parts[0], parts[-1])
    return stem


def _build_unique_map(paths, key_fn):
    mapping = {}
    for path in paths:
        key = key_fn(path)
        if key in mapping:
            raise ValueError('Duplicate pair key "{}" for files: {} and {}'.format(key, mapping[key], path))
        mapping[key] = path
    return mapping


def _align_b_to_a(A_paths, B_paths):
    if len(A_paths) != len(B_paths):
        raise ValueError('Pairing mismatch: {} files in A vs {} files in B'.format(len(A_paths), len(B_paths)))

    a_stems = [_stem(path) for path in A_paths]
    b_stem_map = _build_unique_map(B_paths, _stem)
    if all(stem in b_stem_map for stem in a_stems):
        return [b_stem_map[stem] for stem in a_stems]

    a_keys = [_pair_identity(path) for path in A_paths]
    b_key_map = _build_unique_map(B_paths, _pair_identity)
    missing = [key for key in a_keys if key not in b_key_map]
    if missing:
        sample = ', '.join(missing[:5])
        raise ValueError('Pairing mismatch: could not find B files for A keys: {}'.format(sample))
    return [b_key_map[key] for key in a_keys]


def _mask_stem(path, mask_suffix):
    stem = _stem(path)
    if mask_suffix and stem.endswith(mask_suffix):
        return stem[:-len(mask_suffix)]
    return stem


def _build_mask_map(paths, mask_suffix, use_pair_identity=False):
    mapping = {}
    for path in paths:
        stem = _mask_stem(path, mask_suffix)
        key = _pair_identity_from_stem(stem) if use_pair_identity else stem
        if key in mapping:
            raise ValueError('Duplicate mask key "{}" for files: {} and {}'.format(key, mapping[key], path))
        mapping[key] = path
    return mapping


def _align_masks_to_images(image_paths, mask_paths, mask_suffix):
    image_stems = [_stem(path) for path in image_paths]
    mask_stem_map = _build_mask_map(mask_paths, mask_suffix, use_pair_identity=False)
    if all(stem in mask_stem_map for stem in image_stems):
        return [mask_stem_map[stem] for stem in image_stems]

    image_keys = [_pair_identity(path) for path in image_paths]
    mask_key_map = _build_mask_map(mask_paths, mask_suffix, use_pair_identity=True)
    missing = [key for key in image_keys if key not in mask_key_map]
    if missing:
        sample = ', '.join(missing[:5])
        raise ValueError('Mask pairing mismatch: could not find mask files for image keys: {}'.format(sample))
    return [mask_key_map[key] for key in image_keys]

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))
            self.B_paths = _align_b_to_a(self.A_paths, self.B_paths)

        self.use_mask_channel = bool(getattr(opt, 'use_segmentation_mask_channel', False))
        self.mask_suffix = getattr(opt, 'mask_suffix', '_mask')
        if self.use_mask_channel:
            if self.opt.label_nc != 0:
                raise ValueError('--use_segmentation_mask_channel requires --label_nc 0 (RGB image-to-image mode).')
            if opt.mask_dataroot:
                self.mask_root = opt.mask_dataroot
            else:
                self.mask_root = opt.dataroot + '_mask'

            self.dir_A_mask = os.path.join(self.mask_root, opt.phase + dir_A)
            self.A_mask_paths = sorted(make_dataset(self.dir_A_mask))
            self.A_mask_paths = _align_masks_to_images(self.A_paths, self.A_mask_paths, self.mask_suffix)

            if opt.isTrain or opt.use_encoded_image:
                self.dir_B_mask = os.path.join(self.mask_root, opt.phase + dir_B)
                self.B_mask_paths = sorted(make_dataset(self.dir_B_mask))
                self.B_mask_paths = _align_masks_to_images(self.B_paths, self.B_mask_paths, self.mask_suffix)

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = Image.open(A_path)        
        params = get_params(self.opt, A.size)
        pair_intensity_augment = self.opt.isTrain and not self.opt.no_augment and self.opt.label_nc == 0
        if self.opt.label_nc == 0:
            transform_A = get_transform(
                self.opt,
                params,
                apply_intensity_augment=pair_intensity_augment,
            )
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        transform_mask = None
        if self.use_mask_channel:
            transform_mask = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_mask = Image.open(self.A_mask_paths[index]).convert('L')
            A_mask_tensor = transform_mask(A_mask)
            A_mask_tensor = (A_mask_tensor > 0.5).float() * 2.0 - 1.0
            A_tensor = torch.cat((A_tensor, A_mask_tensor), dim=0)

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            if self.opt.label_nc == 0:
                transform_B = transform_A
            else:
                transform_B = get_transform(
                    self.opt,
                    params,
                    apply_intensity_augment=(self.opt.isTrain and not self.opt.no_augment),
                )
            B_tensor = transform_B(B)
            if self.use_mask_channel:
                B_mask = Image.open(self.B_mask_paths[index]).convert('L')
                B_mask_tensor = transform_mask(B_mask)
                B_mask_tensor = (B_mask_tensor > 0.5).float() * 2.0 - 1.0
                B_tensor = torch.cat((B_tensor, B_mask_tensor), dim=0)

        if self.use_mask_channel and self.opt.label_nc == 0:
            if A_tensor.size(0) != self.opt.input_nc:
                raise ValueError('Input channel mismatch: tensor has {} channels, but --input_nc is {}.'.format(A_tensor.size(0), self.opt.input_nc))
            if (self.opt.isTrain or self.opt.use_encoded_image) and B_tensor.size(0) != self.opt.output_nc:
                raise ValueError('Output channel mismatch: tensor has {} channels, but --output_nc is {}.'.format(B_tensor.size(0), self.opt.output_nc))

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {
            'label': A_tensor,
            'inst': inst_tensor,
            'image': B_tensor,
            'feat': feat_tensor,
            'path': A_path,
            'B_path': B_path if (self.opt.isTrain or self.opt.use_encoded_image) else '',
        }

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'