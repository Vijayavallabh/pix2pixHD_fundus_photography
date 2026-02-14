# pix2pixHD_fundus_photography

## Prerequisites
- Linux or macOS
- Python 3.12
- NVIDIA GPU (11G memory or larger) + CUDA cuDNN

## Getting Started
### Installation
- Install `uv`: https://docs.astral.sh/uv/getting-started/installation/
- Clone this repo:
```bash
git clone https://github.com/Vijayavallabh/pix2pixHD_fundus_photography
cd pix2pixHD_fundus_photography
```
- Create and use a Python 3.12 virtual environment with `uv`, then install dependencies from `requirements.txt`:
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```


### Training with your own dataset
- If you want to train with your own dataset, please generate label maps which are one-channel whose pixel values correspond to the object labels (i.e. 0,1,...,N-1, where N is the number of labels). This is because we need to generate one-hot vectors from the label maps. Please also specity `--label_nc N` during both training and testing.
- If your input is not a label map, please just specify `--label_nc 0` which will directly use the RGB colors as input. The folders should then be named `train_A`, `train_B` instead of `train_label`, `train_img`, where the goal is to translate images from A to B.
- If you don't have instance maps or don't want to use them, please specify `--no_instance`.
- The default setting for preprocessing is `scale_width`, which will scale the width of all training images to `opt.loadSize` (1024) while keeping the aspect ratio. If you want a different setting, please change it by using the `--resize_or_crop` option. For example, `scale_width_and_crop` first resizes the image to have width `opt.loadSize` and then does random cropping of size `(opt.fineSize, opt.fineSize)`. `crop` skips the resizing step and only performs random cropping. If you don't want any preprocessing, please specify `none`, which will do nothing other than making sure the image is divisible by 32.

# Example commands for `datasets/eye_cropped_1`
- Train:
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataroot ./datasets/eye_cropped_1 --name EYE_fund_train_1 --label_nc 0 --no_instance --resize_or_crop resize_and_crop --loadSize 1024 --fineSize 1024 --batchSize 1  --max_dataset_size 12 --use_attention --nThreads 0 > out_fund_train_1.log 2>&1&
```
- Train [Finetune with 2k Resolution]
```bash
CUDA_VISIBLE_DEVICES=3,5 nohup python -m torch.distributed.launch train.py --dataroot ./datasets/eye_cropped_1 --name EYE_fund_train_1_2k --load_pretrain checkpoints/EYE_fund_train_1 --label_nc 0 --no_instance --resize_or_crop resize_and_crop --loadSize 2048 --fineSize 2048 --batchSize 4 --niter 50 --niter_decay 50 --niter_fix_global 10 --netG local --ngf 32 --num_D 3 --max_dataset_size 12 --use_attention --gpu_ids 0,1 > out_fund_train_1_2k.log 2>&1&
```
- Test:
```bash
CUDA_VISIBLE_DEVICES=0 nohup python test.py --dataroot ./datasets/eye_cropped_1 --name EYE_fund_train_1 --label_nc 0 --no_instance --resize_or_crop resize_and_crop --loadSize 1024 --fineSize 1024 --batchSize 1  --use_attention --nThreads 0 > out_test_fund_1.log 2>&1 &
```
- Compute PSNR:
```bash
CUDA_VISIBLE_DEVICES=0 nohup python compute_psnr.py --phases test --dataroot ./datasets/eye_cropped_1 --name EYE_fund_train_1 --label_nc 0 --no_instance --resize_or_crop resize_and_crop --loadSize 1024 --fineSize 1024 --batchSize 1 --use_attention --nThreads 0 > out_psnr_fund_1.log 2>&1 &
```

Training augmentation is restricted to: horizontal flip, vertical flip, rotation (`--aug_rotate`), contrast adjustment (`--aug_contrast`), and Gaussian noise injection (`--aug_noise_std`). Use `--no_augment` to disable augmentation entirely.

## Image Preprocessing Script (s.py)

`s.py` is a custom Python script designed to preprocess images in the `datasets/eye` folder for the fundus photography dataset. It performs the following operations:

- **Black Border Cropping**: Removes black borders from all images in `train_A`, `train_B`, `test_A`, and `test_B` subfolders by detecting and cropping uniform black pixel regions at the edges.
- **Watermark Removal**: For images in `train_B` and `test_B`, it blacks out specific patches in the bottom-left and bottom-right corners to remove watermarks. The patch sizes are dynamically calculated based on image dimensions, with the bottom-right patch having a reduced height (scaled by 0.65) and different width scales (left: 0.55, right: 0.65) to precisely target watermark locations without affecting the main image content.

Processed images are saved to a new directory `datasets/eye_cropped`, preserving the original folder structure.

To run the script:
```bash
python s.py
```

Optional geometric/color postprocessing:
```bash
python s.py --enable_affine_registration --enable_color_normalization
```
If `train_A` and `train_B` (or `test_A` and `test_B`) filenames do not match, the script automatically switches to unpaired folder-wise reference mode.

This preprocessing step ensures clean, watermark-free images for training and testing the pix2pixHD model on the eye dataset.

## Additional Image Preprocessing Script (s_1.py)

`s_1.py` is an enhanced version of `s.py` with additional vertical trimming functionality. It performs the same black border cropping and watermark removal as `s.py`, but also includes:

- **Vertical Trimming**: For images in `train_A` and `test_A` folders, it removes 110 pixels from the top and 115 pixels from the bottom after cropping black borders. This helps remove unwanted header/footer regions specific to the A-type images.

Processed images are saved to `datasets/eye_cropped_1`, maintaining the same folder structure.

To run the script:
```bash
python s_1.py
```

If `train_A` and `train_B` (or `test_A` and `test_B`) filenames do not match, the script automatically switches to unpaired folder-wise reference mode.

Use this script when additional vertical cropping is needed for the A-type images in the dataset.

# CityScapes Dataset Usage

### Testing
- A few example Cityscapes test images are included in the `datasets` folder.
- Please download the pre-trained Cityscapes model from [here](https://drive.google.com/file/d/1OR-2aEPHOxZKuoOV34DvQxreqGCSLcW9/view?usp=drive_link) (google drive link), and put it under `./checkpoints/label2city_1024p/`
- Test the model (`bash ./scripts/test_1024p.sh`):
```bash
#!./scripts/test_1024p.sh
python test.py --name label2city_1024p --netG local --ngf 32 --resize_or_crop none
```
The test results will be saved to a html file here: `./results/label2city_1024p/test_latest/index.html`.

More example scripts can be found in the `scripts` directory.


### Dataset
- We use the Cityscapes dataset. To train a model on the full dataset, please download it from the [official website](https://www.cityscapes-dataset.com/) (registration required).
After downloading, please put it under the `datasets` folder in the same way the example images are provided.


### Training
- Train a model at 1024 x 512 resolution (`bash ./scripts/train_512p.sh`):
```bash
#!./scripts/train_512p.sh
python train.py --name label2city_512p
```
- To view training results, please checkout intermediate results in `./checkpoints/label2city_512p/web/index.html`.
If you have tensorflow installed, you can see tensorboard logs in `./checkpoints/label2city_512p/logs` by adding `--tf_log` to the training scripts.
- At the end of training, additional monitoring artifacts are saved to `./checkpoints/<run_name>/metrics/`:
  - `loss_history.csv` (structured loss history including `G_GAN`, `G_GAN_Feat`, `G_VGG`, `G_SSIM`, `G_GradVar`, `D_real`, `D_fake`, plus derived metrics)
  - `loss_curves.png` and `d_balance.png` (loss and discriminator-balance plots)
  - `training_summary.txt` and `best_step.json` (heuristic best-step summary and metadata)

### Multi-GPU training
- Train a model using multiple GPUs (`bash ./scripts/train_512p_multigpu.sh`):
```bash
#!./scripts/train_512p_multigpu.sh
python train.py --name label2city_512p --batchSize 8 --gpu_ids 0,1,2,3,4,5,6,7
```
Note: this is not tested and we trained our model using single GPU only. Please use at your own discretion.

### Training with Automatic Mixed Precision (AMP) for faster speed
- To train with mixed precision support, please first install apex from: https://github.com/NVIDIA/apex
- You can then train the model by adding `--fp16`. For example,
```bash
#!./scripts/train_512p_fp16.sh
python -m torch.distributed.launch train.py --name label2city_512p --fp16
```
In our test case, it trains about 80% faster with AMP on a Volta machine.

### Training at full resolution
- To train the images at full resolution (2048 x 1024) requires a GPU with 24G memory (`bash ./scripts/train_1024p_24G.sh`), or 16G memory if using mixed precision (AMP).
- If only GPUs with 12G memory are available, please use the 12G script (`bash ./scripts/train_1024p_12G.sh`), which will crop the images during training. Performance is not guaranteed using this script.



