import argparse
import os
import tempfile
from util import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='label2city', help='name of the experiment. It decides where to store samples and models')        
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--model', type=str, default='pix2pixHD', help='which model to use')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')        
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit")
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')
        self.parser.add_argument('--fp16', action='store_true', default=False, help='train with AMP')
        self.parser.add_argument('--local_rank', '--local-rank', dest='local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)), help='local rank for distributed training')

        # input/output sizes       
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=1024, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        self.parser.add_argument('--label_nc', type=int, default=35, help='# of input label channels')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='./datasets/cityscapes/') 
        self.parser.add_argument('--resize_or_crop', type=str, default='scale_width', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')        
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, disable horizontal and vertical flip augmentation') 
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')                
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        # data augmentation
        self.parser.add_argument('--no_augment', action='store_true', help='if specified, disable data augmentation during training')
        self.parser.add_argument('--aug_rotate', type=float, default=5.0, help='max rotation degrees for augmentation (0 to disable)')
        self.parser.add_argument('--aug_contrast', type=float, default=0.2, help='max relative contrast adjustment strength (0 to disable)')
        self.parser.add_argument('--aug_noise_std', type=float, default=0.02, help='Gaussian noise std on [0,1] image tensor (0 to disable)')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for generator
        self.parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG') 
        self.parser.add_argument('--n_blocks_global', type=int, default=9, help='number of residual blocks in the global generator network')
        self.parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')        
        self.parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')        
        self.parser.add_argument('--use_attention', action='store_true', help='if specified, insert self-attention blocks in local/global generators')
        self.parser.add_argument('--lambda_ssim', type=float, default=1.0, help='weight for SSIM loss term')
        self.parser.add_argument('--lambda_gradvar', type=float, default=10.0, help='weight for gradient variance loss term')

        # for instance-wise features
        self.parser.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input')        
        self.parser.add_argument('--instance_feat', action='store_true', help='if specified, add encoded instance features as input')
        self.parser.add_argument('--label_feat', action='store_true', help='if specified, add encoded label features as input')        
        self.parser.add_argument('--feat_num', type=int, default=3, help='vector length for encoded features')        
        self.parser.add_argument('--load_features', action='store_true', help='if specified, load precomputed feature maps')
        self.parser.add_argument('--n_downsample_E', type=int, default=4, help='# of downsampling layers in encoder') 
        self.parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')        
        self.parser.add_argument('--n_clusters', type=int, default=10, help='number of clusters for features')        

        self.initialized = True

    def _configure_multiprocessing_tempdir(self):
        tmpdir_candidates = ['/tmp', '/var/tmp']
        configured_tmpdir = os.environ.get('TMPDIR')

        if configured_tmpdir and len(configured_tmpdir) <= 40 and os.path.isdir(configured_tmpdir) and os.access(configured_tmpdir, os.W_OK):
            tempfile.tempdir = configured_tmpdir
            return

        for tmpdir in tmpdir_candidates:
            if os.path.isdir(tmpdir) and os.access(tmpdir, os.W_OK):
                os.environ['TMPDIR'] = tmpdir
                tempfile.tempdir = tmpdir
                return

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self._configure_multiprocessing_tempdir()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test
        self.opt.world_size = int(os.environ.get('WORLD_SIZE', '1'))
        self.opt.rank = int(os.environ.get('RANK', '0'))
        self.opt.is_distributed = self.opt.world_size > 1

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            if self.opt.is_distributed:
                if self.opt.local_rank < len(self.opt.gpu_ids):
                    selected_gpu = self.opt.gpu_ids[self.opt.local_rank]
                else:
                    selected_gpu = self.opt.local_rank
                torch.cuda.set_device(selected_gpu)
                self.opt.gpu_ids = [selected_gpu]
            else:
                torch.cuda.set_device(self.opt.gpu_ids[0])


        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
