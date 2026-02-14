import time
import os
import numpy as np
import csv

configured_tmpdir = os.environ.get('TMPDIR')
if not configured_tmpdir or len(configured_tmpdir) > 40 or not os.path.isdir(configured_tmpdir):
    for _tmpdir in ['/tmp', '/var/tmp']:
        if os.path.isdir(_tmpdir) and os.access(_tmpdir, os.W_OK):
            os.environ['TMPDIR'] = _tmpdir
            break

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import math
def lcm(a,b): return abs(a * b)/math.gcd(a,b) if a and b else 0

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer

opt = TrainOptions().parse()
is_distributed = getattr(opt, 'is_distributed', False)
if is_distributed and not dist.is_initialized():
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend, init_method='env://')
rank = dist.get_rank() if is_distributed else 0
is_main_process = rank == 0

if is_distributed and not is_main_process:
    opt.no_html = True
    opt.tf_log = False

iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    if is_main_process:
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:    
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)    
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
if is_main_process:
    print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt) if is_main_process else None

pair_audit_csv = None
if is_main_process:
    pair_audit_dir = os.path.join(opt.checkpoints_dir, opt.name, 'metrics')
    os.makedirs(pair_audit_dir, exist_ok=True)
    pair_audit_csv = os.path.join(pair_audit_dir, 'pair_audit.csv')
    if not os.path.exists(pair_audit_csv) or os.path.getsize(pair_audit_csv) == 0:
        with open(pair_audit_csv, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['epoch', 'epoch_iter', 'total_steps', 'a_path', 'b_path'])

def unwrap_model(m):
    return m.module if isinstance(m, (torch.nn.DataParallel, DDP)) else m

if opt.fp16:    
    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler()
    if is_distributed:
        model = model.cuda(opt.gpu_ids[0])
        model = DDP(model, device_ids=opt.gpu_ids, output_device=opt.gpu_ids[0])
    elif not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
else:
    scaler = None
    if is_distributed:
        model = model.cuda(opt.gpu_ids[0])
        model = DDP(model, device_ids=opt.gpu_ids, output_device=opt.gpu_ids[0])

base_model = unwrap_model(model)
optimizer_G, optimizer_D = base_model.optimizer_G, base_model.optimizer_D

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if hasattr(data_loader, 'set_epoch'):
        data_loader.set_epoch(epoch)
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        if is_main_process and pair_audit_csv is not None:
            a_paths = data.get('path', [])
            b_paths = data.get('B_path', [])
            if isinstance(a_paths, str):
                a_paths = [a_paths]
            if isinstance(b_paths, str):
                b_paths = [b_paths]
            max_len = max(len(a_paths), len(b_paths), 1)
            with open(pair_audit_csv, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                for idx in range(max_len):
                    a_path = a_paths[idx] if idx < len(a_paths) else ''
                    b_path = b_paths[idx] if idx < len(b_paths) else ''
                    writer.writerow([epoch, epoch_iter, total_steps, a_path, b_path])

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ############## Forward Pass ######################
        if opt.fp16:
            with autocast():
                losses, generated = model(Variable(data['label']), Variable(data['inst']), 
                    Variable(data['image']), Variable(data['feat']), infer=save_fake)
        else:
            losses, generated = model(Variable(data['label']), Variable(data['inst']), 
                Variable(data['image']), Variable(data['feat']), infer=save_fake)

        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(unwrap_model(model).loss_names, losses))

        if is_distributed:
            reduced = {}
            for key, value in loss_dict.items():
                if isinstance(value, int):
                    reduced[key] = value
                    continue
                reduced_val = value.detach().clone()
                dist.all_reduce(reduced_val, op=dist.ReduceOp.SUM)
                reduced[key] = reduced_val / dist.get_world_size()
            loss_dict_reduced = reduced
        else:
            loss_dict_reduced = loss_dict

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = (
            loss_dict['G_GAN']
            + loss_dict.get('G_GAN_Feat', 0)
            + loss_dict.get('G_VGG', 0)
            + opt.lambda_ssim * loss_dict.get('G_SSIM', 0)
            + opt.lambda_gradvar * loss_dict.get('G_GradVar', 0)
            + opt.lambda_mask * loss_dict.get('G_Mask', 0)
        )

        ############### Backward Pass ####################
        # update generator weights
        optimizer_G.zero_grad()
        if opt.fp16:                                
            scaler.scale(loss_G).backward()                
        else:
            loss_G.backward()          
        if opt.fp16:
            scaler.step(optimizer_G)
            scaler.update()
        else:
            optimizer_G.step()

        # update discriminator weights
        optimizer_D.zero_grad()
        if opt.fp16:                                
            scaler.scale(loss_D).backward()                
        else:
            loss_D.backward()        
        if opt.fp16:
            scaler.step(optimizer_D)
            scaler.update()
        else:
            optimizer_D.step()        

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict_reduced.items()}
            t = (time.time() - iter_start_time) / opt.print_freq
            if is_main_process:
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)
            #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

        ### display output images
        if save_fake:
            visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                   ('synthesized_image', util.tensor2im(generated.data[0])),
                                   ('real_image', util.tensor2im(data['image'][0]))])
            if is_main_process:
                visualizer.display_current_results(visuals, epoch, total_steps)

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            if is_main_process:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                unwrap_model(model).save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break
       
    # end of epoch 
    iter_end_time = time.time()
    if is_main_process:
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        if is_main_process:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            unwrap_model(model).save('latest')
            unwrap_model(model).save(epoch)
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        unwrap_model(model).update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        unwrap_model(model).update_learning_rate()

if is_main_process:
    try:
        visualizer.save_training_plots()
    except Exception as e:
        print('Failed to generate training plots: %s' % str(e))

    try:
        visualizer.save_training_summary()
    except Exception as e:
        print('Failed to generate training summary: %s' % str(e))

if is_distributed and dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()
