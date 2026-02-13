CUDA_VISIBLE_DEVICES=3,4 nohup torchrun --nproc_per_node=2 --master_port=29995 train.py --dataroot ./datasets/eye_cropped_1 --name EYE_fund_train_1_2k --load_pretrain checkpoints/EYE_fund_train_1 --label_nc 0 --no_instance --resize_or_crop resize_and_crop --loadSize 2048 --fineSize 2048 --batchSize 2 --niter 80 --niter_decay 80 --niter_fix_global 20 --lr 0.00002 --netG local --ngf 64 --aug_noise_std 0.05 --num_D 3 --max_dataset_size 12 --use_attention --gpu_ids 0,1 > out_fund_train_1_2k.log 2>&1&


CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataroot ./datasets/eye_cropped_1 --name EYE_fund_train_1 --label_nc 0 --no_instance --resize_or_crop resize_and_crop --loadSize 1024 --fineSize 1024 --batchSize 1  --max_dataset_size 12 --use_attention --nThreads 0 > out_fund_train_1.log 2>&1&
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataroot ./datasets/eye_cropped --name EYE_fund_train --label_nc 0 --no_instance --resize_or_crop resize_and_crop --loadSize 1024 --fineSize 1024 --batchSize 1  --max_dataset_size 12 --use_attention --nThreads 0 > out_fund_train.log 2>&1&
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataroot ./datasets/eye_cropped_1_no_geometric --name EYE_fund_train_1_no_geometric --label_nc 0 --no_instance --resize_or_crop resize_and_crop --loadSize 1024 --fineSize 1024 --batchSize 1  --max_dataset_size 12 --use_attention --nThreads 0 > out_fund_train_1_no_geometric.log 2>&1&
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataroot ./datasets/eye_cropped_no_geometric --name EYE_fund_train_no_geometric --label_nc 0 --no_instance --resize_or_crop resize_and_crop --loadSize 1024 --fineSize 1024 --batchSize 1  --max_dataset_size 12 --use_attention --nThreads 0 > out_fund_train_no_geometric.log 2>&1&



CUDA_VISIBLE_DEVICES=0 nohup python test.py --dataroot ./datasets/eye_cropped_1 --name EYE_fund_train_1_2k --label_nc 0 --no_instance --resize_or_crop resize_and_crop --loadSize 2048 --fineSize 2048 --batchSize 1  --use_attention --nThreads 0  --niter 50  --niter_fix_global 10 --netG local --ngf 32  > out_test_fund_1_2k.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python test.py --dataroot ./datasets/eye_cropped_1 --name EYE_fund_train_1 --label_nc 0 --no_instance --resize_or_crop resize_and_crop --loadSize 1024 --fineSize 1024 --batchSize 1  --use_attention --nThreads 0 > out_test_fund_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python test.py --dataroot ./datasets/eye_cropped --name EYE_fund_train --label_nc 0 --no_instance --resize_or_crop resize_and_crop --loadSize 1024 --fineSize 1024 --batchSize 1  --use_attention --nThreads 0 > out_test_fund.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python test.py --dataroot ./datasets/eye_cropped_1_no_geometric --name EYE_fund_train_1_no_geometric --label_nc 0 --no_instance --resize_or_crop resize_and_crop --loadSize 1024 --fineSize 1024 --batchSize 1  --use_attention --nThreads 0 > out_test_fund_1_no_geometric.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python test.py --dataroot ./datasets/eye_cropped_no_geometric --name EYE_fund_train_no_geometric --label_nc 0 --no_instance --resize_or_crop resize_and_crop --loadSize 1024 --fineSize 1024 --batchSize 1  --use_attention --nThreads 0 > out_test_fund_no_geometric.log 2>&1 &



CUDA_VISIBLE_DEVICES=0 nohup python compute_psnr.py --phases test --dataroot ./datasets/eye_cropped_1 --name EYE_fund_train_1_2k --label_nc 0 --no_instance --resize_or_crop resize_and_crop --loadSize 2048 --fineSize 2048 --batchSize 1 --use_attention  --niter 50  --niter_fix_global 10 --netG local --ngf 32  --nThreads 0 > out_psnr_fund_1_2k.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python compute_psnr.py --phases test --dataroot ./datasets/eye_cropped_1 --name EYE_fund_train_1 --label_nc 0 --no_instance --resize_or_crop resize_and_crop --loadSize 1024 --fineSize 1024 --batchSize 1 --use_attention --nThreads 0 > out_psnr_fund_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python compute_psnr.py --phases test --dataroot ./datasets/eye_cropped --name EYE_fund_train --label_nc 0 --no_instance --resize_or_crop resize_and_crop --loadSize 1024 --fineSize 1024 --batchSize 1 --use_attention --nThreads 0 > out_psnr_fund.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python compute_psnr.py --phases test --dataroot ./datasets/eye_cropped_no_geometric --name EYE_fund_train_no_geometric --label_nc 0 --no_instance --resize_or_crop resize_and_crop --loadSize 1024 --fineSize 1024 --batchSize 1 --use_attention --nThreads 0 > out_psnr_fund_no_geometric.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python compute_psnr.py --phases test --dataroot ./datasets/eye_cropped_1_no_geometric --name EYE_fund_train_1_no_geometric --label_nc 0 --no_instance --resize_or_crop resize_and_crop --loadSize 1024 --fineSize 1024 --batchSize 1 --use_attention --nThreads 0 > out_psnr_fund_1_no_geometric.log 2>&1 &