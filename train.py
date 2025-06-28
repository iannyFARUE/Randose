# -*- encoding: utf-8 -*-
import os
import sys
if os.path.abspath('..') not in sys.path:
    sys.path.insert(0, os.path.abspath('..'))

import argparse

from utils import *
from dose.e_metrics import *

from models.mt import *

from models.mt1 import *

from models.mf import *

from models.model_m import *

from models.mam_t_v2 import *

from models.mt2 import *

from models.mt3 import *

from models.mt31 import *
 
from models.mt4 import *

from models.mtss import *

from models.modelAS import *

from models.models import *

from models.mt5 import *

from loss import *
import torch

import gc

gc.collect()
torch.cuda.empty_cache()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int,
                        help='batch size for training')
    parser.add_argument('--list_GPU_ids', nargs='+', type=int,
                        help='list_GPU_ids for training ')
    parser.add_argument('--max_iter', type=int,
                        help='training iterations')
    parser.add_argument('--model', type=str,
                        help='model to use ')
    parser.add_argument('--project_name', type=str,
                        help='project name')
    parser.add_argument('--loss',type = str, 
                        help='loss to use' )
    args = parser.parse_args()

    # Start training
    trainer = NetworkTrainer()
    trainer.setting.project_name = args.project_name
    trainer.setting.output_dir = args.project_name
    list_GPU_ids = args.list_GPU_ids

    # Choose the model based on the argument
    if args.model == 'Model':
        trainer.setting.network = Model(
            in_ch=9, out_ch=1,
            list_ch_A=[-1, 16, 32, 64, 128, 256],
            list_ch_B=[-1, 32, 64, 128, 256, 512]
        )
    

    elif args.model == 'Model_MT':
        trainer.setting.network = Model_MT(in_ch=9, out_ch=1,
                  list_ch_A=[-1, 16, 32, 64, 128, 256],
                  list_ch_B=[-1, 32, 64, 128, 256, 512],
                  d_state=16, d_conv=4, expand=2, channel_token=False)
        
    elif args.model == 'Model_M':
        trainer.setting.network = Model_M(in_ch=9, out_ch=1,
                  list_ch_A=[-1, 16, 32, 64, 128, 256],
                  list_ch_B=[-1, 32, 64, 128, 256, 512],
                  d_state=16, d_conv=4, expand=2, channel_token=False)
        
    elif args.model == 'Model_MTA':
        trainer.setting.network = Model_MTA(in_ch=9, out_ch=1,
                  list_ch_A=[-1, 16, 32, 64, 128, 256],
                  list_ch_B=[-1, 32, 64, 128, 256, 512],
                  d_state=16, d_conv=4, expand=2, channel_token=False)
        
    elif args.model == 'Model_Mambaformer':
        trainer.setting.network = Model_MT(in_ch=9, out_ch=1,
                  list_ch_A=[-1, 16, 32, 64, 128, 256],
                  list_ch_B=[-1, 32, 64, 128, 256, 512],
                  d_state=16, d_conv=4, expand=2, channel_token=False)
        
    elif args.model == 'Model_MTA2':  
        trainer.setting.network = Model_MTA2(in_ch=9, out_ch=1,
                  list_ch_A=[-1, 16, 32, 64, 128, 256],
                  list_ch_B=[-1, 32, 64, 128, 256, 512],
                  d_state=16, d_conv=4, expand=2, channel_token=False)
        
    elif args.model == 'Model_MTA3':  
        trainer.setting.network = Model_MTA3(in_ch=9, out_ch=1,
                  list_ch_A=[-1, 16, 32, 64, 128, 256],
                  list_ch_B=[-1, 32, 64, 128, 256, 512],
                  d_state=16, d_conv=4, expand=2, channel_token=False)
        
    elif args.model == 'Model_MTA4':  
        trainer.setting.network = Model_MTA4(in_ch=9, out_ch=1,
                  list_ch_A=[-1, 16, 32, 64, 128, 256],
                  list_ch_B=[-1, 32, 64, 128, 256, 512],
                  d_state=16, d_conv=4, expand=2, channel_token=False)
        
    elif args.model == 'Model_MTA4_1':  
        trainer.setting.network = Model_MTA4_1(in_ch=9, out_ch=1,
                  list_ch_A=[-1, 16, 32, 64, 128, 256],
                  list_ch_B=[-1, 32, 64, 128, 256, 512],
                  d_state=16, d_conv=4, expand=2, channel_token=False)
        
    elif args.model == 'Model_MTA5':  
        trainer.setting.network = Model_MTA5(in_ch=1, out_ch=1,
                  list_ch_A=[-1, 16, 32, 64, 128, 256],
                  list_ch_B=[-1, 32, 64, 128, 256, 512],
                  d_state=16, d_conv=4, expand=2, channel_token=False)
        
    elif args.model == 'Model_MTAS':  
        trainer.setting.network = Model_MTAS(in_ch=1, out_ch=1,
                  list_ch_A=[-1, 16, 32, 64, 128, 256],
                  list_ch_B=[-1, 32, 64, 128, 256, 512],
                  d_state=16, d_conv=4, expand=2, channel_token=False)
        
    elif args.model == 'Model_MTAS1':  
        trainer.setting.network = Model_MTAS1(in_ch=1, out_ch=1,
                  list_ch_A=[-1, 16, 32, 64, 128, 256],
                  list_ch_B=[-1, 32, 64, 128, 256, 512],
                  d_state=16, d_conv=4, expand=2, channel_token=False)
        
    elif args.model == 'Model_MTASP':  
        trainer.setting.network = Model_MTASP(in_ch=9, out_ch=1,
                  list_ch_A=[-1, 16, 32, 64, 128, 256],
                  list_ch_B=[-1, 32, 64, 128, 256, 512],
                  d_state=16, d_conv=4, expand=2, channel_token=False)

    elif args.model == 'Model_Dense':
        trainer.setting.network = Model_Den(in_ch=9, out_ch=1,
                  list_ch_A=[-1, 16, 32, 64, 128, 256],
                  list_ch_B=[-1, 32, 64, 128, 256, 512],
                  d_state=16, d_conv=4, expand=2, channel_token=False)

    
        
    else:
        raise ValueError('Invalid model type specified. Choose "Model" or "MambaModel".')

    trainer.setting.network.cuda()
    y = trainer.setting.network(torch.rand(1, 9, 128, 128, 128).cuda())
    print('Param size = {:.3f} MB'.format(calc_param_size(trainer.setting.network)))


    trainer.setting.max_iter = args.max_iter

    trainer.setting.train_loader, trainer.setting.val_loader = get_loader(
        train_bs=args.batch_size,
        val_bs=1,
        train_num_samples_per_epoch=args.batch_size * 500,  # 500 iterations per epoch
        val_num_samples_per_epoch=1,
        num_works=4
    )

    trainer.setting.eps_train_loss = 0.01
    trainer.setting.lr_scheduler_update_on_iter = True

    if args.loss == 'Loss':
        trainer.setting.loss_function = Loss()
    elif args.loss == 'Loss_DC':
        trainer.setting.loss_function = Loss_DC()
    elif args.loss == 'AdvancedLoss':
        trainer.setting.loss_function = AdvancedLoss()
    elif args.loss == 'SharpDoseLoss':
        trainer.setting.loss_function = SharpDoseLoss()
    elif args.loss == 'Loss_DC_PTV':
        trainer.setting.loss_function = Loss_DC_PTV()


    trainer.setting.online_evaluation_function_val = online_evaluation

    trainer.set_optimizer(optimizer_type='Adam',
                          args={
                              'lr': 1e-4,
                              'weight_decay': 1e-4
                          }
                          )

    trainer.set_lr_scheduler(lr_scheduler_type='cosine',
                             args={
                                 'T_max': args.max_iter,
                                 'eta_min': 1e-7,
                                 'last_epoch': -1
                             }
                             )

    if not os.path.exists(trainer.setting.output_dir):
        os.mkdir(trainer.setting.output_dir)
    trainer.set_GPU_device(list_GPU_ids)
    trainer.run()

    trainer.print_log_to_file('# Done !\n', 'a')