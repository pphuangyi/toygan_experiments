import os
import sys
sys.path.append('/home/yhuang2/PROJs/LS4GAN')

from toygan_hybrid_vit_cnn.toygan import train, train_debug

ROOT_OUTDIR = '/home/yhuang2/PROJs/LS4GAN/toygan_outdir/'

BATCH_SIZE = 64

base_model = sys.argv[1]

data_size = '50k'
data_folder = f'/home/yhuang2/data/LS4GAN/toyzero-128-{data_size}-precropped/'

args_dict = {
    'reinitialize': False,
    'save_at_zero_epoch': False,
    'batch_size' : BATCH_SIZE,
    'data' : 'toyzero-precropped',
    'data_args'   : {
        'path'        : data_folder,
        'align_train' : True,
        'align_val'   : True,
        'seed'        : 0,
    },
    'image_shape' : (1, 128, 128),
    'epochs'      : 200,
    'discriminator' : {
        'model' : 'basic',
        'model_args' : None,
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : 2e-4,
            'betas' : (0.5, 0.99),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
    },
    'generator' : {
        # 'model' : 'resnet_9blocks',
        'model' : 'unet_128',
        'model_args' : None,
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : 2e-4,
            'betas' : (0.5, 0.99),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
    },
    'model' : 'cyclegan',
    'model_args' : {
        'lambda_a'   : 10.0,
        'lambda_b'   : 10.0,
        'lambda_idt' : 0.5,
        'pool_size'  : 50,
    },
    'scheduler' : {
        'name'          : 'linear',
        'epochs_warmup' : 100,
        'epochs_anneal' : 100,
    },
    'loss'             : 'lsgan',
    'gradient_penalty' : None,
    'steps_per_epoch'  : 2000,
# transfer args
    'transfer' : {
        'base_model': base_model,
        'transfer_map'  : {
            'gen_ab' : 'encoder',
            'gen_ba' : 'encoder',
        },
        'strict'        : True,
        'allow_partial' : False,
    },
# args
    'label'  : 'pretrain-epoch-5',
    'outdir' : os.path.join(
        ROOT_OUTDIR,
	'toy-adc', 'bc', 'transfer', 'bc-cyclegan-transfer-8x8'
    ),
    'log_level'  : 'DEBUG',
    'checkpoint' : 5,
}

train_debug(args_dict)
