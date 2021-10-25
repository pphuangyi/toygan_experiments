import os
from toygan import ROOT_OUTDIR, train

ROOT_OUTDIR = '/home/yhuang2/PROJs/LS4GAN/toygan_outdir/'

BATCH_SIZE = 64

args_dict = {
    'batch_size' : BATCH_SIZE,
    'data' : 'toyzero-preunaligned',
    'data_args'   : {
        # 'path'     : 'toyzero-1k',
        'path'     : '/home/yhuang2/data/LS4GAN/toy-adc/',
        'fname'    : 'test_1_n100-U-128x128.csv',
        'shuffle'  : False,
        'val_size' : 1000,
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
# args
    'label'  : None,
    'outdir' : os.path.join(
        ROOT_OUTDIR, 'default', 'toyzero-128-default-cyclegan'
    ),
    'log_level'  : 'DEBUG',
    'checkpoint' : 5,
}

train(args_dict)

