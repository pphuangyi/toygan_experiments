import os
from toygan import ROOT_OUTDIR, train

# N_GPU = 1
# BATCH_SIZE = 32 * N_GPU

ROOT_OUTDIR = '/home/yhuang2/PROJs/LS4GAN/toygan_outdir/'

BATCH_SIZE = 128

args_dict = {
    'batch_size' : BATCH_SIZE,
    'data' : 'toyzero-presimple',
    'data_args'   : {
        'path'     : '/home/yhuang2/data/LS4GAN/toy-adc/',
        'fname'    : 'test_1_n100-U-128x128.csv',
        'shuffle'  : False,
        'val_size' : 1000,
    },
    'image_shape'   : (1, 128, 128),
    'epochs'        : 500,
    'discriminator' : None,
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
    'model'      : 'autoencoder',
    'model_args' : {
        'joint' : True,
        'background_penalty' : {
            'epochs_warmup' : 25,
            'epochs_anneal' : 75,
        },
        'masking' : {
            'name'       : 'image-patch-random',
            'patch_size' : (1, 1),
            'fraction'   : 0.4,
        },
    },
    'scheduler' : {
        'name'      : 'CosineAnnealingWarmRestarts',
        'T_0'       : 50,
        'T_mult'    : 1,
        'eta_min'   : BATCH_SIZE * 5e-8 / 512,
    },
    'loss'             : 'l2',
    'gradient_penalty' : None,
    'steps_per_epoch'  : 256 * 1024 // BATCH_SIZE,
# args
    'label'      : None,
    'outdir'     : os.path.join(
        ROOT_OUTDIR, 'cyclegan', 'pretrain', 'bc-pretrain-50k-1x1'
    ),
    'log_level'  : 'DEBUG',
    'checkpoint' : 5,
}

train(args_dict)

