import os
import sys
sys.path.append('/home/yhuang2/PROJs/LS4GAN')

from toygan_hybrid_vit_cnn.toygan import train_val

ROOT_OUTDIR = '/home/yhuang2/PROJs/LS4GAN/toygan_outdir/'

N_GPU = 1
BATCH_SIZE = 256 * N_GPU

args_dict = {
    'batch_size' : BATCH_SIZE,
    'data' : 'toyzero-precropped',
    'data_args'   : {
        'path'        : '/home/dtorbunov/shared/test_1_n100-U-128x128/',
        'align_train' : True,
        'align_val'   : True,
        'seed'        : 0,
    },
    'image_shape' : (1, 128, 128),
    'epochs'      : 500,
    'discriminator' : None,
    'generator' : {
        'model' : 'vitconvdeconv',
        'model_args' : {
            'features'       : 384,
            'n_resblocks'    : 3,
            'n_heads'        : 6,
            'n_blocks'       : 6,
            'ffn_features'   : 1536,
            'embed_features' : 384,
            'activ'          : 'gelu',
            'norm'           : 'layer',
            'token_size'     : (8, 8),
            'rescale'        : False,
            'rezero'         : True,
        },
        'optimizer'  : {
            'name'  : 'AdamW',
            'lr'    : BATCH_SIZE * 5e-5 / 512,
            'betas' : (0.9, 0.99),
            'weight_decay' : 0.05,
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
            'patch_size' : (8, 8),
            'fraction'   : 0.4
        },
    },
    'scheduler' : {
        'name'      : 'cosine',
        'T_max'     : 100,
        'eta_min'   : BATCH_SIZE * 5e-7 / 512,
    },
    'loss'             : 'l2',
    'gradient_penalty' : None,
    'steps_per_epoch'  : None,
# args
    'label'  : None,
    'outdir' : os.path.join(
        ROOT_OUTDIR,
        'toy-adc', 'vit', 'pretrain', 'vitconvdeconv'
    ),
    'log_level'  : 'DEBUG',
    'checkpoint' : 5,
}

train_val(args_dict)

