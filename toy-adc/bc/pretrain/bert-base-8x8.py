import os
from toygan import ROOT_OUTDIR, train

N_GPU = 1
BATCH_SIZE = 32 * N_GPU

args_dict = {
    'batch_size' : BATCH_SIZE,
    'data' : 'toyzero-precropped',
    'data_args'   : {
        'path'     : '2021-09-16_n20_crops-U-128x128-toyzero-1k',
        'align_train' : True,
        'align_val'   : True,
        'seed'        : 0,
    },
    'image_shape'   : (1, 128, 128),
    'epochs'        : 500,
    'discriminator' : None,
    'generator'     : {
        'model' : 'vit-v0',
        'model_args' : {
            'features'       : 768,
            'n_heads'        : 12,
            'n_blocks'       : 12,
            'ffn_features'   : 3072,
            'embed_features' : 768,
            'activ'          : 'gelu',
            'norm'           : 'layer',
            'token_size'     : (8, 8),
            'rescale'        : False,
            'rezero'         : True,
        },
        'optimizer'  : {
            'name'  : 'AdamW',
            'lr'    : BATCH_SIZE * 5e-4 / 512,
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
    'label'      : 'bert-base-8x8',
    'outdir'     : os.path.join(
        ROOT_OUTDIR, 'experiments', 'toyzero', 'vit-bert-cyclegan'
    ),
    'log_level'  : 'DEBUG',
    'checkpoint' : 10,
}

train(args_dict)

