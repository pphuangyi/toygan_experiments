import os
from cyclegan import ROOT_OUTDIR, train, train_val, join_dicts

ROOT_OUTDIR = '/home/yhuang2/PROJs/LS4GAN/toygan_outdir/'

BATCH_SIZE = 128

args_dict = {
    'batch_size' : BATCH_SIZE,
    'data' : 'toyzero-presimple',
    'data_args'   : {
        # 'path'     : 'toyzero-1k',
        'path'     : '/home/yhuang2/data/LS4GAN/toy-adc/',
        'fname'    : 'test_1_n100-U-128x128.csv',
        'shuffle'  : False,
        'val_size' : 5000,
    },
    'image_shape' : (1, 128, 128),
    'epochs'      : 1000,
    'discriminator' : { 'model' : None, },
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
    # 'generator' : {
    #     'model' : 'vitconv',
    #     'model_args' : {
    #         'features'       : 384,
    #         'n_resblocks'    : 3,
    #         'n_heads'        : 6,
    #         'n_blocks'       : 6,
    #         'ffn_features'   : 1536,
    #         'embed_features' : 384,
    #         'activ'          : 'gelu',
    #         'norm'           : 'layer',
    #         'token_size'     : (8, 8),
    #         'rescale'        : False,
    #         'rezero'         : True,
    #         'masking'        : {
    #             'name'     : 'transformer-random',
    #             'fraction' : 0.4,
    #         },
    #     },
    #     'optimizer'  : {
    #         'name'  : 'AdamW',
    #         'lr'    : BATCH_SIZE * 5e-4 / 512,
    #         'betas' : (0.9, 0.99),
    #         'weight_decay' : 0.05,
    #     },
    #     'weight_init' : {
    #         'name'      : 'normal',
    #         'init_gain' : 0.02,
    #     },
    # },
    'model'      : 'autoencoder',
    'model_args' : {
        'joint' : True,
        'background_penalty' : {
            'epochs_warmup' : 100,
            'epochs_anneal' : 400,
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
        ROOT_OUTDIR, 'default', 'autoencoder', 'toyzero-128-bert'
    ),
    'log_level'  : 'DEBUG',
    'checkpoint' : 5,
}

train(args_dict)

