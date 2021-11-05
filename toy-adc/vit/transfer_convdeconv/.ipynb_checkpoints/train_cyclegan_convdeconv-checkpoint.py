import os
import sys
sys.path.append('/home/yhuang2/PROJs/LS4GAN')

from toygan_convdeconv.cyclegan import train

ROOT_OUTDIR = '/home/yhuang2/PROJs/LS4GAN/toygan_outdir/'

BATCH_SIZE = 64

base_model = sys.argv[1]

data_size = '50k'
data_folder = f'/data/datasets/LS4GAN/toyzero-128-{data_size}-precropped/'

args_dict = {
    'batch_size' : BATCH_SIZE,
    'data' : 'toyzero-precropped',
    'data_args'   : {
        'path'        : data_folder,
        'align_train' : True,
        'align_val'   : True,
        'seed'        : 0,
    },
    'image_shape' : (1, 128, 128),
    'epochs'      : 300,
    'discriminator' : {
        'model' : 'basic',
        'model_args' : None,
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : 1e-4 * BATCH_SIZE / 32,
            'betas' : (0, 0.99),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
    },
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
            'masking'        : {
                'name'     : 'transformer-random',
                'fraction' : 0.4,
            },
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
    'model' : 'cyclegan',
    'model_args' : {
        'lambda_a'   : 1.0, # useing a much smaller coefficient cycle loss
        'lambda_b'   : 1.0, # useing a much smaller coefficient cycle loss
        'lambda_idt' : 0.5,
        'pool_size'  : 50,
    },
    'scheduler' : {
        'name'          : 'linear',
        'epochs_warmup' : 100,
        'epochs_anneal' : 200,
    },
    'loss'             : 'wgan',
    'gradient_penalty' : { 'lambda_gp' : 1 },
    'steps_per_epoch'  : 32 * 1024 // BATCH_SIZE,
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
    'label'  : None,
    'outdir' : os.path.join(
        ROOT_OUTDIR,
        'toy-adc',
        'vit',
        'transfer_convdeconv',
        'vit-cyclegan-transfer-convdeconv'
    ),
    'log_level'  : 'DEBUG',
    'checkpoint' : 10,
}

train(args_dict)
