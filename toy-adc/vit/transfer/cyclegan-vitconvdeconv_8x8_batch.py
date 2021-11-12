import os
import sys
sys.path.append('/home/yhuang2/PROJs/LS4GAN')

from toygan_hybrid_vit_cnn.toygan import train_val

ROOT_OUTDIR = '/home/yhuang2/PROJs/LS4GAN/toygan_outdir/'

N_GPU = 1
BATCH_SIZE = 64 * N_GPU
patch_size = 8 # image patch random masking patch size
conv_norm = 'batch'
data_size = '50k'
data_folder = f'/data/datasets/LS4GAN/toyzero-128-{data_size}-precropped/'

args_dict = {
    'batch_size'  : BATCH_SIZE,
    'data' : 'toyzero-precropped',
    'data_args'   : {
        'path'     : data_folder,
        'align_train' : False,
        'align_val'   : True,
        'seed'        : 0,
    },
    'image_shape' : (1, 128, 128),
    'epochs'      : 500,
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
            'conv_norm'      : conv_norm,
            'token_size'     : (patch_size, patch_size),
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
    'model_args' : {
        'lambda_a'   : 1.0,
        'lambda_b'   : 1.0,
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
    'transfer' : {
        'base_model' : (
            f'{ROOT_OUTDIR}/toy-adc/vit/pretrain/'
            'model_d(toyzero-precropped)_md(None)_mg(vitconvdeconv)'
            f'_bert-{patch_size}x{patch_size}_{conv_norm}'
        ),
        'transfer_map'  : {
            'gen_ab' : 'encoder',
            'gen_ba' : 'encoder',
        },
        'strict'        : True,
        'allow_partial' : False,
    },
# args
    'label'      : f'cyclegan-{patch_size}x{patch_size}_{conv_norm}',
    'outdir'     : os.path.join(
        ROOT_OUTDIR,
        'toy-adc',
        'vit',
        'transfer'
    ),
    'log_level'  : 'DEBUG',
    'checkpoint' : 10,
}

train_val(args_dict)
