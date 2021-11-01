mport os
from toygan import ROOT_OUTDIR, train

N_GPU = 1
BATCH_SIZE = 32 * N_GPU

args_dict = {
    'batch_size'  : BATCH_SIZE,
    'data' : 'toyzero-precropped',
    'data_args'   : {
        'path'     : 'test_1_n100-U-128x128',
        'align_train' : False,
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
        'model' : 'vit-hybrid',
        'model_args' : {
            'features'           : 384,
            'n_heads'            : 6,
            'n_blocks'           : 6,
            'ffn_features'       : 1536,
            'embed_features'     : 384,
            'activ'              : 'gelu',
            'norm'               : 'layer',
            'stem_features_list' : [48, 96, 192, 384],
            'stem_activ'         : 'leakyrelu',
            'stem_norm'          : None,
            'stem_downsample'    : 'conv',
            'stem_upsample'      : 'upsample-conv',
            'rezero'             : True,
        },
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : 1e-5 * BATCH_SIZE / 32,
            'betas' : (0, 0.99),
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
            'experiments/toyzero/vit-hybrid-bert-cyclegan'
            '/model_d(toyzero-precropped)_md(None)_mg(vit-hybrid)'
            '_bert-small-16x16_nonorm'
        ),
        'transfer_map'  : {
            'gen_ab' : 'encoder',
            'gen_ba' : 'encoder',
        },
        'strict'        : True,
        'allow_partial' : False,
    },
# args
    'label'      : 'cyclegan-small-16x16_nonorm',
    'outdir'     : os.path.join(
        ROOT_OUTDIR, 'experiments', 'toyzero', 'vit-hybrid-bert-cyclegan'
    ),
    'log_level'  : 'DEBUG',
    'checkpoint' : 10,
}

train(args_dict)
