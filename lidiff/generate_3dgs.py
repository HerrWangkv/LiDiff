import click
from os.path import join, dirname, abspath
from os import environ, makedirs
import numpy as np
import torch
import yaml
import MinkowskiEngine as ME
from tqdm import tqdm

import lidiff.datasets.datasets as datasets
import lidiff.models.models as models
from lidiff.utils.gsplat_utils import export_ply
from lidiff.utils.gsplat_utils import to_attributes

def set_deterministic():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True


@click.command()
### Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'config/config_nuscenes.yaml'))
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default=None)
def main(config, weights):
    set_deterministic()

    cfg = yaml.safe_load(open(config))
    # overwrite the data path in case we have defined in the env variables
    if environ.get('TRAIN_DATABASE'):
        cfg['data']['data_dir'] = environ.get('TRAIN_DATABASE')
    #Load data and model
    assert weights is not None, "Please provide a weight file"
    # we load the current config file just to overwrite inference parameters to try different stuff during inference
    ckpt_cfg = yaml.safe_load(open(weights.split('checkpoints')[0] + '/hparams.yaml'))
    ckpt_cfg['train']['num_workers'] = cfg['train']['num_workers']
    ckpt_cfg['train']['n_gpus'] = 1#cfg['train']['n_gpus']
    ckpt_cfg['train']['batch_size'] = 1#cfg['train']['batch_size']
    ckpt_cfg['data']['data_dir'] = cfg['data']['data_dir']
    ckpt_cfg['diff']['s_steps'] = cfg['diff']['s_steps']
    ckpt_cfg['experiment']['id'] = cfg['experiment']['id']

    if 'dataset_norm' not in ckpt_cfg['data'].keys():
        ckpt_cfg['data']['dataset_norm'] = False
        ckpt_cfg['data']['std_axis_norm'] = False
    if 'max_range' not in ckpt_cfg['data'].keys():
        ckpt_cfg['data']['max_range'] = 10.

    cfg = ckpt_cfg

    model = models.DiffusionSplats.load_from_checkpoint(weights, hparams=cfg, map_location='cuda')
    model.to('cuda')
    print(model.hparams)

    dataloaders = datasets.dataloaders[cfg['data']['dataloader']](cfg)
    val_loader = iter(dataloaders.val_dataloader())
    batch = next(val_loader)
    for k in batch.keys():
        batch[k] = batch[k].cuda()

    gen_gs = model.generate_3dgs(batch)
    gt_gs = batch['splats']
    assert len(gen_gs) == 1 and len(gt_gs) == 1
    gen_gs = gen_gs[0]
    checkpoint_idx = weights.split('=')[1][:2]
    gs_dir = join(dirname(dirname(weights)), 'generated_3dgs', checkpoint_idx)
    makedirs(gs_dir, exist_ok=True)
    gs_path = join(gs_dir, "gen.ply")
    export_ply(gen_gs, gs_path)
    gt_gs = gt_gs[0]
    gt_gs[:,3:] = to_attributes(gt_gs[:,3:])
    gt_gs_path = join(gs_dir, "gt.ply")
    export_ply(gt_gs, gt_gs_path)

    
if __name__ == "__main__":
    main()
