import click
from os.path import join, dirname, abspath
from os import environ, makedirs
import numpy as np
import torch
import yaml
import MinkowskiEngine as ME
from tqdm import tqdm
import time

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
def main(config):
    set_deterministic()

    cfg = yaml.safe_load(open(config))
    # overwrite the data path in case we have defined in the env variables
    if environ.get('TRAIN_DATABASE'):
        cfg['data']['data_dir'] = environ.get('TRAIN_DATABASE')
    
    cfg['train']['n_gpus'] = 1#cfg['train']['n_gpus']
    cfg['train']['batch_size'] = 1#cfg['train']['batch_size']

    model = models.DiffusionSplats(cfg)
    model.to('cuda')
    print(model.hparams)

    dataloaders = datasets.dataloaders[cfg['data']['dataloader']](cfg)
    val_loader = dataloaders.val_dataloader()
    val_iter = iter(val_loader)
    batch = next(val_iter)
    for k in batch.keys():
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].cuda()
    gs_dir = f'./noisy_gs/{time.strftime("%Y%m%d-%H%M%S")}'
    makedirs(gs_dir, exist_ok=True)
    batch_idx = int(batch['indices'][0])
    t_lst = list(range(model.t_steps-1, 0, -50))
    t_lst.append(0)
    for t in t_lst:
        noisy_gs = model.add_noise(batch, t)
        assert len(noisy_gs) == 1
        noisy_gs = noisy_gs[0]
        # gs_path = join(gs_dir, "gen.ply")
        render_path = join(gs_dir, f"gen_render_{t}.png")
        # export_ply(noisy_gs, gs_path)
        val_loader.dataset.splats.render(batch_idx, noisy_gs, render_path)
    noise = model.add_noise(batch)
    assert len(noise) == 1
    noise = noise[0]
    render_path = join(gs_dir, f"gen_render_noise.png")
    val_loader.dataset.splats.render(batch_idx, noise, render_path)
    
if __name__ == "__main__":
    main()
