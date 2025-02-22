import torch
import torch.nn as nn
import torch.nn.functional as F
import lidiff.models.minkunet as minknet
import numpy as np
import MinkowskiEngine as ME
import open3d as o3d
from lidiff.utils.scheduling import beta_func
from tqdm import tqdm
from os import makedirs, path
import wandb

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import LightningDataModule
from lidiff.utils.collations import *
from lidiff.utils.metrics import ChamferDistance, PrecisionRecall
from lidiff.utils.gsplat_utils import to_attributes
from lidiff.models.ptv3 import Point, PointTransformerV3
from diffusers import DPMSolverMultistepScheduler

class DiffusionPoints(LightningModule):
    def __init__(self, hparams:dict, data_module: LightningDataModule = None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module

        # alphas and betas
        if self.hparams['diff']['beta_func'] == 'cosine':
            self.betas = beta_func[self.hparams['diff']['beta_func']](self.hparams['diff']['t_steps'])
        else:
            self.betas = beta_func[self.hparams['diff']['beta_func']](
                    self.hparams['diff']['t_steps'],
                    self.hparams['diff']['beta_start'],
                    self.hparams['diff']['beta_end'],
            )

        self.t_steps = self.hparams['diff']['t_steps']
        self.s_steps = self.hparams['diff']['s_steps']
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.tensor(
            np.cumprod(self.alphas, axis=0), dtype=torch.float32, device=torch.device('cuda')
        )

        self.alphas_cumprod_prev = torch.tensor(
            np.append(1., self.alphas_cumprod[:-1].cpu().numpy()), dtype=torch.float32, device=torch.device('cuda')
        )

        self.betas = torch.tensor(self.betas, device=torch.device('cuda'))
        self.alphas = torch.tensor(self.alphas, device=torch.device('cuda'))

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod) 
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1.)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.sqrt_posterior_variance = torch.sqrt(self.posterior_variance)
        self.posterior_log_var = torch.log(
            torch.max(self.posterior_variance, 1e-20 * torch.ones_like(self.posterior_variance))
        )

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        
        # for fast sampling
        self.dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.t_steps,
                beta_start=self.hparams['diff']['beta_start'],
                beta_end=self.hparams['diff']['beta_end'],
                beta_schedule='linear',
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.dpm_scheduler.set_timesteps(self.s_steps)
        self.scheduler_to_cuda()

        self.partial_enc = minknet.MinkGlobalEnc(in_channels=3, out_channels=self.hparams['model']['out_dim'])
        self.model = minknet.MinkUNetDiff(in_channels=3, out_channels=self.hparams['model']['out_dim'])

        self.chamfer_distance = ChamferDistance()
        self.precision_recall = PrecisionRecall(self.hparams['data']['resolution'],2*self.hparams['data']['resolution'],100)

        self.w_uncond = self.hparams['train']['uncond_w']

    def scheduler_to_cuda(self):
        self.dpm_scheduler.timesteps = self.dpm_scheduler.timesteps.cuda()
        self.dpm_scheduler.betas = self.dpm_scheduler.betas.cuda()
        self.dpm_scheduler.alphas = self.dpm_scheduler.alphas.cuda()
        self.dpm_scheduler.alphas_cumprod = self.dpm_scheduler.alphas_cumprod.cuda()
        self.dpm_scheduler.alpha_t = self.dpm_scheduler.alpha_t.cuda()
        self.dpm_scheduler.sigma_t = self.dpm_scheduler.sigma_t.cuda()
        self.dpm_scheduler.lambda_t = self.dpm_scheduler.lambda_t.cuda()
        self.dpm_scheduler.sigmas = self.dpm_scheduler.sigmas.cuda()

    def q_sample(self, x, t, noise):
        return self.sqrt_alphas_cumprod[t][:,None,None].cuda() * x + \
                self.sqrt_one_minus_alphas_cumprod[t][:,None,None].cuda() * noise

    def classfree_forward(self, x_t, x_cond, x_uncond, t):
        x_t_sparse = x_t.sparse()
        x_cond = self.forward(x_t, x_t_sparse, x_cond, t)            
        x_uncond = self.forward(x_t, x_t_sparse, x_uncond, t)

        return x_uncond + self.w_uncond * (x_cond - x_uncond)

    def visualize_step_t(self, x_t, gt_pts, pcd, pcd_mean, pcd_std, pidx=0):
        points = x_t.F.detach().cpu().numpy()
        points = points.reshape(gt_pts.shape[0],-1,3)
        obj_mean = pcd_mean[pidx][0].detach().cpu().numpy()
        points = np.concatenate((points[pidx], gt_pts[pidx]), axis=0)

        dist_pts = np.sqrt(np.sum((points - obj_mean)**2, axis=-1))
        dist_idx = dist_pts < self.hparams['data']['max_range']

        full_pcd = len(points) - len(gt_pts[pidx])
        print(f'\n[{dist_idx.sum() - full_pcd}|{dist_idx.shape[0] - full_pcd }] points inside margin...')

        pcd.points = o3d.utility.Vector3dVector(points[dist_idx])
       
        colors = np.ones((len(points), 3)) * .5
        colors[:len(gt_pts[0])] = [1.,.3,.3]
        colors[-len(gt_pts[0]):] = [.3,1.,.3]
        pcd.colors = o3d.utility.Vector3dVector(colors[dist_idx])

    def reset_partial_pcd(self, x_part, x_uncond, x_mean, x_std):
        x_part = self.points_to_tensor(x_part.F.reshape(x_mean.shape[0],-1,3).detach(), x_mean, x_std)
        x_uncond = self.points_to_tensor(
                torch.zeros_like(x_part.F.reshape(x_mean.shape[0],-1,3)), torch.zeros_like(x_mean), torch.zeros_like(x_std)
        )

        return x_part, x_uncond

    def p_sample_loop(self, x_init, x_t, x_cond, x_uncond, gt_pts, x_mean, x_std):
        pcd = o3d.geometry.PointCloud()
        self.scheduler_to_cuda()

        for t in tqdm(range(len(self.dpm_scheduler.timesteps))):
            t = torch.ones(gt_pts.shape[0]).cuda().long() * self.dpm_scheduler.timesteps[t].cuda()
            
            noise_t = self.classfree_forward(x_t, x_cond, x_uncond, t)
            input_noise = x_t.F.reshape(t.shape[0],-1,3) - x_init
            x_t = x_init + self.dpm_scheduler.step(noise_t, t[0], input_noise)['prev_sample']
            x_t = self.points_to_tensor(x_t, x_mean, x_std)

            # this is needed otherwise minkEngine will keep "stacking" coords maps over the x_part and x_uncond
            # i.e. memory leak
            x_cond, x_uncond = self.reset_partial_pcd(x_cond, x_uncond, x_mean, x_std)
            torch.cuda.empty_cache()

        makedirs(f'{self.logger.log_dir}/generated_pcd/', exist_ok=True)

        return x_t

    def p_losses(self, y, noise):
        return F.mse_loss(y, noise)

    def forward(self, x_full, x_full_sparse, x_part, t):
        part_feat = self.partial_enc(x_part)
        out = self.model(x_full, x_full_sparse, part_feat, t)
        torch.cuda.empty_cache()
        return out.reshape(t.shape[0],-1,3)

    def points_to_tensor(self, x_feats, mean, std):
        x_feats = ME.utils.batched_coordinates(list(x_feats[:]), dtype=torch.float32, device=self.device)

        x_coord = x_feats.clone()
        x_coord[:,1:] = feats_to_coord(x_feats[:,1:], self.hparams['data']['resolution'], mean, std)

        x_t = ME.TensorField(
            features=x_feats[:,1:],
            coordinates=x_coord,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

        torch.cuda.empty_cache()

        return x_t

    def training_step(self, batch:dict, batch_idx):
        # initial random noise
        torch.cuda.empty_cache()
        noise = torch.randn(batch['pcd_full'].shape, device=self.device)
        
        # sample step t
        t = torch.randint(0, self.t_steps, size=(batch['pcd_full'].shape[0],)).cuda()
        # sample q at step t
        # we sample noise towards zero to then add to each point the noise (without normalizing the pcd)
        t_sample = batch['pcd_full'] + self.q_sample(torch.zeros_like(batch['pcd_full']), t, noise)

        # replace the original points with the noise sampled
        x_full = self.points_to_tensor(t_sample, batch['mean'], batch['std'])

        # for classifier-free guidance switch between conditional and unconditional training
        if torch.rand(1) > self.hparams['train']['uncond_prob'] or batch['pcd_full'].shape[0] == 1:
            x_part = self.points_to_tensor(batch['pcd_part'], batch['mean'], batch['std'])
        else:
            x_part = self.points_to_tensor(
                torch.zeros_like(batch['pcd_part']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
            )

        denoise_t = self.forward(x_full, x_full.sparse(), x_part, t)
        loss_mse = self.p_losses(denoise_t, noise)
        loss_mean = (denoise_t.mean())**2
        loss_std = (denoise_t.std() - 1.)**2
        loss = loss_mse + self.hparams['diff']['reg_weight'] * (loss_mean + loss_std)

        std_noise = (denoise_t - noise)**2
        self.log('train/loss_mse', loss_mse)
        self.log('train/loss_mean', loss_mean)
        self.log('train/loss_std', loss_std)
        self.log('train/loss', loss)
        self.log('train/var', std_noise.var())
        self.log('train/std', std_noise.std())
        torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch:dict, batch_idx):
        if batch_idx != 0:
            return

        self.model.eval()
        self.partial_enc.eval()
        with torch.no_grad():
            gt_pts = batch['pcd_full'].detach().cpu().numpy()

            # for inference we get the partial pcd and sample the noise around the partial
            x_init = batch['pcd_part'].repeat(1,10,1)
            x_feats = x_init + torch.randn(x_init.shape, device=self.device)
            x_full = self.points_to_tensor(x_feats, batch['mean'], batch['std'])
            x_part = self.points_to_tensor(batch['pcd_part'], batch['mean'], batch['std'])
            x_uncond = self.points_to_tensor(
                torch.zeros_like(batch['pcd_part']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
            )

            x_gen_eval = self.p_sample_loop(x_init, x_full, x_part, x_uncond, gt_pts, batch['mean'], batch['std'])
            x_gen_eval = x_gen_eval.F.reshape((gt_pts.shape[0],-1,3))

            for i in range(len(batch['pcd_full'])):
                pcd_pred = o3d.geometry.PointCloud()
                c_pred = x_gen_eval[i].cpu().detach().numpy()
                pcd_pred.points = o3d.utility.Vector3dVector(c_pred)

                pcd_gt = o3d.geometry.PointCloud()
                g_pred = batch['pcd_full'][i].cpu().detach().numpy()
                pcd_gt.points = o3d.utility.Vector3dVector(g_pred)

                self.chamfer_distance.update(pcd_gt, pcd_pred)
                self.precision_recall.update(pcd_gt, pcd_pred)

        cd_mean, cd_std = self.chamfer_distance.compute()
        pr, re, f1 = self.precision_recall.compute_auc()

        self.log('val/cd_mean', cd_mean, on_step=True)
        self.log('val/cd_std', cd_std, on_step=True)
        self.log('val/precision', pr, on_step=True)
        self.log('val/recall', re, on_step=True)
        self.log('val/fscore', f1, on_step=True)
        torch.cuda.empty_cache()

        return {'val/cd_mean': cd_mean, 'val/cd_std': cd_std, 'val/precision': pr, 'val/recall': re, 'val/fscore': f1}
    
    def valid_paths(self, filenames):
        output_paths = []
        skip = []

        for fname in filenames:
            seq_dir =  f'{self.logger.log_dir}/generated_pcd/{fname.split("/")[-3]}'
            ply_name = f'{fname.split("/")[-1].split(".")[0]}.ply'

            skip.append(path.isfile(f'{seq_dir}/{ply_name}'))
            makedirs(seq_dir, exist_ok=True)
            output_paths.append(f'{seq_dir}/{ply_name}')

        return np.all(skip), output_paths

    def test_step(self, batch:dict, batch_idx):
        self.model.eval()
        self.partial_enc.eval()
        with torch.no_grad():
            skip, output_paths = self.valid_paths(batch['filename'])

            if skip:
                print(f'Skipping generation from {output_paths[0]} to {output_paths[-1]}') 
                return {'test/cd_mean': 0., 'test/cd_std': 0., 'test/precision': 0., 'test/recall': 0., 'test/fscore': 0.}

            gt_pts = batch['pcd_full'].detach().cpu().numpy()

            x_init = batch['pcd_part'].repeat(1,10,1)
            x_feats = x_init + torch.randn(x_init.shape, device=self.device)
            x_full = self.points_to_tensor(x_feats, batch['mean'], batch['std'])
            x_part = self.points_to_tensor(batch['pcd_part'], batch['mean'], batch['std'])
            x_uncond = self.points_to_tensor(
                torch.zeros_like(batch['pcd_part']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
            )

            x_gen_eval = self.p_sample_loop(x_init, x_full, x_part, x_uncond, gt_pts, batch['mean'], batch['std'])
            x_gen_eval = x_gen_eval.F.reshape((gt_pts.shape[0],-1,3))

            for i in range(len(batch['pcd_full'])):
                pcd_pred = o3d.geometry.PointCloud()
                c_pred = x_gen_eval[i].cpu().detach().numpy()
                dist_pts = np.sqrt(np.sum((c_pred)**2, axis=-1))
                dist_idx = dist_pts < self.hparams['data']['max_range']
                points = c_pred[dist_idx]
                max_z = x_init[i][...,2].max().item()
                min_z = (x_init[i][...,2].mean() - 2 * x_init[i][...,2].std()).item()
                pcd_pred.points = o3d.utility.Vector3dVector(points[(points[:,2] < max_z) & (points[:,2] > min_z)])
                pcd_pred.paint_uniform_color([1.0, 0.,0.])

                pcd_gt = o3d.geometry.PointCloud()
                g_pred = batch['pcd_full'][i].cpu().detach().numpy()
                pcd_gt.points = o3d.utility.Vector3dVector(g_pred)
                pcd_gt.paint_uniform_color([0., 1.,0.])
                
                print(f'Saving {output_paths[i]}')
                o3d.io.write_point_cloud(f'{output_paths[i]}', pcd_pred)

                self.chamfer_distance.update(pcd_gt, pcd_pred)
                self.precision_recall.update(pcd_gt, pcd_pred)

        cd_mean, cd_std = self.chamfer_distance.compute()
        pr, re, f1 = self.precision_recall.compute_auc()
        print(f'CD Mean: {cd_mean}\tCD Std: {cd_std}')
        print(f'Precision: {pr}\tRecall: {re}\tF-Score: {f1}')

        self.log('test/cd_mean', cd_mean, on_step=True)
        self.log('test/cd_std', cd_std, on_step=True)
        self.log('test/precision', pr, on_step=True)
        self.log('test/recall', re, on_step=True)
        self.log('test/fscore', f1, on_step=True)
        torch.cuda.empty_cache()

        return {'test/cd_mean': cd_mean, 'test/cd_std': cd_std, 'test/precision': pr, 'test/recall': re, 'test/fscore': f1}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['train']['lr'], betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
        scheduler = {
            'scheduler': scheduler, # lr * 0.5
            'interval': 'epoch', # interval is epoch-wise
            'frequency': 5, # after 5 epochs
        }

        return [optimizer], [scheduler]

class DiffusionParams:
    def __init__(self, betas:torch.Tensor, t_steps:int):
        self.betas = betas
        self.t_steps = t_steps
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.tensor(
            np.cumprod(self.alphas, axis=0), dtype=torch.float32, device=torch.device('cuda')
        )

        self.alphas_cumprod_prev = torch.tensor(
            np.append(1., self.alphas_cumprod[:-1].cpu().numpy()), dtype=torch.float32, device=torch.device('cuda')
        )

        self.betas = torch.tensor(self.betas, device=torch.device('cuda'))
        self.alphas = torch.tensor(self.alphas, device=torch.device('cuda'))

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod) 
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1.)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.sqrt_posterior_variance = torch.sqrt(self.posterior_variance)
        self.posterior_log_var = torch.log(
            torch.max(self.posterior_variance, 1e-20 * torch.ones_like(self.posterior_variance))
        )

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        
        # self.sampling_coef1_xyz = (1 - self.alphas) * self.sqrt_one_minus_alphas_cumprod / (self.betas + (1. - self.alphas_cumprod_prev))
        # self.sampling_coef2_xyz = torch.sqrt(self.betas * (1. - self.alphas_cumprod_prev) / (self.betas + (1. - self.alphas_cumprod_prev)))
        self.sampling_coef_others = (1 - self.alphas) / torch.sqrt(1 - self.alphas_cumprod)
        

class DiffusionSplats(LightningModule):
    def __init__(self, hparams:dict, data_module: LightningDataModule = None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module

        # alphas and betas
        self.color_betas = beta_func[self.hparams['diff']['color']['beta_func']](self.hparams['diff']['t_steps']) if self.hparams['diff']['color']['beta_func'] == 'cosine' else beta_func[self.hparams['diff']['color']['beta_func']](
                self.hparams['diff']['t_steps'],
                self.hparams['diff']['color']['beta_start'],
                self.hparams['diff']['color']['beta_end'],
        )
        self.color_params = DiffusionParams(self.color_betas, self.hparams['diff']['t_steps'])

        self.opacity_betas = beta_func[self.hparams['diff']['opacity']['beta_func']](self.hparams['diff']['t_steps']) if self.hparams['diff']['opacity']['beta_func'] == 'cosine' else beta_func[self.hparams['diff']['opacity']['beta_func']](
                self.hparams['diff']['t_steps'],
                self.hparams['diff']['opacity']['beta_start'],
                self.hparams['diff']['opacity']['beta_end'],
        )
        self.opacity_params = DiffusionParams(self.opacity_betas, self.hparams['diff']['t_steps'])

        self.scale_betas = beta_func[self.hparams['diff']['scale']['beta_func']](self.hparams['diff']['t_steps']) if self.hparams['diff']['scale']['beta_func'] == 'cosine' else beta_func[self.hparams['diff']['scale']['beta_func']](
                self.hparams['diff']['t_steps'],
                self.hparams['diff']['scale']['beta_start'],
                self.hparams['diff']['scale']['beta_end'],
        )
        self.scale_params = DiffusionParams(self.scale_betas, self.hparams['diff']['t_steps'])

        self.quat0_betas = beta_func[self.hparams['diff']['quat_0']['beta_func']](self.hparams['diff']['t_steps']) if self.hparams['diff']['quat_0']['beta_func'] == 'cosine' else beta_func[self.hparams['diff']['quat_0']['beta_func']](
                self.hparams['diff']['t_steps'],
                self.hparams['diff']['quat_0']['beta_start'],
                self.hparams['diff']['quat_0']['beta_end'],
        )
        self.quat0_params = DiffusionParams(self.quat0_betas, self.hparams['diff']['t_steps'])

        self.quatr_betas = beta_func[self.hparams['diff']['quat_rest']['beta_func']](self.hparams['diff']['t_steps']) if self.hparams['diff']['quat_rest']['beta_func'] == 'cosine' else beta_func[self.hparams['diff']['quat_rest']['beta_func']](
                self.hparams['diff']['t_steps'],
                self.hparams['diff']['quat_rest']['beta_start'],
                self.hparams['diff']['quat_rest']['beta_end'],
        )
        self.quatr_params = DiffusionParams(self.quatr_betas, self.hparams['diff']['t_steps'])

        self.t_steps = self.hparams['diff']['t_steps']
        self.s_steps = self.hparams['diff']['s_steps']

        # for fast sampling
        self.color_dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.t_steps,
                beta_start=self.hparams['diff']['color']['beta_start'],
                beta_end=self.hparams['diff']['color']['beta_end'],
                beta_schedule=self.hparams['diff']['color']['beta_func'],
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.color_dpm_scheduler.set_timesteps(self.s_steps)

        self.opacity_dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.t_steps,
                beta_start=self.hparams['diff']['opacity']['beta_start'],
                beta_end=self.hparams['diff']['opacity']['beta_end'],
                beta_schedule=self.hparams['diff']['opacity']['beta_func'],
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.opacity_dpm_scheduler.set_timesteps(self.s_steps)

        self.scale_dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.t_steps,
                beta_start=self.hparams['diff']['scale']['beta_start'],
                beta_end=self.hparams['diff']['scale']['beta_end'],
                beta_schedule=self.hparams['diff']['scale']['beta_func'],
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.scale_dpm_scheduler.set_timesteps(self.s_steps)

        self.quat0_dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.t_steps,
                beta_start=self.hparams['diff']['quat_0']['beta_start'],
                beta_end=self.hparams['diff']['quat_0']['beta_end'],
                beta_schedule=self.hparams['diff']['quat_0']['beta_func'],
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.quat0_dpm_scheduler.set_timesteps(self.s_steps)

        self.quatr_dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.t_steps,
                beta_start=self.hparams['diff']['quat_rest']['beta_start'],
                beta_end=self.hparams['diff']['quat_rest']['beta_end'],
                beta_schedule=self.hparams['diff']['quat_rest']['beta_func'],
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.quatr_dpm_scheduler.set_timesteps(self.s_steps)
        self.scheduler_to_cuda()

        # self.lidar_enc = minknet.MinkGlobalEnc(in_channels=3)
        self.model = minknet.MinkUNetDiff(in_channels=self.hparams['model']['in_dim'], out_channels=self.hparams['model']['out_dim'])

        self.chamfer_distance = ChamferDistance()
        self.precision_recall = PrecisionRecall(self.hparams['data']['resolution'],2*self.hparams['data']['resolution'],100)

        self.w_uncond = self.hparams['train']['uncond_w']

    def scheduler_to_cuda(self):
        for scheduler in [self.color_dpm_scheduler, self.opacity_dpm_scheduler, self.scale_dpm_scheduler, self.quat0_dpm_scheduler, self.quatr_dpm_scheduler]:
            scheduler.timesteps = scheduler.timesteps.cuda()
            scheduler.betas = scheduler.betas.cuda()
            scheduler.alphas = scheduler.alphas.cuda()
            scheduler.alphas_cumprod = scheduler.alphas_cumprod.cuda()
            scheduler.alpha_t = scheduler.alpha_t.cuda()
            scheduler.sigma_t = scheduler.sigma_t.cuda()
            scheduler.lambda_t = scheduler.lambda_t.cuda()
            scheduler.sigmas = scheduler.sigmas.cuda()

    def q_sample(self, x, t, param, noise):
        return param.sqrt_alphas_cumprod[t.cpu()][:,None,None].cuda() * x + \
                param.sqrt_one_minus_alphas_cumprod[t.cpu()][:,None,None].cuda() * noise

    def inverse_q_sample(self, xt, t, param, x0):
        return (xt - param.sqrt_alphas_cumprod[t.cpu()][:,None,None].cuda() * x0) / param.sqrt_one_minus_alphas_cumprod[t.cpu()][:,None,None].cuda()
    
    def classfree_forward(self, x_t, x_cond, x_uncond, t):
        x_t_sparse = x_t.sparse()
        x_cond = self.forward(x_t, x_t_sparse, x_cond, t)            
        x_uncond = self.forward(x_t, x_t_sparse, x_uncond, t)

        return x_uncond + self.w_uncond * (x_cond - x_uncond)

    def reset_partial_pcd(self, x_part, x_uncond, x_mean, x_std):
        x_part = self.points_to_tensor(x_part.F.reshape(x_mean.shape[0],-1,3).detach(), x_mean, x_std)
        x_uncond = self.points_to_tensor(
                torch.zeros_like(x_part.F.reshape(x_mean.shape[0],-1,3)), torch.zeros_like(x_mean), torch.zeros_like(x_std)
        )

        return x_part, x_uncond
    
    def get_noise(self, x_t, t):
        x0 = self.forward(x_t, x_t.sparse(), None, t)
        xt = x_t.F.reshape(t.shape[0],-1,self.hparams['model']['in_dim'])
        noise_color = self.inverse_q_sample(xt[:,:,3:6], t, self.color_params, x0[:,:,:3])
        noise_opacity = self.inverse_q_sample(xt[:,:,6:7], t, self.opacity_params, x0[:,:,3:4])
        noise_scale = self.inverse_q_sample(xt[:,:,7:10], t, self.scale_params, x0[:,:,4:7])
        noise_quat0 = self.inverse_q_sample(xt[:,:,10:11], t, self.quat0_params, x0[:,:,7:8])
        noise_quatr = self.inverse_q_sample(xt[:,:,11:], t, self.quatr_params, x0[:,:,8:])
        return torch.cat((noise_color, noise_opacity, noise_scale, noise_quat0, noise_quatr), dim=-1)
     
    # def p_sample_step(self, x_t, t):
    #     noise_t = self.get_noise(x_t, t)#self.classfree_forward(x_t, x_cond, x_uncond, t)
    #     features = x_t.F.reshape(t.shape[0],-1,self.hparams['model']['in_dim'])
    #     # xyz = features[:,:,:3]
    #     # others = features[:,:,3:]
    #     # x_tm1_xyz = xyz - self.sampling_coef1_xyz[t.cpu()] * noise_t[:,:,:3] + self.sampling_coef2_xyz[t.cpu()] * torch.randn_like(xyz)
    #     # x_tm1_others = self.sqrt_recip_alphas[t.cpu()] * (others - self.sampling_coef_others[t.cpu()] * noise_t[:,:,3:]) + self.sqrt_posterior_variance[t.cpu()] * torch.randn_like(others)
    #     # x_tm1 = torch.cat((x_tm1_xyz, x_tm1_others), dim=-1)
    #     colors = features[:,:,3:6]
    #     colors = self.color_params.sqrt_recip_alphas[t.cpu()] * (colors - self.color_params.sampling_coef_others[t.cpu()] * noise_t[:,:,:3]) + self.color_params.sqrt_posterior_variance[t.cpu()] * torch.randn_like(colors)
    #     opacity = features[:,:,6:7]
    #     # opacity = self.opacity_params.sqrt_recip_alphas[t.cpu()] * (opacity - self.opacity_params.sampling_coef_others[t.cpu()] * noise_t[:,:,3:4]) + self.opacity_params.sqrt_posterior_variance[t.cpu()] * torch.randn_like(opacity)
    #     scale = features[:,:,7:10]
    #     # scale = self.scale_params.sqrt_recip_alphas[t.cpu()] * (scale - self.scale_params.sampling_coef_others[t.cpu()] * noise_t[:,:,4:7]) + self.scale_params.sqrt_posterior_variance[t.cpu()] * torch.randn_like(scale)
    #     quat0 = features[:,:,10:11]
    #     # quat0 = self.quat0_params.sqrt_recip_alphas[t.cpu()] * (quat0 - self.quat0_params.sampling_coef_others[t.cpu()] * noise_t[:,:,7:8]) + self.quat0_params.sqrt_posterior_variance[t.cpu()] * torch.randn_like(quat0)
    #     quatr = features[:,:,11:]
    #     # quatr = self.quatr_params.sqrt_recip_alphas[t.cpu()] * (quatr - self.quatr_params.sampling_coef_others[t.cpu()] * noise_t[:,:,8:]) + self.quatr_params.sqrt_posterior_variance[t.cpu()] * torch.randn_like(quatr)
    #     x_tm1 = torch.cat((features[:,:,:3], colors, opacity, scale, quat0, quatr), dim=-1)
    #     return x_tm1

    # def p_sample_loop(self, x_init, x_t, gt_pts, x_mean, x_std):
    #     for t in tqdm(range(self.t_steps - 1,0,-1)):
    #         t_ = torch.ones(gt_pts.shape[0]).cuda().long() * t
    #         x_t = self.p_sample_step(x_t, t_)
    #         x_t = self.points_to_tensor(x_t, x_mean, x_std)

    #         # this is needed otherwise minkEngine will keep "stacking" coords maps over the x_part and x_uncond
    #         # i.e. memory leak
    #         # x_cond, x_uncond = self.reset_partial_pcd(x_cond, x_uncond, x_mean, x_std)
    #         torch.cuda.empty_cache()
    #     return x_t

    def p_sample_loop(self, x_init, x_t, gt_pts, x_mean, x_std):
        self.scheduler_to_cuda()

        for t in tqdm(range(len(self.color_dpm_scheduler.timesteps))):
            t = torch.ones(gt_pts.shape[0]).cuda().long() * self.color_dpm_scheduler.timesteps[t].cuda()
            
            noise_t = self.get_noise(x_t, t) #self.forward(x_t, x_t.sparse(), None, t)
            input_noise = x_t.F.reshape(t.shape[0],-1,self.hparams['model']['in_dim'])[:,:,3:3+self.hparams['model']['out_dim']] - x_init[:,:,3:3+self.hparams['model']['out_dim']]
            x_t = torch.tensor(x_init)
            x_t[:,:,3:6] += self.color_dpm_scheduler.step(noise_t[:,:,:3], t[0], input_noise[:,:,:3])['prev_sample']
            x_t[:,:,6:7] += self.opacity_dpm_scheduler.step(noise_t[:,:,3:4], t[0], input_noise[:,:,3:4])['prev_sample']
            x_t[:,:,7:10] += self.scale_dpm_scheduler.step(noise_t[:,:,4:7], t[0], input_noise[:,:,4:7])['prev_sample']
            x_t[:,:,10:11] += self.quat0_dpm_scheduler.step(noise_t[:,:,7:8], t[0], input_noise[:,:,7:8])['prev_sample']
            x_t[:,:,11:] += self.quatr_dpm_scheduler.step(noise_t[:,:,8:], t[0], input_noise[:,:,8:])['prev_sample']
            x_t = self.points_to_tensor(x_t, x_mean, x_std)

            # this is needed otherwise minkEngine will keep "stacking" coords maps over the x_part and x_uncond
            # i.e. memory leak
            # x_cond, x_uncond = self.reset_partial_pcd(x_cond, x_uncond, x_mean, x_std)
            torch.cuda.empty_cache()
        return x_t

    def p_losses(self, y, noise):
        return F.mse_loss(y, noise)

    def forward(self, x_full, x_full_sparse, x_lidar, t):
        lidar_feat = None#self.lidar_enc(x_lidar)
        out = self.model(x_full, x_full_sparse, lidar_feat, t)
        torch.cuda.empty_cache()
        return out.reshape(t.shape[0],-1,self.hparams['model']['out_dim'])

    def points_to_tensor(self, x_feats, mean, std):
        x_feats = ME.utils.batched_coordinates(list(x_feats[:]), dtype=torch.float32, device=self.device)

        x_coord = x_feats[:,:4].clone() # [batch, x, y, z]
        x_coord[:,1:] = feats_to_coord(x_coord[:,1:], self.hparams['data']['resolution'], mean, std)

        x_t = ME.TensorField(
            features=x_feats[:,1:],
            coordinates=x_coord,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

        torch.cuda.empty_cache()

        return x_t

    def training_step(self, batch:dict, batch_idx):
        # initial random noise
        torch.cuda.empty_cache()
        x0 = batch['splats'].detach()[:,:,3:3+self.hparams['model']['out_dim']]

        noise = torch.randn([batch['splats'].shape[0], batch['splats'].shape[1], self.hparams['model']['out_dim']], device=self.device) 

        # sample step t
        t = torch.randint(0, self.t_steps, size=(batch['splats'].shape[0],)).cuda()
        # sample q at step t
        # For means: we sample noise towards zero to then add to each point the noise (without normalizing the pcd)
        t_means = batch['splats'][:,:,:3]# + self.q_sample(torch.zeros_like(batch['splats'][:,:,:3]), t, noise[:,:,:3])
        # For other attributes: we sample noise regularly
        t_color = self.q_sample(batch['splats'][:,:,3:6], t, self.color_params, noise[:,:,:3])
        t_opacity = self.q_sample(batch['splats'][:,:,6:7], t, self.opacity_params, noise[:,:,3:4])
        t_scale = self.q_sample(batch['splats'][:,:,7:10], t, self.scale_params, noise[:,:,4:7])
        t_quat0 = self.q_sample(batch['splats'][:,:,10:11], t, self.quat0_params, noise[:,:,7:8])
        t_quatr = self.q_sample(batch['splats'][:,:,11:], t, self.quatr_params, noise[:,:,8:])
        t_sample = torch.cat((t_means, t_color, t_opacity, t_scale, t_quat0, t_quatr), dim=-1)
        # replace the original points with the noise sampled
        x_full = self.points_to_tensor(t_sample, batch['mean'], batch['std'])

        # for classifier-free guidance switch between conditional and unconditional training
        # if torch.rand(1) > self.hparams['train']['uncond_prob'] or batch['splats'].shape[0] == 1:
        #     x_part = self.points_to_tensor(batch['points'], batch['mean'], batch['std'])
        # else:
        #     x_part = self.points_to_tensor(
        #         torch.zeros_like(batch['points']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
        #     )
        x_part = None
        denoise_t = self.forward(x_full, x_full.sparse(), x_part, t)
        loss_mse = self.p_losses(denoise_t, x0)
        # loss_mean = (denoise_t.mean())**2
        # loss_std = (denoise_t.std() - 1.)**2
        loss = loss_mse# + self.hparams['diff']['reg_weight'] * (loss_mean + loss_std)
        with torch.no_grad():
            # loss_x = F.mse_loss(denoise_t[:,:,0], x0[:,:,0])
            # loss_y = F.mse_loss(denoise_t[:,:,1], x0[:,:,1])
            # loss_z = F.mse_loss(denoise_t[:,:,2], x0[:,:,2])
            # loss_color = F.mse_loss(denoise_t[:,:,3:6], x0[:,:,3:6])
            # loss_opacity = F.mse_loss(denoise_t[:,:,6], x0[:,:,6])
            # loss_scale = F.mse_loss(denoise_t[:,:,7:10], x0[:,:,7:10])
            # loss_quat = F.mse_loss(denoise_t[:,:,10:], x0[:,:,10:])
            loss_color = F.mse_loss(denoise_t[:,:,:3], x0[:,:,:3])
            loss_opacity = F.mse_loss(denoise_t[:,:,3], x0[:,:,3])
            loss_scale = F.mse_loss(denoise_t[:,:,4:7], x0[:,:,4:7])
            loss_quat = F.mse_loss(denoise_t[:,:,7:], x0[:,:,7:])

        # std_noise = (denoise_t - noise)**2
        # self.log('train/loss_mse', loss_mse)
        # self.log('train/loss_mean', loss_mean)
        # self.log('train/loss_std', loss_std)
        self.log('train/loss', loss)
        # self.log('train/var', std_noise.var())
        # self.log('train/std', std_noise.std())
        if self.hparams['log']['wandb']:
            wandb.log({'train/loss': loss, 
                    # 'train/loss_x': loss_x, 'train/loss_y': loss_y, 'train/loss_z': loss_z,
                    'train/loss_color': loss_color, 'train/loss_opacity': loss_opacity,
                    'train/loss_scale': loss_scale, 'train/loss_quat': loss_quat
                    })
        torch.cuda.empty_cache()

        return loss
    
    def add_noise(self, batch:dict, t=None):
        gt_pts = batch['splats'].detach().cpu().numpy()
        noise = torch.randn([batch['splats'].shape[0], batch['splats'].shape[1], self.hparams['model']['out_dim']], device=self.device) 
        if t == 0:
            print(f"Step 0: ")
            t_sample = torch.tensor(batch['splats'])
            torch.set_printoptions(precision=2, sci_mode=False)
            print(t_sample[:,:,3:].mean(dim=[0,1]))
            print(t_sample[:,:,3:].std(dim=[0,1]))
            x_full = self.points_to_tensor(t_sample, batch['mean'], batch['std'])
            
            x_gen_eval = x_full.F.reshape((gt_pts.shape[0],-1,self.hparams['model']['in_dim']))
        elif t is not None:
            # sample step t
            t = torch.tensor([t,]).cuda()
            # sample q at step t
            # For means: we sample noise towards zero to then add to each point the noise (without normalizing the pcd)
            t_means = batch['splats'][:,:,:3]# + self.q_sample(torch.zeros_like(batch['splats'][:,:,:3]), t, noise[:,:,:3])
            # For other attributes: we sample noise regularly
            t_color = self.q_sample(batch['splats'][:,:,3:6], t, self.color_params, noise[:,:,:3])
            t_opacity = self.q_sample(batch['splats'][:,:,6:7], t, self.opacity_params, noise[:,:,3:4])
            t_scale = self.q_sample(batch['splats'][:,:,7:10], t, self.scale_params, noise[:,:,4:7])
            t_quat0 = self.q_sample(batch['splats'][:,:,10:11], t, self.quat0_params, noise[:,:,7:8])
            t_quatr = self.q_sample(batch['splats'][:,:,11:], t, self.quatr_params, noise[:,:,8:])

            t_sample = torch.cat((t_means, t_color, t_opacity, t_scale, t_quat0, t_quatr), dim=-1)
            print(f"Step {t.item()}: ")
            torch.set_printoptions(precision=2, sci_mode=False)
            print(t_sample[:,:,3:].mean(dim=[0,1]))
            print(t_sample[:,:,3:].std(dim=[0,1]))
            

            # replace the original points with the noise sampled
            x_full = self.points_to_tensor(t_sample, batch['mean'], batch['std'])
            
            x_gen_eval = x_full.F.reshape((gt_pts.shape[0],-1,self.hparams['model']['in_dim']))
        else:
            x_gen_eval = batch['splats'].clone()
            x_gen_eval[:,:,3:3+self.hparams['model']['out_dim']] = noise
        attributes = to_attributes(x_gen_eval[:,:,3:])
        gs = torch.cat([batch['splats'][:,:,:3], attributes], dim=-1)
        return gs           
    
    def generate_3dgs(self, batch:dict):
        self.model.eval()
        # self.lidar_enc.eval()
        with torch.no_grad():
            gt_pts = batch['splats'].detach().cpu().numpy()

            # for inference we get the partial pcd and sample the noise around the partial
            x_init = torch.tensor(batch['splats'])
            x_init[:,:,3:3+self.hparams['model']['out_dim']] = 0
            noise = torch.randn(x_init.shape, device="cuda")
            noise[:,:,:3] = 0
            noise[:,:,3+self.hparams['model']['out_dim']:] = 0
            x_feats = x_init + noise
            x_full = self.points_to_tensor(x_feats, batch['mean'], batch['std'])
            # x_part = self.points_to_tensor(batch['points'], batch['mean'], batch['std'])
            # x_uncond = self.points_to_tensor(
            #     torch.zeros_like(batch['points']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
            # )

            x_gen_eval = self.p_sample_loop(x_init, x_full, gt_pts, batch['mean'], batch['std'])
            x_gen_eval = x_gen_eval.F.reshape((gt_pts.shape[0],-1,self.hparams['model']['in_dim']))
            attributes = to_attributes(x_gen_eval[:,:,3:])
            gs = torch.cat([batch['splats'][:,:,:3], attributes], dim=-1)
        return gs           

    def validation_step(self, batch:dict, batch_idx):
        if batch_idx != 0:
            return

        self.model.eval()
        # self.lidar_enc.eval()
        with torch.no_grad():
            gt_pts = batch['splats'].detach().cpu().numpy()

            # for inference we get the partial pcd and sample the noise around the partial
            x_init = torch.tensor(batch['splats'])
            x_init[:,:,3:3+self.hparams['model']['out_dim']] = 0
            noise = torch.randn(x_init.shape, device="cuda")
            noise[:,:,:3] = 0
            noise[:,:,3+self.hparams['model']['out_dim']:] = 0
            x_feats = x_init + noise
            x_full = self.points_to_tensor(x_feats, batch['mean'], batch['std'])
            # x_part = self.points_to_tensor(batch['points'], batch['mean'], batch['std'])
            # x_uncond = self.points_to_tensor(
            #     torch.zeros_like(batch['points']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
            # )

            x_gen_eval = self.p_sample_loop(x_init, x_full, gt_pts, batch['mean'], batch['std'])
            x_gen_eval = x_gen_eval.F.reshape((gt_pts.shape[0],-1,self.hparams['model']['in_dim']))

            for i in range(len(batch['splats'])):
                pcd_pred = o3d.geometry.PointCloud()
                c_pred = x_gen_eval[i].cpu().detach().numpy()
                pcd_pred.points = o3d.utility.Vector3dVector(c_pred[:,:3])

                pcd_gt = o3d.geometry.PointCloud()
                g_pred = batch['splats'][i].cpu().detach().numpy()
                pcd_gt.points = o3d.utility.Vector3dVector(g_pred[:,:3])

                self.chamfer_distance.update(pcd_gt, pcd_pred)
                self.precision_recall.update(pcd_gt, pcd_pred)

        cd_mean, cd_std = self.chamfer_distance.compute()
        pr, re, f1 = self.precision_recall.compute_auc()

        self.log('val/cd_mean', cd_mean, on_step=True)
        self.log('val/cd_std', cd_std, on_step=True)
        self.log('val/precision', pr, on_step=True)
        self.log('val/recall', re, on_step=True)
        self.log('val/fscore', f1, on_step=True)

        if self.hparams['log']['wandb']:
            wandb.log({'val/cd_mean': cd_mean, 'val/cd_std': cd_std, 'val/precision': pr, 'val/recall': re, 'val/fscore': f1})
        torch.cuda.empty_cache()

        return {'val/cd_mean': cd_mean, 'val/cd_std': cd_std, 'val/precision': pr, 'val/recall': re, 'val/fscore': f1}
    
    def valid_paths(self, filenames):
        output_paths = []
        skip = []

        for fname in filenames:
            seq_dir =  f'{self.logger.log_dir}/generated_pcd/{fname.split("/")[-3]}'
            ply_name = f'{fname.split("/")[-1].split(".")[0]}.ply'

            skip.append(path.isfile(f'{seq_dir}/{ply_name}'))
            makedirs(seq_dir, exist_ok=True)
            output_paths.append(f'{seq_dir}/{ply_name}')

        return np.all(skip), output_paths

    def test_step(self, batch:dict, batch_idx):
        self.model.eval()
        # self.lidar_enc.eval()
        with torch.no_grad():
            skip, output_paths = self.valid_paths(batch['filename'])

            if skip:
                print(f'Skipping generation from {output_paths[0]} to {output_paths[-1]}') 
                return {'test/cd_mean': 0., 'test/cd_std': 0., 'test/precision': 0., 'test/recall': 0., 'test/fscore': 0.}

            gt_pts = batch['splats'].detach().cpu().numpy()

            x_init = torch.zeros_like(batch['splats'])
            x_init[:,:,:3] = batch['splats'][:,:,:3]#batch['points'].repeat(1,10,1)
            noise = torch.randn(x_init.shape, device="cuda")
            noise[:,:,:3] = 0
            x_feats = x_init + noise
            x_full = self.points_to_tensor(x_feats, batch['mean'], batch['std'])
            # x_part = self.points_to_tensor(batch['points'], batch['mean'], batch['std'])
            # x_uncond = self.points_to_tensor(
            #     torch.zeros_like(batch['points']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
            # )

            x_gen_eval = self.p_sample_loop(x_init, x_full, gt_pts, batch['mean'], batch['std'])
            x_gen_eval = x_gen_eval.F.reshape((gt_pts.shape[0],-1,3))

            for i in range(len(batch['splats'])):
                pcd_pred = o3d.geometry.PointCloud()
                c_pred = x_gen_eval[i].cpu().detach().numpy()
                dist_pts = np.sqrt(np.sum((c_pred)**2, axis=-1))
                dist_idx = dist_pts < self.hparams['data']['max_range']
                points = c_pred[dist_idx]
                max_z = x_init[i][...,2].max().item()
                min_z = (x_init[i][...,2].mean() - 2 * x_init[i][...,2].std()).item()
                pcd_pred.points = o3d.utility.Vector3dVector(points[(points[:,2] < max_z) & (points[:,2] > min_z)])
                pcd_pred.paint_uniform_color([1.0, 0.,0.])

                pcd_gt = o3d.geometry.PointCloud()
                g_pred = batch['splats'][i].cpu().detach().numpy()
                pcd_gt.points = o3d.utility.Vector3dVector(g_pred)
                pcd_gt.paint_uniform_color([0., 1.,0.])
                
                print(f'Saving {output_paths[i]}')
                o3d.io.write_point_cloud(f'{output_paths[i]}', pcd_pred)

                self.chamfer_distance.update(pcd_gt, pcd_pred)
                self.precision_recall.update(pcd_gt, pcd_pred)

        cd_mean, cd_std = self.chamfer_distance.compute()
        pr, re, f1 = self.precision_recall.compute_auc()
        print(f'CD Mean: {cd_mean}\tCD Std: {cd_std}')
        print(f'Precision: {pr}\tRecall: {re}\tF-Score: {f1}')

        self.log('test/cd_mean', cd_mean, on_step=True)
        self.log('test/cd_std', cd_std, on_step=True)
        self.log('test/precision', pr, on_step=True)
        self.log('test/recall', re, on_step=True)
        self.log('test/fscore', f1, on_step=True)
        torch.cuda.empty_cache()

        return {'test/cd_mean': cd_mean, 'test/cd_std': cd_std, 'test/precision': pr, 'test/recall': re, 'test/fscore': f1}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['train']['lr'], betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
        scheduler = {
            'scheduler': scheduler, # lr * 0.5
            'interval': 'epoch', # interval is epoch-wise
            'frequency': 5, # after 5 epochs
        }

        return [optimizer], [scheduler]

class DiffusionSplatsPT(LightningModule):
    def __init__(self, hparams:dict, data_module: LightningDataModule = None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module

        # alphas and betas
        self.color_betas = beta_func[self.hparams['diff']['color']['beta_func']](self.hparams['diff']['t_steps']) if self.hparams['diff']['color']['beta_func'] == 'cosine' else beta_func[self.hparams['diff']['color']['beta_func']](
                self.hparams['diff']['t_steps'],
                self.hparams['diff']['color']['beta_start'],
                self.hparams['diff']['color']['beta_end'],
        )
        self.color_params = DiffusionParams(self.color_betas, self.hparams['diff']['t_steps'])

        self.opacity_betas = beta_func[self.hparams['diff']['opacity']['beta_func']](self.hparams['diff']['t_steps']) if self.hparams['diff']['opacity']['beta_func'] == 'cosine' else beta_func[self.hparams['diff']['opacity']['beta_func']](
                self.hparams['diff']['t_steps'],
                self.hparams['diff']['opacity']['beta_start'],
                self.hparams['diff']['opacity']['beta_end'],
        )
        self.opacity_params = DiffusionParams(self.opacity_betas, self.hparams['diff']['t_steps'])

        self.scale_betas = beta_func[self.hparams['diff']['scale']['beta_func']](self.hparams['diff']['t_steps']) if self.hparams['diff']['scale']['beta_func'] == 'cosine' else beta_func[self.hparams['diff']['scale']['beta_func']](
                self.hparams['diff']['t_steps'],
                self.hparams['diff']['scale']['beta_start'],
                self.hparams['diff']['scale']['beta_end'],
        )
        self.scale_params = DiffusionParams(self.scale_betas, self.hparams['diff']['t_steps'])

        self.quat0_betas = beta_func[self.hparams['diff']['quat_0']['beta_func']](self.hparams['diff']['t_steps']) if self.hparams['diff']['quat_0']['beta_func'] == 'cosine' else beta_func[self.hparams['diff']['quat_0']['beta_func']](
                self.hparams['diff']['t_steps'],
                self.hparams['diff']['quat_0']['beta_start'],
                self.hparams['diff']['quat_0']['beta_end'],
        )
        self.quat0_params = DiffusionParams(self.quat0_betas, self.hparams['diff']['t_steps'])

        self.quatr_betas = beta_func[self.hparams['diff']['quat_rest']['beta_func']](self.hparams['diff']['t_steps']) if self.hparams['diff']['quat_rest']['beta_func'] == 'cosine' else beta_func[self.hparams['diff']['quat_rest']['beta_func']](
                self.hparams['diff']['t_steps'],
                self.hparams['diff']['quat_rest']['beta_start'],
                self.hparams['diff']['quat_rest']['beta_end'],
        )
        self.quatr_params = DiffusionParams(self.quatr_betas, self.hparams['diff']['t_steps'])

        self.t_steps = self.hparams['diff']['t_steps']
        self.s_steps = self.hparams['diff']['s_steps']

        # for fast sampling
        self.color_dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.t_steps,
                beta_start=self.hparams['diff']['color']['beta_start'],
                beta_end=self.hparams['diff']['color']['beta_end'],
                beta_schedule=self.hparams['diff']['color']['beta_func'],
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.color_dpm_scheduler.set_timesteps(self.s_steps)

        self.opacity_dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.t_steps,
                beta_start=self.hparams['diff']['opacity']['beta_start'],
                beta_end=self.hparams['diff']['opacity']['beta_end'],
                beta_schedule=self.hparams['diff']['opacity']['beta_func'],
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.opacity_dpm_scheduler.set_timesteps(self.s_steps)

        self.scale_dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.t_steps,
                beta_start=self.hparams['diff']['scale']['beta_start'],
                beta_end=self.hparams['diff']['scale']['beta_end'],
                beta_schedule=self.hparams['diff']['scale']['beta_func'],
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.scale_dpm_scheduler.set_timesteps(self.s_steps)

        self.quat0_dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.t_steps,
                beta_start=self.hparams['diff']['quat_0']['beta_start'],
                beta_end=self.hparams['diff']['quat_0']['beta_end'],
                beta_schedule=self.hparams['diff']['quat_0']['beta_func'],
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.quat0_dpm_scheduler.set_timesteps(self.s_steps)

        self.quatr_dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.t_steps,
                beta_start=self.hparams['diff']['quat_rest']['beta_start'],
                beta_end=self.hparams['diff']['quat_rest']['beta_end'],
                beta_schedule=self.hparams['diff']['quat_rest']['beta_func'],
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.quatr_dpm_scheduler.set_timesteps(self.s_steps)
        self.scheduler_to_cuda()

        # self.lidar_enc = minknet.MinkGlobalEnc(in_channels=3)
        self.model = PointTransformerV3(in_channels=self.hparams['model']['in_dim'], 
                                        out_channels=self.hparams['model']['out_dim'],
                                        enable_flash=False)

        self.chamfer_distance = ChamferDistance()
        self.precision_recall = PrecisionRecall(self.hparams['data']['resolution'],2*self.hparams['data']['resolution'],100)

        self.w_uncond = self.hparams['train']['uncond_w']

    def scheduler_to_cuda(self):
        for scheduler in [self.color_dpm_scheduler, self.opacity_dpm_scheduler, self.scale_dpm_scheduler, self.quat0_dpm_scheduler, self.quatr_dpm_scheduler]:
            scheduler.timesteps = scheduler.timesteps.cuda()
            scheduler.betas = scheduler.betas.cuda()
            scheduler.alphas = scheduler.alphas.cuda()
            scheduler.alphas_cumprod = scheduler.alphas_cumprod.cuda()
            scheduler.alpha_t = scheduler.alpha_t.cuda()
            scheduler.sigma_t = scheduler.sigma_t.cuda()
            scheduler.lambda_t = scheduler.lambda_t.cuda()
            scheduler.sigmas = scheduler.sigmas.cuda()

    def q_sample(self, x, t, param, noise):
        return param.sqrt_alphas_cumprod[t.cpu()][:,None,None].cuda() * x + \
                param.sqrt_one_minus_alphas_cumprod[t.cpu()][:,None,None].cuda() * noise

    def inverse_q_sample(self, xt, t, param, x0):
        return (xt - param.sqrt_alphas_cumprod[t.cpu()][:,None,None].cuda() * x0) / param.sqrt_one_minus_alphas_cumprod[t.cpu()][:,None,None].cuda()
    
    def classfree_forward(self, x_t, x_cond, x_uncond, t):
        x_t_sparse = x_t.sparse()
        x_cond = self.forward(x_t, x_t_sparse, x_cond, t)            
        x_uncond = self.forward(x_t, x_t_sparse, x_uncond, t)

        return x_uncond + self.w_uncond * (x_cond - x_uncond)

    def reset_partial_pcd(self, x_part, x_uncond, x_mean, x_std):
        x_part = self.points_to_tensor(x_part['feat'].reshape(x_mean.shape[0],-1,3).detach(), x_mean, x_std)
        x_uncond = self.points_to_tensor(
                torch.zeros_like(x_part['feat'].reshape(x_mean.shape[0],-1,3)), torch.zeros_like(x_mean), torch.zeros_like(x_std)
        )

        return x_part, x_uncond
    
    def get_noise(self, x_t, t):
        x0 = self.forward(x_t, t) #self.forward(x_t, x_t.sparse(), None, t)
        xt = x_t['feat'].reshape(t.shape[0],-1,self.hparams['model']['in_dim'])
        noise_color = self.inverse_q_sample(xt[:,:,3:6], t, self.color_params, x0[:,:,:3])
        noise_opacity = self.inverse_q_sample(xt[:,:,6:7], t, self.opacity_params, x0[:,:,3:4])
        noise_scale = self.inverse_q_sample(xt[:,:,7:10], t, self.scale_params, x0[:,:,4:7])
        noise_quat0 = self.inverse_q_sample(xt[:,:,10:11], t, self.quat0_params, x0[:,:,7:8])
        noise_quatr = self.inverse_q_sample(xt[:,:,11:], t, self.quatr_params, x0[:,:,8:])
        return torch.cat((noise_color, noise_opacity, noise_scale, noise_quat0, noise_quatr), dim=-1)
     
    # def p_sample_step(self, x_t, t):
    #     noise_t = self.get_noise(x_t, t)#self.classfree_forward(x_t, x_cond, x_uncond, t)
    #     features = x_t['feat'].reshape(t.shape[0],-1,self.hparams['model']['in_dim'])
    #     # xyz = features[:,:,:3]
    #     # others = features[:,:,3:]
    #     # x_tm1_xyz = xyz - self.sampling_coef1_xyz[t.cpu()] * noise_t[:,:,:3] + self.sampling_coef2_xyz[t.cpu()] * torch.randn_like(xyz)
    #     # x_tm1_others = self.sqrt_recip_alphas[t.cpu()] * (others - self.sampling_coef_others[t.cpu()] * noise_t[:,:,3:]) + self.sqrt_posterior_variance[t.cpu()] * torch.randn_like(others)
    #     # x_tm1 = torch.cat((x_tm1_xyz, x_tm1_others), dim=-1)
    #     colors = features[:,:,3:6]
    #     colors = self.color_params.sqrt_recip_alphas[t.cpu()] * (colors - self.color_params.sampling_coef_others[t.cpu()] * noise_t[:,:,:3]) + self.color_params.sqrt_posterior_variance[t.cpu()] * torch.randn_like(colors)
    #     opacity = features[:,:,6:7]
    #     # opacity = self.opacity_params.sqrt_recip_alphas[t.cpu()] * (opacity - self.opacity_params.sampling_coef_others[t.cpu()] * noise_t[:,:,3:4]) + self.opacity_params.sqrt_posterior_variance[t.cpu()] * torch.randn_like(opacity)
    #     scale = features[:,:,7:10]
    #     # scale = self.scale_params.sqrt_recip_alphas[t.cpu()] * (scale - self.scale_params.sampling_coef_others[t.cpu()] * noise_t[:,:,4:7]) + self.scale_params.sqrt_posterior_variance[t.cpu()] * torch.randn_like(scale)
    #     quat0 = features[:,:,10:11]
    #     # quat0 = self.quat0_params.sqrt_recip_alphas[t.cpu()] * (quat0 - self.quat0_params.sampling_coef_others[t.cpu()] * noise_t[:,:,7:8]) + self.quat0_params.sqrt_posterior_variance[t.cpu()] * torch.randn_like(quat0)
    #     quatr = features[:,:,11:]
    #     # quatr = self.quatr_params.sqrt_recip_alphas[t.cpu()] * (quatr - self.quatr_params.sampling_coef_others[t.cpu()] * noise_t[:,:,8:]) + self.quatr_params.sqrt_posterior_variance[t.cpu()] * torch.randn_like(quatr)
    #     x_tm1 = torch.cat((features[:,:,:3], colors, opacity, scale, quat0, quatr), dim=-1)
    #     return x_tm1

    # def p_sample_loop(self, x_init, x_t, gt_pts, x_mean, x_std):
    #     for t in tqdm(range(self.t_steps - 1,0,-1)):
    #         t_ = torch.ones(gt_pts.shape[0]).cuda().long() * t
    #         x_t = self.p_sample_step(x_t, t_)
    #         x_t = self.points_to_tensor(x_t, x_mean, x_std)

    #         # this is needed otherwise minkEngine will keep "stacking" coords maps over the x_part and x_uncond
    #         # i.e. memory leak
    #         # x_cond, x_uncond = self.reset_partial_pcd(x_cond, x_uncond, x_mean, x_std)
    #         torch.cuda.empty_cache()
    #     return x_t

    def p_sample_loop(self, x_init, x_t, gt_pts, x_mean, x_std):
        self.scheduler_to_cuda()

        for t in tqdm(range(len(self.color_dpm_scheduler.timesteps))):
            t = torch.ones(gt_pts.shape[0]).cuda().long() * self.color_dpm_scheduler.timesteps[t].cuda()
            
            noise_t = self.get_noise(x_t, t) #self.forward(x_t, x_t.sparse(), None, t)
            input_noise = x_t['feat'].reshape(t.shape[0],-1,self.hparams['model']['in_dim'])[:,:,3:3+self.hparams['model']['out_dim']] - x_init[:,:,3:3+self.hparams['model']['out_dim']]
            x_t = torch.tensor(x_init)
            x_t[:,:,3:6] += self.color_dpm_scheduler.step(noise_t[:,:,:3], t[0], input_noise[:,:,:3])['prev_sample']
            x_t[:,:,6:7] += self.opacity_dpm_scheduler.step(noise_t[:,:,3:4], t[0], input_noise[:,:,3:4])['prev_sample']
            x_t[:,:,7:10] += self.scale_dpm_scheduler.step(noise_t[:,:,4:7], t[0], input_noise[:,:,4:7])['prev_sample']
            x_t[:,:,10:11] += self.quat0_dpm_scheduler.step(noise_t[:,:,7:8], t[0], input_noise[:,:,7:8])['prev_sample']
            x_t[:,:,11:] += self.quatr_dpm_scheduler.step(noise_t[:,:,8:], t[0], input_noise[:,:,8:])['prev_sample']
            x_t = self.to_Point(x_t, x_mean, x_std)

            # this is needed otherwise minkEngine will keep "stacking" coords maps over the x_part and x_uncond
            # i.e. memory leak
            # x_cond, x_uncond = self.reset_partial_pcd(x_cond, x_uncond, x_mean, x_std)
            torch.cuda.empty_cache()
        return x_t

    def p_losses(self, y, noise):
        return F.mse_loss(y, noise)

    def forward(self, x_full, t):
        out = self.model(x_full, t)
        torch.cuda.empty_cache()
        return out.reshape(t.shape[0],-1,self.hparams['model']['out_dim'])
    
    def to_Point(self, feat, mean, std):
        feat = ME.utils.batched_coordinates(list(feat[:]), dtype=torch.float32, device=self.device)

        batch = feat[:,0].int()
        coord = feat[:,1:4].clone()
        grid_coord = feats_to_coord(coord, self.hparams['data']['resolution'], mean, std).int()

        x_t = Point(
            coord = coord,
            grid_coord = grid_coord,
            batch = batch,
            feat = feat[:,1:],
        )
        torch.cuda.empty_cache()

        return x_t

    def training_step(self, batch:dict, batch_idx):
        # initial random noise
        torch.cuda.empty_cache()
        x0 = batch['splats'].detach()[:,:,3:3+self.hparams['model']['out_dim']]

        noise = torch.randn([batch['splats'].shape[0], batch['splats'].shape[1], self.hparams['model']['out_dim']], device=self.device) 

        # sample step t
        t = torch.randint(0, self.t_steps, size=(batch['splats'].shape[0],)).cuda()
        # sample q at step t
        # For means: we sample noise towards zero to then add to each point the noise (without normalizing the pcd)
        t_means = batch['splats'][:,:,:3]# + self.q_sample(torch.zeros_like(batch['splats'][:,:,:3]), t, noise[:,:,:3])
        # For other attributes: we sample noise regularly
        t_color = self.q_sample(batch['splats'][:,:,3:6], t, self.color_params, noise[:,:,:3])
        t_opacity = self.q_sample(batch['splats'][:,:,6:7], t, self.opacity_params, noise[:,:,3:4])
        t_scale = self.q_sample(batch['splats'][:,:,7:10], t, self.scale_params, noise[:,:,4:7])
        t_quat0 = self.q_sample(batch['splats'][:,:,10:11], t, self.quat0_params, noise[:,:,7:8])
        t_quatr = self.q_sample(batch['splats'][:,:,11:], t, self.quatr_params, noise[:,:,8:])
        t_sample = torch.cat((t_means, t_color, t_opacity, t_scale, t_quat0, t_quatr), dim=-1)
        # replace the original points with the noise sampled
        x_full = self.to_Point(t_sample, batch['mean'], batch['std'])

        # for classifier-free guidance switch between conditional and unconditional training
        # if torch.rand(1) > self.hparams['train']['uncond_prob'] or batch['splats'].shape[0] == 1:
        #     x_part = self.points_to_tensor(batch['points'], batch['mean'], batch['std'])
        # else:
        #     x_part = self.points_to_tensor(
        #         torch.zeros_like(batch['points']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
        #     )
        x_part = None
        denoise_t = self.forward(x_full, t)
        loss_mse = self.p_losses(denoise_t, x0)
        # loss_mean = (denoise_t.mean())**2
        # loss_std = (denoise_t.std() - 1.)**2
        loss = loss_mse# + self.hparams['diff']['reg_weight'] * (loss_mean + loss_std)
        with torch.no_grad():
            # loss_x = F.mse_loss(denoise_t[:,:,0], x0[:,:,0])
            # loss_y = F.mse_loss(denoise_t[:,:,1], x0[:,:,1])
            # loss_z = F.mse_loss(denoise_t[:,:,2], x0[:,:,2])
            # loss_color = F.mse_loss(denoise_t[:,:,3:6], x0[:,:,3:6])
            # loss_opacity = F.mse_loss(denoise_t[:,:,6], x0[:,:,6])
            # loss_scale = F.mse_loss(denoise_t[:,:,7:10], x0[:,:,7:10])
            # loss_quat = F.mse_loss(denoise_t[:,:,10:], x0[:,:,10:])
            loss_color = F.mse_loss(denoise_t[:,:,:3], x0[:,:,:3])
            loss_opacity = F.mse_loss(denoise_t[:,:,3], x0[:,:,3])
            loss_scale = F.mse_loss(denoise_t[:,:,4:7], x0[:,:,4:7])
            loss_quat = F.mse_loss(denoise_t[:,:,7:], x0[:,:,7:])

        # std_noise = (denoise_t - noise)**2
        # self.log('train/loss_mse', loss_mse)
        # self.log('train/loss_mean', loss_mean)
        # self.log('train/loss_std', loss_std)
        self.log('train/loss', loss)
        # self.log('train/var', std_noise.var())
        # self.log('train/std', std_noise.std())
        if self.hparams['log']['wandb']:
            wandb.log({'train/loss': loss, 
                    # 'train/loss_x': loss_x, 'train/loss_y': loss_y, 'train/loss_z': loss_z,
                    'train/loss_color': loss_color, 'train/loss_opacity': loss_opacity,
                    'train/loss_scale': loss_scale, 'train/loss_quat': loss_quat
                    })
        torch.cuda.empty_cache()

        return loss
    
    def add_noise(self, batch:dict, t=None):
        gt_pts = batch['splats'].detach().cpu().numpy()
        noise = torch.randn([batch['splats'].shape[0], batch['splats'].shape[1], self.hparams['model']['out_dim']], device=self.device) 
        if t == 0:
            print(f"Step 0: ")
            t_sample = torch.tensor(batch['splats'])
            torch.set_printoptions(precision=2, sci_mode=False)
            print(t_sample[:,:,3:].mean(dim=[0,1]))
            print(t_sample[:,:,3:].std(dim=[0,1]))
            x_full = self.to_Point(t_sample, batch['mean'], batch['std'])
            
            x_gen_eval = x_full['feat'].reshape((gt_pts.shape[0],-1,self.hparams['model']['in_dim']))
        elif t is not None:
            # sample step t
            t = torch.tensor([t,]).cuda()
            # sample q at step t
            # For means: we sample noise towards zero to then add to each point the noise (without normalizing the pcd)
            t_means = batch['splats'][:,:,:3]# + self.q_sample(torch.zeros_like(batch['splats'][:,:,:3]), t, noise[:,:,:3])
            # For other attributes: we sample noise regularly
            t_color = self.q_sample(batch['splats'][:,:,3:6], t, self.color_params, noise[:,:,:3])
            t_opacity = self.q_sample(batch['splats'][:,:,6:7], t, self.opacity_params, noise[:,:,3:4])
            t_scale = self.q_sample(batch['splats'][:,:,7:10], t, self.scale_params, noise[:,:,4:7])
            t_quat0 = self.q_sample(batch['splats'][:,:,10:11], t, self.quat0_params, noise[:,:,7:8])
            t_quatr = self.q_sample(batch['splats'][:,:,11:], t, self.quatr_params, noise[:,:,8:])

            t_sample = torch.cat((t_means, t_color, t_opacity, t_scale, t_quat0, t_quatr), dim=-1)
            print(f"Step {t.item()}: ")
            torch.set_printoptions(precision=2, sci_mode=False)
            print(t_sample[:,:,3:].mean(dim=[0,1]))
            print(t_sample[:,:,3:].std(dim=[0,1]))
            

            # replace the original points with the noise sampled
            x_full = self.to_Point(t_sample, batch['mean'], batch['std'])
            
            x_gen_eval = x_full['feat'].reshape((gt_pts.shape[0],-1,self.hparams['model']['in_dim']))
        else:
            x_gen_eval = batch['splats'].clone()
            x_gen_eval[:,:,3:3+self.hparams['model']['out_dim']] = noise
        attributes = to_attributes(x_gen_eval[:,:,3:])
        gs = torch.cat([batch['splats'][:,:,:3], attributes], dim=-1)
        return gs           
    
    def generate_3dgs(self, batch:dict):
        self.model.eval()
        # self.lidar_enc.eval()
        with torch.no_grad():
            gt_pts = batch['splats'].detach().cpu().numpy()

            # for inference we get the partial pcd and sample the noise around the partial
            x_init = torch.tensor(batch['splats'])
            x_init[:,:,3:3+self.hparams['model']['out_dim']] = 0
            noise = torch.randn(x_init.shape, device="cuda")
            noise[:,:,:3] = 0
            noise[:,:,3+self.hparams['model']['out_dim']:] = 0
            x_feats = x_init + noise
            x_full = self.to_Point(x_feats, batch['mean'], batch['std'])
            # x_part = self.points_to_tensor(batch['points'], batch['mean'], batch['std'])
            # x_uncond = self.points_to_tensor(
            #     torch.zeros_like(batch['points']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
            # )

            x_gen_eval = self.p_sample_loop(x_init, x_full, gt_pts, batch['mean'], batch['std'])
            x_gen_eval = x_gen_eval['feat'].reshape((gt_pts.shape[0],-1,self.hparams['model']['in_dim']))
            attributes = to_attributes(x_gen_eval[:,:,3:])
            gs = torch.cat([batch['splats'][:,:,:3], attributes], dim=-1)
        return gs           

    def validation_step(self, batch:dict, batch_idx):
        if batch_idx != 0:
            return

        self.model.eval()
        # self.lidar_enc.eval()
        with torch.no_grad():
            gt_pts = batch['splats'].detach().cpu().numpy()

            # for inference we get the partial pcd and sample the noise around the partial
            x_init = torch.tensor(batch['splats'])
            x_init[:,:,3:3+self.hparams['model']['out_dim']] = 0
            noise = torch.randn(x_init.shape, device="cuda")
            noise[:,:,:3] = 0
            noise[:,:,3+self.hparams['model']['out_dim']:] = 0
            x_feats = x_init + noise
            x_full = self.to_Point(x_feats, batch['mean'], batch['std'])
            # x_part = self.points_to_tensor(batch['points'], batch['mean'], batch['std'])
            # x_uncond = self.points_to_tensor(
            #     torch.zeros_like(batch['points']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
            # )

            x_gen_eval = self.p_sample_loop(x_init, x_full, gt_pts, batch['mean'], batch['std'])
            x_gen_eval = x_gen_eval['feat'].reshape((gt_pts.shape[0],-1,self.hparams['model']['in_dim']))

            for i in range(len(batch['splats'])):
                pcd_pred = o3d.geometry.PointCloud()
                c_pred = x_gen_eval[i].cpu().detach().numpy()
                pcd_pred.points = o3d.utility.Vector3dVector(c_pred[:,:3])

                pcd_gt = o3d.geometry.PointCloud()
                g_pred = batch['splats'][i].cpu().detach().numpy()
                pcd_gt.points = o3d.utility.Vector3dVector(g_pred[:,:3])

                self.chamfer_distance.update(pcd_gt, pcd_pred)
                self.precision_recall.update(pcd_gt, pcd_pred)

        cd_mean, cd_std = self.chamfer_distance.compute()
        pr, re, f1 = self.precision_recall.compute_auc()

        self.log('val/cd_mean', cd_mean, on_step=True)
        self.log('val/cd_std', cd_std, on_step=True)
        self.log('val/precision', pr, on_step=True)
        self.log('val/recall', re, on_step=True)
        self.log('val/fscore', f1, on_step=True)

        if self.hparams['log']['wandb']:
            wandb.log({'val/cd_mean': cd_mean, 'val/cd_std': cd_std, 'val/precision': pr, 'val/recall': re, 'val/fscore': f1})
        torch.cuda.empty_cache()

        return {'val/cd_mean': cd_mean, 'val/cd_std': cd_std, 'val/precision': pr, 'val/recall': re, 'val/fscore': f1}
    
    def valid_paths(self, filenames):
        output_paths = []
        skip = []

        for fname in filenames:
            seq_dir =  f'{self.logger.log_dir}/generated_pcd/{fname.split("/")[-3]}'
            ply_name = f'{fname.split("/")[-1].split(".")[0]}.ply'

            skip.append(path.isfile(f'{seq_dir}/{ply_name}'))
            makedirs(seq_dir, exist_ok=True)
            output_paths.append(f'{seq_dir}/{ply_name}')

        return np.all(skip), output_paths

    def test_step(self, batch:dict, batch_idx):
        self.model.eval()
        # self.lidar_enc.eval()
        with torch.no_grad():
            skip, output_paths = self.valid_paths(batch['filename'])

            if skip:
                print(f'Skipping generation from {output_paths[0]} to {output_paths[-1]}') 
                return {'test/cd_mean': 0., 'test/cd_std': 0., 'test/precision': 0., 'test/recall': 0., 'test/fscore': 0.}

            gt_pts = batch['splats'].detach().cpu().numpy()

            x_init = torch.zeros_like(batch['splats'])
            x_init[:,:,:3] = batch['splats'][:,:,:3]#batch['points'].repeat(1,10,1)
            noise = torch.randn(x_init.shape, device="cuda")
            noise[:,:,:3] = 0
            x_feats = x_init + noise
            x_full = self.points_to_tensor(x_feats, batch['mean'], batch['std'])
            # x_part = self.points_to_tensor(batch['points'], batch['mean'], batch['std'])
            # x_uncond = self.points_to_tensor(
            #     torch.zeros_like(batch['points']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
            # )

            x_gen_eval = self.p_sample_loop(x_init, x_full, gt_pts, batch['mean'], batch['std'])
            x_gen_eval = x_gen_eval['feat'].reshape((gt_pts.shape[0],-1,3))

            for i in range(len(batch['splats'])):
                pcd_pred = o3d.geometry.PointCloud()
                c_pred = x_gen_eval[i].cpu().detach().numpy()
                dist_pts = np.sqrt(np.sum((c_pred)**2, axis=-1))
                dist_idx = dist_pts < self.hparams['data']['max_range']
                points = c_pred[dist_idx]
                max_z = x_init[i][...,2].max().item()
                min_z = (x_init[i][...,2].mean() - 2 * x_init[i][...,2].std()).item()
                pcd_pred.points = o3d.utility.Vector3dVector(points[(points[:,2] < max_z) & (points[:,2] > min_z)])
                pcd_pred.paint_uniform_color([1.0, 0.,0.])

                pcd_gt = o3d.geometry.PointCloud()
                g_pred = batch['splats'][i].cpu().detach().numpy()
                pcd_gt.points = o3d.utility.Vector3dVector(g_pred)
                pcd_gt.paint_uniform_color([0., 1.,0.])
                
                print(f'Saving {output_paths[i]}')
                o3d.io.write_point_cloud(f'{output_paths[i]}', pcd_pred)

                self.chamfer_distance.update(pcd_gt, pcd_pred)
                self.precision_recall.update(pcd_gt, pcd_pred)

        cd_mean, cd_std = self.chamfer_distance.compute()
        pr, re, f1 = self.precision_recall.compute_auc()
        print(f'CD Mean: {cd_mean}\tCD Std: {cd_std}')
        print(f'Precision: {pr}\tRecall: {re}\tF-Score: {f1}')

        self.log('test/cd_mean', cd_mean, on_step=True)
        self.log('test/cd_std', cd_std, on_step=True)
        self.log('test/precision', pr, on_step=True)
        self.log('test/recall', re, on_step=True)
        self.log('test/fscore', f1, on_step=True)
        torch.cuda.empty_cache()

        return {'test/cd_mean': cd_mean, 'test/cd_std': cd_std, 'test/precision': pr, 'test/recall': re, 'test/fscore': f1}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['train']['lr'], betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
        scheduler = {
            'scheduler': scheduler, # lr * 0.5
            'interval': 'epoch', # interval is epoch-wise
            'frequency': 5, # after 5 epochs
        }

        return [optimizer], [scheduler]

#######################################
# Modules
#######################################

