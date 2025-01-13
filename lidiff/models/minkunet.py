import time
from collections import OrderedDict

import torch
import torch.nn as nn
import MinkowskiEngine as ME
import numpy as np
from pykeops.torch import LazyTensor

__all__ = ['MinkUNetDiff']


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(inc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=stride,
                                 dimension=D),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(inplace=True)
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(inc,
                                 outc,
                                 kernel_size=ks,
                                 stride=stride,
                                 dimension=D),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(inc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=stride,
                                 dimension=D),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(outc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=1,
                                 dimension=D),
            ME.MinkowskiBatchNorm(outc)
        )

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                ME.MinkowskiConvolution(inc, outc, kernel_size=1, dilation=1, stride=stride, dimension=D),
                ME.MinkowskiBatchNorm(outc)
            )

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class MinkGlobalEnc(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        cr = kwargs.get('cr', 1.0)
        in_channels = kwargs.get('in_channels', 3)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.embed_dim = cs[-1]
        self.run_up = kwargs.get('run_up', True)
        self.D = kwargs.get('D', 3)
        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, cs[0], kernel_size=3, stride=1, dimension=self.D),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(cs[0], cs[0], kernel_size=3, stride=1, dimension=self.D),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(inplace=True)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.stem(x.sparse())
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        return x4


class MinkUNetDiff(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        in_channels = kwargs.get('in_channels', 3)
        out_channels = kwargs.get('out_channels', 3)
        cs = [64, 64, 128, 256, 512, 1024, 1024, 1024, 1024, 512, 256, 192, 192]
        cs = [int(cr * x) for x in cs] 
        self.embed_dim = cs[-1]
        self.run_up = kwargs.get('run_up', True)
        self.D = kwargs.get('D', 3)
        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, cs[0], kernel_size=3, stride=1, dimension=self.D),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(cs[0], cs[0], kernel_size=3, stride=1, dimension=self.D),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(inplace=True)
        )
        hidden_dim = cs[6]
        # Stage1 temp embed proj and conv
        # self.latent_stage1 = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),              
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Linear(hidden_dim, hidden_dim),
        # )

        self.latemp_stage1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Linear(hidden_dim+hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_dim, cs[0]),
        )

        self.stage1_temp  = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embed_dim, hidden_dim),
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1, D=self.D),
        )

        # Stage2 temp embed proj and conv
        # self.latent_stage2 = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Linear(hidden_dim, hidden_dim),
        # )

        self.latemp_stage2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Linear(hidden_dim+hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_dim, cs[1]),
        )

        self.stage2_temp  = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embed_dim, hidden_dim),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1, D=self.D)
        )

        # Stage3 temp embed proj and conv
        # self.latent_stage3 = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Linear(hidden_dim, hidden_dim),
        # )

        self.latemp_stage3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Linear(hidden_dim+hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_dim, cs[2]),
        )

        self.stage3_temp  = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embed_dim, hidden_dim),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1, D=self.D),
        )

        # Stage4 temp embed proj and conv
        # self.latent_stage4 = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Linear(hidden_dim, hidden_dim),              
        # )

        self.latemp_stage4 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Linear(hidden_dim+hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_dim, cs[3]),
        )

        self.stage4_temp  = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embed_dim, hidden_dim),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1, D=self.D),
        )

        # Stage5 temp embed proj and conv
        # self.latent_stage5 = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Linear(hidden_dim, hidden_dim),              
        # )

        self.latemp_stage5 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Linear(hidden_dim+hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_dim, cs[4]),
        )

        self.stage5_temp  = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embed_dim, hidden_dim),
        )

        self.stage5 = nn.Sequential(
            BasicConvolutionBlock(cs[4], cs[4], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[4], cs[5], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1, D=self.D),
        )

        # Stage6 temp embed proj and conv
        # self.latent_stage6 = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Linear(hidden_dim, hidden_dim),              
        # )

        self.latemp_stage6 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Linear(hidden_dim+hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_dim, cs[5]),
        )

        self.stage6_temp  = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embed_dim, hidden_dim),
        )

        self.stage6 = nn.Sequential(
            BasicConvolutionBlock(cs[5], cs[5], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[5], cs[6], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1, D=self.D),
        )

        # Up1 temp embed proj and conv
        # self.latent_up1 = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Linear(hidden_dim, hidden_dim),
        # )

        self.latemp_up1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Linear(hidden_dim+hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.up1_temp  = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embed_dim, hidden_dim),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlock(cs[7] + cs[5], cs[7], ks=3, stride=1,
                              dilation=1, D=self.D),
                ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1, D=self.D),
            )
        ])

        # Up2 temp embed proj and conv
        # self.latent_up2 = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Linear(hidden_dim, hidden_dim),
        # )

        self.latemp_up2 = nn.Sequential(
            nn.Linear(hidden_dim, cs[7]),
            # nn.Linear(hidden_dim+hidden_dim, cs[7]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(cs[7], cs[7]),
        )

        self.up2_temp  = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embed_dim, hidden_dim),
        )

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlock(cs[8] + cs[4], cs[8], ks=3, stride=1,
                              dilation=1, D=self.D),
                ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1, D=self.D),
            )
        ])

        # Up3 temp embed proj and conv
        # self.latent_up3 = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Linear(hidden_dim, hidden_dim),
        # )

        self.latemp_up3 = nn.Sequential(
            nn.Linear(hidden_dim, cs[8]),
            # nn.Linear(hidden_dim+hidden_dim, cs[8]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(cs[8], cs[8]),
        )

        self.up3_temp  = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embed_dim, hidden_dim),
        )

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[8], cs[9], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlock(cs[9] + cs[3], cs[9], ks=3, stride=1,
                              dilation=1, D=self.D),
                ResidualBlock(cs[9], cs[9], ks=3, stride=1, dilation=1, D=self.D),
            )
        ])

        # Up4 temp embed proj and conv
        # self.latent_up4 = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),              
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Linear(hidden_dim, hidden_dim),
        # )

        self.latemp_up4 = nn.Sequential(
            nn.Linear(hidden_dim, cs[9]),
            # nn.Linear(hidden_dim+hidden_dim, cs[9]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(cs[9], cs[9]),
        )

        self.up4_temp  = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embed_dim, hidden_dim),
        )

        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[9], cs[10], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlock(cs[10] + cs[2], cs[10], ks=3, stride=1,
                              dilation=1, D=self.D),
                ResidualBlock(cs[10], cs[10], ks=3, stride=1, dilation=1, D=self.D),
            )
        ])

        # Up5 temp embed proj and conv
        # self.latent_up5 = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),              
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Linear(hidden_dim, hidden_dim),
        # )

        self.latemp_up5 = nn.Sequential(
            nn.Linear(hidden_dim, cs[10]),
            # nn.Linear(hidden_dim+hidden_dim, cs[10]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(cs[10], cs[10]),
        )

        self.up5_temp  = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embed_dim, hidden_dim),
        )

        self.up5 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[10], cs[11], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlock(cs[11] + cs[1], cs[11], ks=3, stride=1,
                              dilation=1, D=self.D),
                ResidualBlock(cs[11], cs[11], ks=3, stride=1, dilation=1, D=self.D),
            )
        ])

        # Up6 temp embed proj and conv
        # self.latent_up6 = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),              
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Linear(hidden_dim, hidden_dim),
        # )

        self.latemp_up6 = nn.Sequential(
            nn.Linear(hidden_dim, cs[11]),
            # nn.Linear(hidden_dim+hidden_dim, cs[11]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(cs[11], cs[11]),
        )

        self.up6_temp  = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embed_dim, hidden_dim),
        )

        self.up6 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[11], cs[12], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlock(cs[12] + cs[0], cs[12], ks=3, stride=1,
                              dilation=1, D=self.D),
                ResidualBlock(cs[12], cs[12], ks=3, stride=1, dilation=1, D=self.D),
            )
        ])

        self.last  = nn.Sequential(
            nn.Linear(cs[12], 40),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(40, out_channels),
        )

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_timestep_embedding(self, timesteps):
        assert len(timesteps.shape) == 1 

        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(torch.device('cuda'))
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        assert emb.shape == torch.Size([timesteps.shape[0], self.embed_dim])
        return emb

    def match_part_to_full(self, x_full, x_part):
        full_c = x_full.C.clone().float()
        part_c = x_part.C.clone().float()

        # hash batch coord
        max_coord = full_c.max()
        full_c[:,0] *= max_coord * 2.
        part_c[:,0] *= max_coord * 2.

        f_coord = LazyTensor(full_c[:,None,:])
        p_coord = LazyTensor(part_c[None,:,:])

        dist_fp = ((f_coord - p_coord)**2).sum(-1)
        match_feats = dist_fp.argKmin(1,dim=1)[:,0]

        return x_part.F[match_feats]

    def forward(self, x, x_sparse, part_feats, t):
        temp_emb = self.get_timestep_embedding(t)

        x0 = self.stem(x_sparse)
        # print(x0.F.shape)
        # match0 = self.match_part_to_full(x0, part_feats)
        # p0 = self.latent_stage1(match0) 
        t0 = self.stage1_temp(temp_emb)
        batch_temp = torch.unique(x0.C[:,0], return_counts=True)[1]
        t0 = torch.repeat_interleave(t0, batch_temp, dim=0) 
        w0 = self.latemp_stage1(t0)#torch.cat((p0,t0),-1))

        x1 = self.stage1(x0*w0)
        # print(x1.F.shape)
        # match1 = self.match_part_to_full(x1, part_feats)
        # p1 = self.latent_stage2(match1) 
        t1 = self.stage2_temp(temp_emb)
        batch_temp = torch.unique(x1.C[:,0], return_counts=True)[1]
        t1 = torch.repeat_interleave(t1, batch_temp, dim=0)
        w1 = self.latemp_stage2(t1)#torch.cat((p1,t1),-1))

        x2 = self.stage2(x1*w1)
        # print(x2.F.shape)
        # match2 = self.match_part_to_full(x2, part_feats)
        # p2 = self.latent_stage3(match2) 
        t2 = self.stage3_temp(temp_emb)
        batch_temp = torch.unique(x2.C[:,0], return_counts=True)[1]
        t2 = torch.repeat_interleave(t2, batch_temp, dim=0)
        w2 = self.latemp_stage3(t2)#torch.cat((p2,t2),-1))

        x3 = self.stage3(x2*w2)
        # print(x3.F.shape)
        # match3 = self.match_part_to_full(x3, part_feats)
        # p3 = self.latent_stage4(match3) 
        t3 = self.stage4_temp(temp_emb)
        batch_temp = torch.unique(x3.C[:,0], return_counts=True)[1]
        t3 = torch.repeat_interleave(t3, batch_temp, dim=0)
        w3 = self.latemp_stage4(t3)#torch.cat((p3,t3),-1))

        x4 = self.stage4(x3*w3)
        # print(x4.F.shape)
        # match4 = self.match_part_to_full(x4, part_feats)
        # p4 = self.latent_stage5(match4) 
        t4 = self.stage5_temp(temp_emb)
        batch_temp = torch.unique(x4.C[:,0], return_counts=True)[1]
        t4 = torch.repeat_interleave(t4, batch_temp, dim=0)
        w4 = self.latemp_stage5(t4)#torch.cat((p4,t4),-1))

        x5 = self.stage5(x4*w4)
        # print(x5.F.shape)
        # match5 = self.match_part_to_full(x5, part_feats)
        # p5 = self.latent_stage6(match5) 
        t5 = self.stage6_temp(temp_emb)
        batch_temp = torch.unique(x5.C[:,0], return_counts=True)[1]
        t5 = torch.repeat_interleave(t5, batch_temp, dim=0)
        w5 = self.latemp_stage6(t5)#torch.cat((p5,t5),-1))

        x6 = self.stage6(x5*w5)
        # print(x6.F.shape)
        # match6 = self.match_part_to_full(x6, part_feats)
        # p6 = self.latent_up1(match6) 
        t6 = self.up1_temp(temp_emb)
        batch_temp = torch.unique(x6.C[:,0], return_counts=True)[1]
        t6 = torch.repeat_interleave(t6, batch_temp, dim=0)
        w6 = self.latemp_up1(t6)#torch.cat((t6,p6),-1))

        y1 = self.up1[0](x6*w6)
        # print(y1.F.shape)
        y1 = ME.cat(y1, x5)
        y1 = self.up1[1](y1)
        # match7 = self.match_part_to_full(y1, part_feats)
        # p7 = self.latent_up2(match7) 
        t7 = self.up2_temp(temp_emb)
        batch_temp = torch.unique(y1.C[:,0], return_counts=True)[1]
        t7 = torch.repeat_interleave(t7, batch_temp, dim=0)
        w7 = self.latemp_up2(t7)#torch.cat((p7,t7),-1))

        y2 = self.up2[0](y1*w7)
        # print(y2.F.shape)
        y2 = ME.cat(y2, x4)
        y2 = self.up2[1](y2)
        # match8 = self.match_part_to_full(y2, part_feats)
        # p8 = self.latent_up3(match8) 
        t8 = self.up3_temp(temp_emb)
        batch_temp = torch.unique(y2.C[:,0], return_counts=True)[1]
        t8 = torch.repeat_interleave(t8, batch_temp, dim=0)
        w8 = self.latemp_up3(t8)#torch.cat((p8,t8),-1))       

        y3 = self.up3[0](y2*w8)
        # print(y3.F.shape)
        y3 = ME.cat(y3, x3)
        y3 = self.up3[1](y3)
        # match9 = self.match_part_to_full(y3, part_feats)
        # p9 = self.latent_up4(match9) 
        t9 = self.up4_temp(temp_emb)
        batch_temp = torch.unique(y3.C[:,0], return_counts=True)[1]
        t9 = torch.repeat_interleave(t9, batch_temp, dim=0)
        w9 = self.latemp_up4(t9)#torch.cat((p9,t9),-1))
        
        y4 = self.up4[0](y3*w9)
        # print(y4.F.shape)
        y4 = ME.cat(y4, x2)
        y4 = self.up4[1](y4)
        # match10 = self.match_part_to_full(y4, part_feats)
        # p10 = self.latent_up5(match10) 
        t10 = self.up5_temp(temp_emb)
        batch_temp = torch.unique(y4.C[:,0], return_counts=True)[1]
        t10 = torch.repeat_interleave(t10, batch_temp, dim=0)
        w10 = self.latemp_up5(t10)#torch.cat((p10,t10),-1))

        y5 = self.up5[0](y4*w10)
        # print(y5.F.shape)
        y5 = ME.cat(y5, x1)
        y5 = self.up5[1](y5)
        # match11 = self.match_part_to_full(y5, part_feats)
        # p11 = self.latent_up6(match11) 
        t11 = self.up6_temp(temp_emb)
        batch_temp = torch.unique(y5.C[:,0], return_counts=True)[1]
        t11 = torch.repeat_interleave(t11, batch_temp, dim=0)
        w11 = self.latemp_up6(t11)#torch.cat((p11,t11),-1))

        y6 = self.up6[0](y5*w11)
        # print(y6.F.shape)
        y6 = ME.cat(y6, x0)
        y6 = self.up6[1](y6)
         
        return self.last(y6.slice(x).F)


class MinkUNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        in_channels = kwargs.get('in_channels', 3)
        out_channels = kwargs.get('out_channels', 3)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.run_up = kwargs.get('run_up', True)
        self.D = kwargs.get('D', 3)
        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, cs[0], kernel_size=3, stride=1, dimension=self.D),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(cs[0], cs[0], kernel_size=3, stride=1, dimension=self.D),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(inplace=True)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1, D=self.D)
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlock(cs[5] + cs[3], cs[5], ks=3, stride=1,
                              dilation=1, D=self.D),
                ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1, D=self.D),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlock(cs[6] + cs[2], cs[6], ks=3, stride=1,
                              dilation=1, D=self.D),
                ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1, D=self.D),
            )
        ])

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlock(cs[7] + cs[1], cs[7], ks=3, stride=1,
                              dilation=1, D=self.D),
                ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1, D=self.D),
            )
        ])

        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1,
                              dilation=1, D=self.D),
                ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1, D=self.D),
            )
        ])

        self.last  = nn.Sequential(
            nn.Linear(cs[8], 20),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(20, out_channels),
            nn.Tanh(),
        )

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.stem(x.sparse())
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        y1 = self.up1[0](x4)
        y1 = ME.cat(y1, x3)
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = ME.cat(y2, x2)
        y2 = self.up2[1](y2)

        y3 = self.up3[0](y2)
        y3 = ME.cat(y3, x1)
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = ME.cat(y4, x0)
        y4 = self.up4[1](y4)

        return self.last(y4.slice(x).F)


