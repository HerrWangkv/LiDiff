import numpy as np
import torch
from plyfile import PlyData, PlyElement
from gsplat.rendering import rasterization

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def to_attributes(x):
    '''
    Args:
        x: torch.Tensor of shape (N, 11) or (B, N, 11), x consists of [f_dc_0, f_dc_1, f_dc_2, opacity, scale_0, scale_1, scale_2, rotation_0, rotation_1, rotation_2, rotation_3]
    Returns:
        torch.Tensor of shape (N, 11) or (B, N, 11)
    '''
    dim = len(x.shape)
    assert dim in [2, 3]
    if dim == 2:
        x = x.unsqueeze(0)
    x[:, :, :3] = inverse_sigmoid(torch.clamp(x[:, :, :3] * 0.5 + 0.5, min=0, max=1))
    x[:, :, 3] = torch.clamp(x[:, :, 3] * 0.5 + 0.5, min=0, max=1)
    x[:, :, 4:7] = torch.clamp(torch.exp(x[:, :, 4:7] * 8 - 3), min=0, max=40)
    x[:, :, 7] = torch.clamp(x[:, :, 7] * 0.25 + 0.75, min=0.5, max=1)
    x[:, :, 8:] = torch.clamp(x[:, :, 8:] * (0.5 ** 0.5), min=-1, max=0.5**0.5)
    return x if dim == 3 else x[0]


def normalize_attributes(x):
    '''
    Normalize all attributes to about [-1, 1]
    Args:
        x: np.array of shape (N, 11), x consists of [f_dc_0, f_dc_1, f_dc_2, opacity, scale_0, scale_1, scale_2, rotation_0, rotation_1, rotation_2, rotation_3]
    Returns:
        np.array of shape (N, 11)
    '''
    assert len(x.shape) == 2
    x[:, :3] = (sigmoid(x[:, :3]) - 0.5) / 0.5
    x[:, 3] = (x[:, 3] - 0.5) / 0.5
    x[:, 4:7] = (np.log(x[:, 4:7] + 1e-6) + 3) / 8
    x[:, 7] = (x[:, 7] - 0.75) / 0.25
    x[:, 8:] = x[:, 8:] / (0.5 ** 0.5)
    return x

def export_ply(gaussian, out_path):
    if isinstance(gaussian, torch.Tensor):
        gaussian = gaussian.detach().cpu().numpy()

    l = ['x', 'y', 'z']
    for i in range(3):
        l.append('f_dc_{}'.format(i))
    l.append('opacity')
    for i in range(3):
        l.append('scale_{}'.format(i))
    for i in range(4):
        l.append('rot_{}'.format(i))

    dtype_full = [(attribute, 'f4') for attribute in l]

    elements = np.empty(gaussian.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, gaussian))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(out_path)

def render_gaussian(gaussian, extrinsics, intrinsics, width=533, height=300):
    gaussian = torch.tensor(gaussian).float().cuda()
    extrinsics = torch.tensor(extrinsics).float().cuda()
    intrinsics = torch.tensor(intrinsics).float().cuda()
    intrinsics[0] *= width / 1600
    intrinsics[1] *= height / 900
    assert len(gaussian.shape) == 2 and gaussian.shape[1] == 14
    means = gaussian[:, :3]
    f_dc = gaussian[:, 3:6]
    opacities = gaussian[:, 6]
    scales = gaussian[:, 7:10]
    rotations = gaussian[:, 10:]

    rgbs = torch.sigmoid(f_dc)
    renders, _, _ = rasterization(
        means=means,
        quats=rotations,
        scales=scales,
        opacities=opacities.squeeze(),
        colors=rgbs,
        viewmats=torch.linalg.inv(extrinsics)[None, ...],  # [C, 4, 4]
        Ks=intrinsics[None, ...],  # [C, 3, 3]
        width=width,
        height=height,
        packed=False,
        absgrad=True,
        sparse_grad=False,
        rasterize_mode="classic",
        near_plane=0.1,
        far_plane=10000000000.0,
        render_mode="RGB",
        radius_clip=0.
    )
    renders = torch.clamp(renders, max=1.0)
    return renders

