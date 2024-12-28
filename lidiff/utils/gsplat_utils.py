import numpy as np
import torch
from plyfile import PlyData, PlyElement

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
    x[:, :, :3] = x[:, :, :3] * 2
    x[:, :, 3] = torch.clamp(x[:, :, 3] * 0.5 + 0.5, min=0, max=1)
    x[:, :, 4:7] = torch.clamp(torch.exp(x[:, :, 4:7] * 8 - 3), min=0, max=40)
    x[:, :, 7] = torch.clamp(x[:, :, 7] * 0.25 + 0.75, min=0.5, max=1)
    x[:, :, 8:] = torch.clamp(x[:, :, 8:] * (0.5 ** 0.5), min=-0.5**0.5, max=0.5**0.5)
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
    x[:, :3] = x[:, :3] / 2
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