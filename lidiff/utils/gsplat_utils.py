import numpy as np
import torch
from plyfile import PlyData, PlyElement
from gsplat.rendering import rasterization

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def quat_to_rotmat(q):
    '''
    Args:
        q: np.array of shape (N, 4), q consists of [rotation_0, rotation_1, rotation_2, rotation_3]
    Returns:
        np.array of shape (N, 3, 3)
    '''
    assert q.shape[-1] == 4
    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r00 = 1 - 2 * q2 ** 2 - 2 * q3 ** 2
    r01 = 2 * q1 * q2 - 2 * q0 * q3
    r02 = 2 * q1 * q3 + 2 * q0 * q2
    r10 = 2 * q1 * q2 + 2 * q0 * q3
    r11 = 1 - 2 * q1 ** 2 - 2 * q3 ** 2
    r12 = 2 * q2 * q3 - 2 * q0 * q1
    r20 = 2 * q1 * q3 - 2 * q0 * q2
    r21 = 2 * q2 * q3 + 2 * q0 * q1
    r22 = 1 - 2 * q1 ** 2 - 2 * q2 ** 2
    rotmat = np.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], axis=-1).reshape(-1, 3, 3)
    return rotmat

def rotmat_to_orth6d(rotmat):
    '''
    Args:
        rotmat: np.array of shape (N, 3, 3)
    Returns:
        np.array of shape (N, 6), following `On the Continuity of Rotation Representations in Neural Networks`
    '''
    assert rotmat.shape[-2:] == (3, 3)
    a1 = rotmat[:, :, 0]
    a2 = rotmat[:, :, 1]
    ret = np.concatenate([a1, a2], axis=-1)
    return ret

def quat_to_orth6d(q):
    return rotmat_to_orth6d(quat_to_rotmat(q))

def normalize_vector( v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v / v_mag
    return v

def cross_product( u, v):
    batch = u.shape[0]
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out

def orth6d_to_rotmat(orth6d):
    '''
    Args:
        orth6d: torch.Tensor of shape (N, 6), following https://github.com/papagina/RotationContinuity/blob/master/shapenet/code/tools.py#L82C49-L95C18
    Returns:
        torch.Tensor of shape (N, 3, 3)
    '''

    x_raw = orth6d[:,0:3]#batch*3
    y_raw = orth6d[:,3:6]#batch*3
        
    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = cross_product(z,x)#batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    rotmat = torch.cat((x,y,z), 2) #batch*3*3
    return rotmat

def rotmat_to_quat(rotmat):
    '''
    Args:
        rotmat: torch.Tensor of shape (N, 3, 3)
    Returns:
        torch.Tensor of shape (N, 4), q consists of [rotation_0, rotation_1, rotation_2, rotation_3]
    '''
    assert rotmat.shape[-2:] == (3, 3)
    r00, r01, r02 = rotmat[:, 0, 0], rotmat[:, 0, 1], rotmat[:, 0, 2]
    r10, r11, r12 = rotmat[:, 1, 0], rotmat[:, 1, 1], rotmat[:, 1, 2]
    r20, r21, r22 = rotmat[:, 2, 0], rotmat[:, 2, 1], rotmat[:, 2, 2]
    trace = r00 + r11 + r22
    q0 = 0.5 * torch.sqrt(1 + trace)
    q1 = 0.5 * torch.sign(r21 - r12) * torch.sqrt(1 + r00 - r11 - r22)
    q2 = 0.5 * torch.sign(r02 - r20) * torch.sqrt(1 - r00 + r11 - r22)
    q3 = 0.5 * torch.sign(r10 - r01) * torch.sqrt(1 - r00 - r11 + r22)
    q = torch.stack([q0, q1, q2, q3], dim=-1)
    return q

def orth6d_to_quat(orth6d):
    return rotmat_to_quat(orth6d_to_rotmat(orth6d))

def to_attributes(x):
    '''
    Args:
        x: torch.Tensor of shape (N, 11) or (B, N, 11), x consists of [f_dc_0, f_dc_1, f_dc_2, opacity, scale_0, scale_1, scale_2, rotmat_00, rotmat_10, rotmat_20, rotmat_01, rotmat_11, rotmat_21]
    Returns:
        torch.Tensor of shape (N, 11) or (B, N, 11)
    '''
    dim = len(x.shape)
    assert dim in [2, 3]
    if dim == 3:
        B, N, _ = x.shape
        x = x.view(-1, 11)
    else:
        B = 1
        N = x.shape[0]
    ret = torch.zeros(B * N, 11).cuda()
    ret[:, :3] = inverse_sigmoid(torch.clamp(x[:, :3] * 0.5 + 0.5, min=0, max=1))
    ret[:, 3] = torch.clamp(x[:, 3] * 0.5 + 0.5, min=0, max=1)
    ret[:, 4:7] = torch.clamp(torch.exp(x[:, 4:7] * 8 - 3), min=0, max=40)
    # ret[:, 7:] = orth6d_to_quat(torch.clamp(x[:,7:], min=-1, max=1))
    ret[:, 7] = torch.clamp(x[:, 7] * 0.25 + 0.75, min=0.5, max=1)
    ret[:, 8:] = torch.clamp(x[:, 8:] * ((0.5**0.5 + 1)/2) + (0.5**0.5 - 1)/2, min=-1, max=(0.5**0.5))
    return ret.view(B, N, 11) if dim == 3 else ret


def normalize_attributes(x):
    '''
    Normalize all attributes to about [-1, 1]
    Args:
        x: np.array of shape (N, 11), x consists of [f_dc_0, f_dc_1, f_dc_2, opacity, scale_0, scale_1, scale_2, quat_0, quat_1, quat_2, quat_3]
    Returns:
        np.array of shape (N, 11)
    '''
    assert len(x.shape) == 2
    ret = np.zeros((x.shape[0], 11))
    ret[:, :3] = (sigmoid(x[:, :3]) - 0.5) / 0.5
    ret[:, 3] = (x[:, 3] - 0.5) / 0.5
    ret[:, 4:7] = (np.log(x[:, 4:7] + 1e-6) + 3) / 8
    # ret[:, 7:] = quat_to_orth6d(x[:, 7:])
    ret[:, 7] = (x[:, 7] - 0.75) / 0.25 # 0.5 < quat_0 < 1
    ret[:, 8:] = (x[:, 8:] - (0.5**0.5 - 1)/2) / ((0.5**0.5 + 1)/2) # -1 < quat_1, quat_2, quat_3 < sqrt(2)/2
    return ret


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


# quat = np.random.randn(10, 4)
# quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
# orth6d = quat_to_orth6d(quat)
# breakpoint()
# tmp = torch.tensor(orth6d, device='cuda')
# q_new = orth6d_to_quat(tmp)
