import numpy as np
import MinkowskiEngine as ME
import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d

def feats_to_coord(p_feats, resolution, mean, std):
    p_feats = p_feats.reshape(mean.shape[0],-1,3)
    p_coord = torch.round(p_feats / resolution)

    return p_coord.reshape(-1,3)

def normalize_pcd(points, mean, std):
    return (points - mean[:,None,:]) / std[:,None,:] if len(mean.shape) == 2 else (points - mean) / std

def unormalize_pcd(points, mean, std):
    return (points * std[:,None,:]) + mean[:,None,:] if len(mean.shape) == 2 else (points * std) + mean

def point_set_to_sparse_refine(p_full, p_part, n_full, n_part, resolution, filename):
    concat_full = np.ceil(n_full / p_full.shape[0])
    concat_part = np.ceil(n_part / p_part.shape[0])

    #if mode == 'diffusion':
    #p_full = p_full[torch.randperm(p_full.shape[0])]
    #p_part = p_part[torch.randperm(p_part.shape[0])]
    #elif mode == 'refine':
    p_full = p_full[torch.randperm(p_full.shape[0])]
    p_full = torch.tensor(p_full.repeat(concat_full, 0)[:n_full])   

    p_part = p_part[torch.randperm(p_part.shape[0])]
    p_part = torch.tensor(p_part.repeat(concat_part, 0)[:n_part])

    #p_feats = ME.utils.batched_coordinates([p_feats], dtype=torch.float32)[:2000]
    
    # after creating the voxel coordinates we normalize the floating coordinates towards mean=0 and std=1
    p_mean, p_std = p_full.mean(axis=0), p_full.std(axis=0)

    return [p_full, p_mean, p_std, p_part, filename]

def point_set_to_sparse(p_full, p_part, n_full, n_part, resolution, filename, p_mean=None, p_std=None):
    concat_part = np.ceil(n_part / p_part.shape[0]) 
    p_part = p_part.repeat(concat_part, 0)
    pcd_part = o3d.geometry.PointCloud()
    pcd_part.points = o3d.utility.Vector3dVector(p_part)
    viewpoint_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_part, voxel_size=10.)
    pcd_part = pcd_part.farthest_point_down_sample(n_part)
    p_part = torch.tensor(np.array(pcd_part.points))
    
    in_viewpoint = viewpoint_grid.check_if_included(o3d.utility.Vector3dVector(p_full))
    p_full = p_full[in_viewpoint] 
    concat_full = np.ceil(n_full / p_full.shape[0])

    p_full = p_full[torch.randperm(p_full.shape[0])]
    p_full = p_full.repeat(concat_full, 0)[:n_full]

    p_full = torch.tensor(p_full)
    
    # after creating the voxel coordinates we normalize the floating coordinates towards mean=0 and std=1
    p_mean = p_full.mean(axis=0) if p_mean is None else p_mean
    p_std = p_full.std(axis=0) if p_std is None else p_std

    return [p_full, p_mean, p_std, p_part, filename]

def numpy_to_sparse_tensor(p_coord, p_feats, p_label=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_coord = ME.utils.batched_coordinates(p_coord, dtype=torch.float32)
    p_feats = torch.vstack(p_feats).float()

    if p_label is not None:
        p_label = ME.utils.batched_coordinates(p_label, device=torch.device('cpu')).numpy()
    
        return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            ), p_label

    return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            )

class SparseSegmentCollation:
    def __init__(self, mode='diffusion'):
        self.mode = mode
        return

    def __call__(self, data):
        # "transpose" the  batch(pt, ptn) to batch(pt), batch(ptn)
        batch = list(zip(*data))

        return {'pcd_full': torch.stack(batch[0]).float(),
            'mean': torch.stack(batch[1]).float(),
            'std': torch.stack(batch[2]).float(),
            'pcd_part' if self.mode == 'diffusion' else 'pcd_noise': torch.stack(batch[3]).float(),
            'filename': batch[4],
        }

class LidarSplatsCollation:
    def __init__(self, num_lidar_points, mode='diffusion'):
        self.num_lidar_points = num_lidar_points
        self.mode = mode
        return

    def __call__(self, data):
        indices_list = []
        splats_lst = []
        points_lst = []
        p_mean_lst = []
        p_std_lst = []
        for i in range(len(data)):
            index = data[i]['index']
            splats = data[i]['splats']
            points = data[i]['lidar']
            splats, p_mean, p_std, points = splats_and_lidar_to_sparse(splats, points, self.num_lidar_points)
            indices_list.append(index)
            splats_lst.append(splats)
            p_mean_lst.append(p_mean)
            p_std_lst.append(p_std)
            points_lst.append(points)

        return {
            'indices': indices_list,
            'splats': torch.stack(splats_lst).float(),
            'mean': torch.stack(p_mean_lst).float(),
            'std': torch.stack(p_std_lst).float(),
            'points' if self.mode == 'diffusion' else 'pcd_noise': torch.stack(points_lst).float(),
        }
    
def splats_and_lidar_to_sparse(splats, points, num_lidar_points):
    pcd_splats = o3d.geometry.PointCloud()
    pcd_splats.points = o3d.utility.Vector3dVector(splats[:, :3])
    viewpoint_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_splats, voxel_size=10.)
    splats = torch.tensor(splats)

    in_viewpoint = viewpoint_grid.check_if_included(o3d.utility.Vector3dVector(points))
    points = points[in_viewpoint]
    concat_lidar = np.ceil(num_lidar_points / points.shape[0]) 
    for _ in range(int(concat_lidar)):
        points_copy = points + np.random.normal(0, 0.1, points.shape)
        points = np.vstack([points, points_copy])
    
    pcd_lidar = o3d.geometry.PointCloud()
    pcd_lidar.points = o3d.utility.Vector3dVector(points)
    pcd_lidar = pcd_lidar.farthest_point_down_sample(num_lidar_points)
    assert len(pcd_lidar.points) == num_lidar_points, f'Original: {points.shape[0]}, Now: {len(pcd_lidar.points)}'
    points = torch.tensor(np.array(pcd_lidar.points))
    
    p_mean = splats[:,:3].mean(axis=0)
    p_std = splats[:,:3].std(axis=0)

    return [splats, p_mean, p_std, points]