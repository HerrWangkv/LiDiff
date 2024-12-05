import torch
from torch.utils.data import Dataset
# from lidiff.utils.pcd_preprocess import point_set_to_coord_feats, aggregate_pcds, load_poses
# from lidiff.utils.pcd_transforms import *
# from lidiff.utils.data_map import learning_map
# from lidiff.utils.collations import point_set_to_sparse
from natsort import natsorted
import os
import numpy as np
from PIL import Image
import yaml
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion

import warnings
from typing import List
import math
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

#################################################
################## Data loader ##################
#################################################

class NuScenesDataset(NuScenes):
    def __init__(self, version, dataroot, verbose=True, seqs=None, N=4, **kwargs):
        '''
        Args:
            version (str): version of the dataset, e.g. 'v1.0-trainval'
            dataroot (str): directory of the dataset
            verbose (bool): whether to print information of the dataset
            seqs (list): list of scene indices to load
            N (int): number of interpolated frames between keyframes
        '''
        super().__init__(version=version, dataroot=dataroot, verbose=verbose, **kwargs)        
        self.seqs = seqs if seqs is not None else range(len(self.scene))
        self.N = N # Number of interpolated frames between keyframes
        self.timestamps = [None for _ in self.seqs] # Nested lists of timestamps for each scene
        self.get_timestamps()
        self.lidar = 'LIDAR_TOP'
        self.lidar_data_tokens = [] # Stacked list of LiDAR data tokens
        self.scene_indices = [] # [(scene_idx, timestamp_idx)]
        self.accumulate_lidar_tokens()
        self.cameras = {k:i for i,k in enumerate(['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'])}
        self.img_data_tokens = [[] for _ in range(len(self.cameras))]
        self.camera_calib_tokens = [[] for _ in range(len(self.cameras))]
        self.ego_pose_tokens = [] # Stacked list of ego pose tokens whenever CAM_FRONT is used
        self.world_to_cam_front = np.empty((0, 4, 4)) # Transformation matrix from world to CAM_FRONT
        self.accumulate_img_and_calib_tokens()
        # https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/map_expansion/map_api.py#L684
        self.bev_colors = dict(drivable_area='#a6cee3',
            road_segment='#1f78b4',
            road_block='#b2df8a',
            lane='#33a02c',
            ped_crossing='#fb9a99',
            walkway='#e31a1c',
            stop_line='#fdbf6f',
            carpark_area='#ff7f00',
            road_divider='#cab2d6',
            lane_divider='#6a3d9a',
            # traffic_light='#7e772e'
        )
        self.object_colors = {  # RGB.
            "noise": (0, 0, 0),  # Black.
            "animal": (70, 130, 180),  # Steelblue
            "human.pedestrian.adult": (0, 0, 230),  # Blue
            "human.pedestrian.child": (135, 206, 235),  # Skyblue,
            "human.pedestrian.construction_worker": (100, 149, 237),  # Cornflowerblue
            "human.pedestrian.personal_mobility": (219, 112, 147),  # Palevioletred
            "human.pedestrian.police_officer": (0, 0, 128),  # Navy,
            "human.pedestrian.stroller": (240, 128, 128),  # Lightcoral
            "human.pedestrian.wheelchair": (138, 43, 226),  # Blueviolet
            "movable_object.barrier": (112, 128, 144),  # Slategrey
            "movable_object.debris": (210, 105, 30),  # Chocolate
            "movable_object.pushable_pullable": (105, 105, 105),  # Dimgrey
            "movable_object.trafficcone": (47, 79, 79),  # Darkslategrey
            "static_object.bicycle_rack": (188, 143, 143),  # Rosybrown
            "vehicle.bicycle": (220, 20, 60),  # Crimson
            "vehicle.bus.bendy": (255, 127, 80),  # Coral
            "vehicle.bus.rigid": (255, 69, 0),  # Orangered
            "vehicle.car": (255, 158, 0),  # Orange
            "vehicle.construction": (233, 150, 70),  # Darksalmon
            "vehicle.emergency.ambulance": (255, 83, 0),
            "vehicle.emergency.police": (255, 215, 0),  # Gold
            "vehicle.motorcycle": (255, 61, 99),  # Red
            "vehicle.trailer": (255, 140, 0),  # Darkorange
            "vehicle.truck": (255, 99, 71),  # Tomato
            "flat.driveable_surface": (0, 207, 191),  # nuTonomy green
            "flat.other": (175, 0, 75),
            "flat.sidewalk": (75, 0, 75),
            "flat.terrain": (112, 180, 60),
            "static.manmade": (222, 184, 135),  # Burlywood
            "static.other": (255, 228, 196),  # Bisque
            "static.vegetation": (0, 175, 0),  # Green
            "vehicle.ego": (255, 240, 245)
        }
        self.instance_infos = [None for _ in self.seqs]
        self.frame_instances = [None for _ in self.seqs]
        self.accumulate_objects()

    def get_keyframe_timestamps(self, scene_data):
        """ Get keyframe timestamps from a scene data """
        first_sample_token = scene_data['first_sample_token']
        last_sample_token = scene_data['last_sample_token']
        curr_sample_record = self.get('sample', first_sample_token)
        
        keyframe_timestamps = []
        
        while True:
            # Add the timestamp of the current keyframe
            keyframe_timestamps.append(curr_sample_record['timestamp'])
            
            if curr_sample_record['token'] == last_sample_token:
                break
            
            # Move to the next keyframe
            curr_sample_record = self.get('sample', curr_sample_record['next'])
        
        return keyframe_timestamps
    
    def get_interpolated_timestamps(self, keyframe_timestamps: List[int]):
        """Interpolate timestamps between keyframes."""
        interpolated_timestamps = []
        
        for i in range(len(keyframe_timestamps) - 1):
            start_time = keyframe_timestamps[i]
            end_time = keyframe_timestamps[i + 1]
            
            # Calculate the time step for interpolation
            time_step = (end_time - start_time) / (self.N + 1)
            
            # Add the start timestamp
            interpolated_timestamps.append(start_time)
            
            # Add N interpolated timestamps
            for j in range(1, self.N + 1):
                interpolated_time = start_time + j * time_step
                interpolated_timestamps.append(int(interpolated_time))
        
        # Add the last keyframe timestamp
        interpolated_timestamps.append(keyframe_timestamps[-1])
        
        return interpolated_timestamps
      
    def get_timestamps(self):
        for i, scene_idx in enumerate(self.seqs):
            scene = self.scene[scene_idx]
            frame_num = (self.N + 1) * (scene['nbr_samples'] - 1) + 1
            scene_data = self.get('scene', scene['token'])
            keyframe_timestamps = self.get_keyframe_timestamps(scene_data)
            assert len(keyframe_timestamps) == scene['nbr_samples']
            interpolated_timestamps = self.get_interpolated_timestamps(keyframe_timestamps)
            assert len(interpolated_timestamps) == frame_num
            self.timestamps[i] = interpolated_timestamps

    def find_cloest_lidar_tokens(self, scene_data, timestamps: List[int]):
        """Find the closest LiDAR tokens for given timestamps."""
        first_sample_token = scene_data['first_sample_token']
        first_sample_record = self.get('sample', first_sample_token)
        lidar_token = first_sample_record['data']['LIDAR_TOP']
        lidar_data = self.get('sample_data', lidar_token)
        
        # Collect all LiDAR timestamps and tokens
        lidar_timestamps = []
        lidar_tokens = []
        current_lidar = lidar_data
        while True:
            lidar_timestamps.append(current_lidar['timestamp'])
            lidar_tokens.append(current_lidar['token'])
            if current_lidar['next'] == '':
                break
            current_lidar = self.get('sample_data', current_lidar['next'])
        
        lidar_timestamps = np.array(lidar_timestamps, dtype=np.int64)
        
        # Find closest LiDAR tokens for each timestamp
        closest_tokens = []
        for timestamp in timestamps:
            idx = np.argmin(np.abs(lidar_timestamps - timestamp))
            closest_tokens.append(lidar_tokens[idx])
            
        # DEBUG USAGE: find is there any duplicated tokens
        # if len(closest_tokens) != len(set(closest_tokens)):
        #     duplicates = [token for token, count in Counter(closest_tokens).items() if count > 1]
        #     print(f"\nWARNING: {len(duplicates)} duplicated tokens in lidar")
        
        return closest_tokens
    
    def find_closest_img_tokens(self, scene_data, timestamps: List[int], cam_name):
        """Find the closest image tokens for given timestamps for a specific camera."""
        first_sample_token = scene_data['first_sample_token']
        first_sample_record = self.get('sample', first_sample_token)
        img_token = first_sample_record['data'][cam_name]
        img_data = self.get('sample_data', img_token)
        
        # Collect all image timestamps and tokens for the specified camera
        img_timestamps = []
        img_tokens = []
        current_img = img_data
        while True:
            img_timestamps.append(current_img['timestamp'])
            img_tokens.append(current_img['token'])
            if current_img['next'] == '':
                break
            current_img = self.get('sample_data', current_img['next'])
        
        img_timestamps = np.array(img_timestamps, dtype=np.int64)
        
        # Find closest image tokens for each timestamp
        closest_tokens = []
        for timestamp in timestamps:
            idx = np.argmin(np.abs(img_timestamps - timestamp))
            closest_tokens.append(img_tokens[idx])
        
        # DEBUG USAGE: find is there any duplicated tokens
        # if len(closest_tokens) != len(set(closest_tokens)):
        #     duplicates = [token for token, count in Counter(closest_tokens).items() if count > 1]
        #     print(f"\nWARNING: {len(duplicates)} duplicated tokens in {cam_name}")
        
        return closest_tokens
    
    def accumulate_lidar_tokens(self):
        print('Accumulating LiDAR data tokens...')
        for i, scene_idx in enumerate(self.seqs):
            scene_token = self.scene[scene_idx]['token']
            scene_data = self.get('scene', scene_token)
            timestamps = self.timestamps[i]
            self.lidar_data_tokens += self.find_cloest_lidar_tokens(scene_data, timestamps)
            self.scene_indices += [(i, t) for t in range(len(timestamps))]
        assert len(self.lidar_data_tokens) == len(self.scene_indices)

    def accumulate_img_and_calib_tokens(self):
        print('Accumulating image and calibration tokens...')
        for i, scene_idx in enumerate(self.seqs):
            scene_token = self.scene[scene_idx]['token']
            scene_data = self.get('scene', scene_token)
            timestamps = self.timestamps[i]
            for cam_name in self.cameras.keys():
                img_tokens = self.find_closest_img_tokens(scene_data, timestamps, cam_name)
                self.img_data_tokens[self.cameras[cam_name]] += img_tokens
                calib_tokens = [self.get('sample_data', token)['calibrated_sensor_token'] for token in img_tokens]
                self.camera_calib_tokens[self.cameras[cam_name]] += calib_tokens
                if cam_name == 'CAM_FRONT':
                    # save ego poses whenever CAM_FRONT is used
                    for token in img_tokens:
                        cam_data = self.get('sample_data', token)
                        ego_pose_token = cam_data['ego_pose_token']
                        self.ego_pose_tokens.append(ego_pose_token)
                        # save transformation matrix from world to CAM_FRONT
                        calib_data = self.get('calibrated_sensor', calib_tokens[0])
                        cam_to_ego = np.eye(4)
                        cam_to_ego[:3, :3] = Quaternion(calib_data['rotation']).rotation_matrix
                        cam_to_ego[:3, 3] = np.array(calib_data['translation'])
                        ego_pose = self.get("ego_pose", ego_pose_token)
                        ego_to_world = np.eye(4)
                        ego_to_world[:3, :3] = Quaternion(ego_pose['rotation']).rotation_matrix
                        ego_to_world[:3, 3] = np.array(ego_pose['translation'])
                        cam_to_world = ego_to_world @ cam_to_ego
                        self.world_to_cam_front = np.append(self.world_to_cam_front, [np.linalg.inv(cam_to_world)], axis=0)
        assert all([len(tokens) == len(self.scene_indices) for tokens in self.img_data_tokens])
        assert all([len(tokens) == len(self.scene_indices) for tokens in self.camera_calib_tokens])

    def fetch_keyframe_objects(self, scene_data):
        """Parse and save the objects annotation data."""
        first_sample_token, last_sample_token = scene_data['first_sample_token'], scene_data['last_sample_token']
        curr_sample_record = self.get('sample', first_sample_token)
        key_frame_idx = 0
        
        instances_info = {}
        while True:
            anns = [self.get('sample_annotation', token) for token in curr_sample_record['anns']]
            
            for ann in anns:
                if ann['category_name'] not in self.object_colors:
                    continue
                
                instance_token = ann['instance_token']
                if instance_token not in instances_info:
                    instances_info[instance_token] = {
                        'id': instance_token,
                        'class_name': ann['category_name'],
                        'frame_annotations': {
                            'frame_idx': [],
                            'obj_to_world': [],
                            'box_size': [],
                        }
                    }
                
                # Object to world transformation
                o2w = np.eye(4)
                o2w[:3, :3] = Quaternion(ann['rotation']).rotation_matrix
                o2w[:3, 3] = np.array(ann['translation'])
                
                # Key frames are spaced (N + 1) frames apart in the new sequence
                obj_frame_idx = key_frame_idx * (self.N + 1)
                instances_info[instance_token]['frame_annotations']['frame_idx'].append(obj_frame_idx)
                instances_info[instance_token]['frame_annotations']['obj_to_world'].append(o2w.tolist())
                # convert wlh to lwh
                lwh = [ann['size'][1], ann['size'][0], ann['size'][2]]
                instances_info[instance_token]['frame_annotations']['box_size'].append(lwh)
            
            
            if curr_sample_record['next'] == '' or curr_sample_record['token'] == last_sample_token:
                break
            key_frame_idx += 1
            curr_sample_record = self.get('sample', curr_sample_record['next'])
        
        # Correct ID mapping
        id_map = {}
        for i, (k, v) in enumerate(instances_info.items()):
            id_map[v["id"]] = i

        # Update keys in instances_info
        new_instances_info = {}
        for k, v in instances_info.items():
            new_instances_info[id_map[v["id"]]] = v

        return new_instances_info
    
    def interpolate_boxes(self, instances_info, max_frame_idx):
        """Interpolate object positions and sizes between keyframes."""
        new_instances_info = {}
        new_frame_instances = {}

        for obj_id, obj_info in instances_info.items():
            frame_annotations = obj_info['frame_annotations']
            keyframe_indices = frame_annotations['frame_idx']
            obj_to_world_list = frame_annotations['obj_to_world']
            box_size_list = frame_annotations['box_size']

            new_frame_idx = []
            new_obj_to_world = []
            new_box_size = []

            for i in range(len(keyframe_indices) - 1):
                start_frame = keyframe_indices[i]
                start_transform = np.array(obj_to_world_list[i])
                end_transform = np.array(obj_to_world_list[i + 1])
                start_quat = Quaternion(matrix=start_transform[:3, :3])
                end_quat = Quaternion(matrix=end_transform[:3, :3])
                start_size = np.array(box_size_list[i])
                end_size = np.array(box_size_list[i + 1])

                for j in range(self.N + 1):
                    t = j / (self.N + 1)
                    current_frame = start_frame + j

                    # Interpolate translation
                    translation = (1 - t) * start_transform[:3, 3] + t * end_transform[:3, 3]

                    # Interpolate rotation using Quaternions
                    current_quat = Quaternion.slerp(start_quat, end_quat, t)

                    # Construct interpolated transformation matrix
                    current_transform = np.eye(4)
                    current_transform[:3, :3] = current_quat.rotation_matrix
                    current_transform[:3, 3] = translation

                    # Interpolate box size
                    current_size = (1 - t) * start_size + t * end_size

                    new_frame_idx.append(current_frame)
                    new_obj_to_world.append(current_transform.tolist())
                    new_box_size.append(current_size.tolist())

            # Add the last keyframe
            new_frame_idx.append(keyframe_indices[-1])
            new_obj_to_world.append(obj_to_world_list[-1])
            new_box_size.append(box_size_list[-1])

            # Update instance info
            new_instances_info[obj_id] = {
                'id': obj_info['id'],
                'class_name': obj_info['class_name'],
                'frame_annotations': {new_frame_idx[f]:(new_obj_to_world[f], new_box_size[f]) for f in range(len(new_frame_idx))},
            }

            # Update frame instances
            for frame in new_frame_idx:
                if frame not in new_frame_instances:
                    new_frame_instances[frame] = []
                new_frame_instances[frame].append(obj_id)

        for k in range(max_frame_idx):
            if k not in new_frame_instances:
                new_frame_instances[k] = []
        return new_instances_info, new_frame_instances
    
    def accumulate_objects(self):
        print('Accumulating objects...')
        for i, scene_idx in enumerate(self.seqs):
            scene_token = self.scene[scene_idx]['token']
            scene_data = self.get('scene', scene_token)
            instances_info = self.fetch_keyframe_objects(scene_data)
            max_frame_idx = (self.N + 1) * (scene_data['nbr_samples'] - 1)
            instances_info, frame_instances = self.interpolate_boxes(instances_info, max_frame_idx)
            self.instance_infos[i] = instances_info
            self.frame_instances[i] = frame_instances

    def is_keyframe(self, index):
        return self.scene_indices[index][1] % (self.N + 1) == 0

class NuScenesCameras(Dataset):
    def __init__(self, nusc):
        super().__init__()
        self.nusc = nusc

    def __len__(self):
        return len(self.nusc.img_data_tokens[0])
    
    def __getitem__(self, index):
        ret = torch.tensor([])
        for i in range(len(self.nusc.cameras)):
            filename = self.nusc.get('sample_data', self.nusc.img_data_tokens[i][index])['filename']
            ret = torch.cat((ret, torch.tensor(np.array(Image.open(os.path.join(self.nusc.dataroot, filename)).convert('RGB')).transpose(2,0,1)).unsqueeze(0)), 0)
        return ret
    
    def next_frame(self, index):
        return self[index+1] if (index + 1 < len(self.nusc.scene_indices)) and (self.nusc.scene_indices[index][0] == self.nusc.scene_indices[index+1][0]) else None

class NuScenesLidar(Dataset):
    def __init__(self, nusc):
        super().__init__()
        self.nusc = nusc

    def __len__(self):
        return len(self.nusc.lidar_data_tokens)
    
    def __getitem__(self, index):
        lidar_data = self.nusc.get('sample_data', self.nusc.lidar_data_tokens[index])
        calib_data = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        lidar_to_ego = np.eye(4)
        lidar_to_ego[:3, :3] = Quaternion(calib_data['rotation']).rotation_matrix
        lidar_to_ego[:3, 3] = np.array(calib_data['translation'])
        ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        ego_to_world = np.eye(4)
        ego_to_world[:3, :3] = Quaternion(ego_pose['rotation']).rotation_matrix
        ego_to_world[:3, 3] = np.array(ego_pose['translation'])
        lidar_to_world = ego_to_world @ lidar_to_ego
        filename = lidar_data['filename']
        points = np.fromfile(os.path.join(self.nusc.dataroot, filename), dtype=np.float32).reshape(-1, 5)
        xyz_in_lidar = points[:, :3]
        xyz_in_world = xyz_in_lidar[:, :3] @ lidar_to_world[:3, :3].T + lidar_to_world[:3, 3]
        world_to_cam_front = self.nusc.world_to_cam_front[index]
        xyz_in_cam_front = xyz_in_world @ world_to_cam_front[:3, :3].T + world_to_cam_front[:3, 3]
        points[:, :3] = xyz_in_cam_front
        return torch.tensor(points)
    
    def next_frame(self, index):
        return self[index+1] if (index + 1 < len(self.nusc.scene_indices)) and (self.nusc.scene_indices[index][0] == self.nusc.scene_indices[index+1][0]) else None
    
class NuScenesBev(Dataset):
    def __init__(self, nusc, map_size=500, canvas_size=500):
        super().__init__()
        self.nusc = nusc
        self.map_size = map_size
        self.canvas_size = canvas_size
        self.maps = {}
        self.iterate_maps()

    def iterate_maps(self):
        for scene_idx in self.nusc.seqs:
            scene = self.nusc.scene[scene_idx]
            if scene['log_token'] not in self.maps:
                log = self.nusc.get('log', scene['log_token'])
                map_data = NuScenesMap(self.nusc.dataroot, log['location'])
                self.maps[scene['log_token']] = map_data

    def __len__(self):
        return len(self.nusc.scene_indices)
    
    def __getitem__(self, index):
        scene_idx = self.nusc.scene_indices[index][0]
        scene = self.nusc.scene[scene_idx]
        map_ = self.maps[scene['log_token']]
        ego_pose = self.nusc.get('ego_pose', self.nusc.ego_pose_tokens[index])
        cam_front_in_world = np.linalg.inv(self.nusc.world_to_cam_front[index])[:3, 3]
        patch_box = (cam_front_in_world[0], cam_front_in_world[1], self.map_size//2, self.map_size//2)
        ypr_rad = Quaternion(ego_pose['rotation']).yaw_pitch_roll
        patch_angle = math.degrees(ypr_rad[0])
        canvas_size = (self.canvas_size, self.canvas_size)
        map_mask = map_.get_map_mask(patch_box, patch_angle, list(self.nusc.bev_colors.keys()), canvas_size)
        return torch.tensor(map_mask)
    
    def next_frame(self, index):
        return self[index+1] if (index + 1 < len(self.nusc.scene_indices)) and (self.nusc.scene_indices[index][0] == self.nusc.scene_indices[index+1][0]) else None
    
    def vis(self, index, points=None, boxes=None):
        plt.figure(figsize=(15, 15))
        map_mask = self[index].numpy()
        map_img = np.ones((self.canvas_size, self.canvas_size, 3))
        for i, layer_color in enumerate(self.nusc.bev_colors.values()):
            rgb_color = np.array(mcolors.hex2color(layer_color))
            map_img *= 1 - map_mask[i][:,:, None]
            map_img += map_mask[i][:,:, None] * rgb_color
        plt.imshow(map_img, origin='lower')
        if points is not None:
            ratio = self.canvas_size / self.map_size
            plt.scatter(points[:, 2] * ratio + self.canvas_size//2, -points[:, 0] * ratio + self.canvas_size//2, s=0.05, c='k')
        if boxes is not None:
            for (class_name, obj_corners_in_cam_front) in boxes.values():
                obj_corners_in_cam_front = obj_corners_in_cam_front[:, [2, 0]]
                obj_corners_in_cam_front[:, 1] *= -1
                corners = obj_corners_in_cam_front * ratio + self.canvas_size//2
                center = corners.mean(axis=0)
                color = np.array(self.nusc.object_colors[class_name]) / 255.0
                plt.plot([corners[0,0], corners[2,0], corners[6,0], corners[4,0], corners[0,0]], [corners[0,1], corners[2,1], corners[6,1], corners[4,1], corners[0,1]], c=color, linewidth=2)
                plt.plot([(corners[0,0] + corners[2,0])/2, center[0]], [(corners[0,1] + corners[2,1])/2, center[1]], c=color, linewidth=2)

        plt.xlim(0, self.canvas_size)
        plt.ylim(0, self.canvas_size)
        plt.xticks([0, self.canvas_size//2, self.canvas_size], [-self.map_size//2, 0, self.map_size//2])
        plt.yticks([0, self.canvas_size//2, self.canvas_size], [-self.map_size//2, 0, self.map_size//2])
        plt.savefig('conditions.png', bbox_inches='tight')
        plt.close()

class NuScenesBoxes(Dataset):
    def __init__(self, nusc):
        super().__init__()
        self.nusc = nusc
    
    def __len__(self):
        return len(self.nusc.scene_indices)
    
    def __getitem__(self, index):
        scene_idx, frame_idx = self.nusc.scene_indices[index]
        frame_instances = self.nusc.frame_instances[scene_idx][frame_idx]
        boxes = dict()
        for instance_id in frame_instances:
            instance_info = self.nusc.instance_infos[scene_idx][instance_id]
            class_name = instance_info['class_name']
            obj_to_world, box_size = instance_info['frame_annotations'][frame_idx]
            obj_to_cam_front = self.nusc.world_to_cam_front[index] @ obj_to_world
            obj_corners = np.array([[+box_size[0]/2, +box_size[1]/2, +box_size[2]/2],
                                    [+box_size[0]/2, +box_size[1]/2, -box_size[2]/2],
                                    [+box_size[0]/2, -box_size[1]/2, +box_size[2]/2],
                                    [+box_size[0]/2, -box_size[1]/2, -box_size[2]/2],
                                    [-box_size[0]/2, +box_size[1]/2, +box_size[2]/2],
                                    [-box_size[0]/2, +box_size[1]/2, -box_size[2]/2],
                                    [-box_size[0]/2, -box_size[1]/2, +box_size[2]/2],
                                    [-box_size[0]/2, -box_size[1]/2, -box_size[2]/2]])
            obj_corners_in_cam_front = obj_corners @ obj_to_cam_front[:3, :3].T + obj_to_cam_front[:3, 3]
            boxes[instance_id] = (class_name, obj_corners_in_cam_front)
        return boxes
    
def visualize(idx, m, l=None, b=None):
    points = l[idx] if l is not None else None
    boxes = b[idx] if b is not None else None
    m.vis(idx, points, boxes)

## Example Usage
# n = NuScenesDataset(version='v1.0-mini', dataroot='/tmp/nuscenes_kwang', verbose=True, seqs=range(10))
# c = NuScenesCameras(n)
# l = NuScenesLidar(n)
# m = NuScenesBev(n, map_size=80, canvas_size=1000)
# b = NuScenesBoxes(n)
# print([i for i, t in enumerate(n.scene_indices) if t[1]==0])
# visualize(387, m, l, b)
