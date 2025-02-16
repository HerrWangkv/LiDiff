import torch
from torch.utils.data import Dataset
from natsort import natsorted
import os
import numpy as np
from PIL import Image
import yaml
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion

import warnings
from typing import List
import math
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import open3d as o3d
import time
import json

from lidiff.utils.gsplat_utils import normalize_attributes, render_gaussian
from tqdm import tqdm
import argparse

warnings.filterwarnings('ignore')

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        print(f"Done in {time.time() - start_time:.2f} seconds")
        print("="*6)
        return ret
    return wrapper

#################################################
################## Data loader ##################
#################################################

class NuScenesBase(NuScenes):
    def __init__(self, version, dataroot, split, map_size, canvas_size, verbose=True, N=4, keys=['cameras', 'lidar', 'bev', 'boxes', 'splats'], **kwargs):
        '''
        Args:
            version (str): version of the dataset, e.g. 'v1.0-trainval'
            dataroot (str): directory of the dataset
            verbose (bool): whether to print information of the dataset
            seqs (list): list of scene indices to load
            N (int): number of interpolated frames between keyframes
            keys (list): list of keys to load
        '''
        super().__init__(version=version, dataroot=dataroot, verbose=verbose, **kwargs) 
        self.split = split
        self.map_size = map_size
        self.canvas_size = canvas_size
        if isinstance(self.split, str):
            self.seqs = []
            self.accumulate_seqs()
        else:
            self.seqs = self.split
            print(f"Number of scenes: {len(self.seqs)}")
            print("="*6)
        print(f"Data to load: {keys}")
        self.N = N # Number of interpolated frames between keyframes

        self.lidar = 'LIDAR_TOP'
        self.lidar_data_tokens = [] # Stacked list of LiDAR data tokens
        self.seq_indices = [] # [(seq_idx, timestamp_idx)]
        self.accumulate_lidar_tokens()
        self.cameras = {k:i for i,k in enumerate(['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'])}
        self.img_data_tokens = [[] for _ in range(len(self.cameras))]
        self.camera_calib_tokens = [[] for _ in range(len(self.cameras))]
        self.accumulate_img_and_calib_tokens()

        # https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/map_expansion/map_api.py#L684
        self.bev_colors = dict(drivable_area='#a6cee3',
            road_segment='#1f78b4', # blue
            road_block='#b2df8a', # light green
            lane='#33a02c', # green
            ped_crossing='#fb9a99', # pink
            walkway='#e31a1c', # red
            stop_line='#fdbf6f', # light orange
            carpark_area='#ff7f00', # orange
            # road_divider='#cab2d6', # light purple
            # lane_divider='#6a3d9a', # purple
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

        self.cache_dir = os.path.join(os.path.dirname(self.dataroot), f'cache/{self.version}')
        if 'boxes' in keys:
            self.instance_infos = [None for _ in self.seqs]
            self.frame_instances = [None for _ in self.seqs]
            self.accumulate_objects()
        if 'bev' in keys:
            self.maps = {}
            self.accumulate_maps()

    def accumulate_seqs(self):
        if self.version == 'v1.0-mini' and not self.split.startswith('mini_'):
            self.split = 'mini_' + self.split
        assert self.split in ['train', 'val', 'test', 'mini_train', 'mini_val'], f"Invalid split: {self.split}"
        scene_names = create_splits_scenes()[self.split]
        for i in range(len(self.scene)):
            if self.scene[i]['name'] in scene_names:
                self.seqs.append(i)
        print(f"Current split: {self.split}, number of scenes: {len(self.seqs)}")
        print("="*6)

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
      
    def get_timestamps(self, scene_idx):
        scene_data = self.scene[scene_idx]
        frame_num = (self.N + 1) * (scene_data['nbr_samples'] - 1) + 1
        keyframe_timestamps = self.get_keyframe_timestamps(scene_data)
        assert len(keyframe_timestamps) == scene_data['nbr_samples']
        interpolated_timestamps = self.get_interpolated_timestamps(keyframe_timestamps)
        assert len(interpolated_timestamps) == frame_num
        return interpolated_timestamps

    def find_closest_lidar_tokens(self, scene_data, timestamps: List[int]):
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
    
    @timer
    def accumulate_lidar_tokens(self):
        print('Accumulating LiDAR data tokens...')
        for i, scene_idx in enumerate(tqdm(self.seqs)):
            scene_data = self.scene[scene_idx]
            timestamps = self.get_timestamps(scene_idx)
            closest_lidar_tokens = self.find_closest_lidar_tokens(scene_data, timestamps)
            self.lidar_data_tokens += closest_lidar_tokens
            self.seq_indices += [(i, t) for t in range(len(timestamps))]
        assert len(self.lidar_data_tokens) == len(self.seq_indices)

    @timer
    def accumulate_img_and_calib_tokens(self):        
        print('Accumulating image and calibration tokens...')
        for scene_idx in tqdm(self.seqs):
            scene_data = self.scene[scene_idx]
            timestamps = self.get_timestamps(scene_idx)
            for cam_name in self.cameras.keys():
                img_tokens = self.find_closest_img_tokens(scene_data, timestamps, cam_name)
                self.img_data_tokens[self.cameras[cam_name]] += img_tokens
                calib_tokens = [self.get('sample_data', token)['calibrated_sensor_token'] for token in img_tokens]
                self.camera_calib_tokens[self.cameras[cam_name]] += calib_tokens
                # if cam_name == 'CAM_FRONT':
                #     # save ego poses whenever CAM_FRONT is used
                #     for token in img_tokens:
                #         cam_data = self.get('sample_data', token)
                #         ego_pose_token = cam_data['ego_pose_token']
                #         self.ego_pose_tokens.append(ego_pose_token)
                #         # save transformation matrix from world to CAM_FRONT
                #         calib_data = self.get('calibrated_sensor', calib_tokens[0])
                #         cam_to_ego = np.eye(4)
                #         cam_to_ego[:3, :3] = Quaternion(calib_data['rotation']).rotation_matrix
                #         cam_to_ego[:3, 3] = np.array(calib_data['translation'])
                #         ego_pose = self.get("ego_pose", ego_pose_token)
                #         ego_to_world = np.eye(4)
                #         ego_to_world[:3, :3] = Quaternion(ego_pose['rotation']).rotation_matrix
                #         ego_to_world[:3, 3] = np.array(ego_pose['translation'])
                #         cam_to_world = ego_to_world @ cam_to_ego
                #         self.world_to_cam_front = np.append(self.world_to_cam_front, [np.linalg.inv(cam_to_world)], axis=0)
        assert all([len(tokens) == len(self.seq_indices) for tokens in self.img_data_tokens])
        assert all([len(tokens) == len(self.seq_indices) for tokens in self.camera_calib_tokens])

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
    
    @timer
    def accumulate_objects(self):
        objects_cache_dir = os.path.join(self.cache_dir, 'objects')
        print('Accumulating objects...')
        for i, scene_idx in enumerate(tqdm(self.seqs)):
            os.makedirs(os.path.join(objects_cache_dir, f"{scene_idx}"), exist_ok=True)
            instances_info_cache_path = os.path.join(objects_cache_dir, f'{scene_idx}/instances_info.json')
            frame_instances_cache_path = os.path.join(objects_cache_dir, f'{scene_idx}/frame_instances.json')
            if os.path.exists(instances_info_cache_path) and os.path.exists(frame_instances_cache_path):
                with open(instances_info_cache_path, 'r') as f:
                    instances_info = json.load(f)
                with open(frame_instances_cache_path, 'r') as f:
                    frame_instances = json.load(f)
                self.instance_infos[i] = instances_info
                self.frame_instances[i] = frame_instances
            else:
                scene_data = self.scene[scene_idx]
                instances_info = self.fetch_keyframe_objects(scene_data)
                max_frame_idx = (self.N + 1) * (scene_data['nbr_samples'] - 1)
                instances_info, frame_instances = self.interpolate_boxes(instances_info, max_frame_idx)
                with open(instances_info_cache_path, 'w') as f:
                    json.dump(instances_info, f)
                with open(frame_instances_cache_path, 'w') as f:
                    json.dump(frame_instances, f)
                self.instance_infos[i] = instances_info
                self.frame_instances[i] = frame_instances
    
    @timer
    def accumulate_maps(self):
        maps_cache_dir = os.path.join(self.cache_dir, 'maps')
        print('Accumulating maps...')
        current_seq = -1
        for index in tqdm(range(len(self.seq_indices))):
            seq_idx, frame_idx = self.seq_indices[index]
            scene_idx = self.seqs[seq_idx]
            if os.path.exists(os.path.join(maps_cache_dir, f"{scene_idx}/frame_{frame_idx}.npy")):
                continue
            if (seq_idx != current_seq):
                os.makedirs(os.path.join(maps_cache_dir, f"{scene_idx}"), exist_ok=True)
                scene_data = self.scene[scene_idx]
                if scene_data['log_token'] not in self.maps:
                    log = self.get('log', scene_data['log_token'])
                    map_data = NuScenesMap(self.dataroot, log['location'])
                    self.maps[scene_data['log_token']] = map_data
                map_ = self.maps[scene_data['log_token']]
                current_seq += 1
            ego_pose = self.get_ego_pose(index)
            cam_front_in_world = np.linalg.inv(self.get_world_to_cam_front(index))[:3, 3]
            patch_box = (cam_front_in_world[0], cam_front_in_world[1], self.map_size, self.map_size)
            ypr_rad = Quaternion(ego_pose['rotation']).yaw_pitch_roll
            patch_angle = math.degrees(ypr_rad[0])
            canvas_size = (self.canvas_size, self.canvas_size)
            map_mask = map_.get_map_mask(patch_box, patch_angle, list(self.bev_colors.keys()), canvas_size)
            np.save(os.path.join(maps_cache_dir, f"{scene_idx}/frame_{frame_idx}.npy"), map_mask)


    def is_keyframe(self, index):
        return self.seq_indices[index][1] % (self.N + 1) == 0

    def get_frame_instances(self, index):
        seq_idx, frame_idx = self.seq_indices[index]
        if frame_idx in self.frame_instances[seq_idx]:
            return self.frame_instances[seq_idx][frame_idx]
        else:
            return self.frame_instances[seq_idx][str(frame_idx)]
    
    def get_frame_annotation(self, index, instance_id):
        seq_idx, frame_idx = self.seq_indices[index]
        if instance_id in self.instance_infos[seq_idx]:
            instance_info = self.instance_infos[seq_idx][instance_id]
        else:
            instance_info = self.instance_infos[seq_idx][str(instance_id)]
        class_name = instance_info['class_name']
        if frame_idx in instance_info['frame_annotations']:
            obj_to_world, box_size = instance_info['frame_annotations'][frame_idx]
        else:
            obj_to_world, box_size = instance_info['frame_annotations'][str(frame_idx)]
        return class_name, obj_to_world, box_size
    
    def get_ego_pose(self, index):
        img_token = self.img_data_tokens[0][index]
        cam_data = self.get('sample_data', img_token)
        ego_pose_token = cam_data['ego_pose_token']
        ego_pose = self.get('ego_pose', ego_pose_token)
        return ego_pose
    
    def get_world_to_cam_front(self, index):
        calib_token = self.camera_calib_tokens[0][index]
        calib_data = self.get('calibrated_sensor', calib_token)
        cam_to_ego = np.eye(4)
        cam_to_ego[:3, :3] = Quaternion(calib_data['rotation']).rotation_matrix
        cam_to_ego[:3, 3] = np.array(calib_data['translation'])
        ego_pose = self.get_ego_pose(index)
        ego_to_world = np.eye(4)
        ego_to_world[:3, :3] = Quaternion(ego_pose['rotation']).rotation_matrix
        ego_to_world[:3, 3] = np.array(ego_pose['translation'])
        cam_to_world = ego_to_world @ cam_to_ego
        return np.linalg.inv(cam_to_world)

class NuScenesCameras(Dataset):
    def __init__(self, nusc):
        super().__init__()
        self.nusc = nusc

    def __len__(self):
        return len(self.nusc.img_data_tokens[0])
    
    def __getitem__(self, index):
        ret = []
        for cam in ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']:
            i = self.nusc.cameras[cam]
            filename = self.nusc.get('sample_data', self.nusc.img_data_tokens[i][index])['filename']
            img = np.array(Image.open(os.path.join(self.nusc.dataroot, filename)).convert('RGB')).transpose(2, 0, 1)
            ret.append(img)
        ret = np.array(ret)
        return ret
    
    def next_frame(self, index):
        return self[index+1] if (index + 1 < len(self.nusc.seq_indices)) and (self.nusc.seq_indices[index][0] == self.nusc.seq_indices[index+1][0]) else None
    
    def vis(self, index):
        imgs = self[index]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for i, ax in enumerate(axes.flat):
            img = imgs[i].transpose(1, 2, 0)
            ax.imshow(img)
            ax.set_title(['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'][i])
            ax.axis('off')
        plt.tight_layout()
        plt.savefig("cameras.png", bbox_inches='tight')
        plt.close()


class NuScenesLidar(Dataset):
    def __init__(self, nusc, horizontal_range: List[float] = [-100, 100], vertical_range: List[float] = [-15, 5]):
        super().__init__()
        self.nusc = nusc
        self.min_distance = 1.0 # in LiDAR frame
        self.horizontal_range = horizontal_range # in CAM_FRONT frame
        self.vertical_range = vertical_range # in CAM_FRONT frame

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
        points = self.remove_close(points)
        xyz_in_lidar = points[:, :3]
        xyz_in_world = xyz_in_lidar[:, :3] @ lidar_to_world[:3, :3].T + lidar_to_world[:3, 3]
        world_to_cam_front = self.nusc.get_world_to_cam_front(index)
        xyz_in_cam_front = xyz_in_world @ world_to_cam_front[:3, :3].T + world_to_cam_front[:3, 3]
        points[:, :3] = xyz_in_cam_front
        points = self.remove_far(points)
        return points[:, :3]
    
    def next_frame(self, index):
        return self[index+1] if (index + 1 < len(self.nusc.seq_indices)) and (self.nusc.seq_indices[index][0] == self.nusc.seq_indices[index+1][0]) else None
    
    def remove_close(self, points):
        x_filt = np.abs(points[:, 0]) < self.min_distance
        y_filt = np.abs(points[:, 1]) < self.min_distance
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]
    
    def remove_far(self, points):
        x_filt = np.logical_and(points[:, 0] > self.horizontal_range[0], points[:, 0] < self.horizontal_range[1])
        y_filt = np.logical_and(points[:, 1] > self.vertical_range[0], points[:, 1] < self.vertical_range[1])
        z_filt = np.logical_and(points[:, 2] > self.horizontal_range[0], points[:, 2] < self.horizontal_range[1])
        mask = np.logical_and(np.logical_and(x_filt, y_filt), z_filt)
        return points[mask]

class NuScenesBoxes(Dataset):
    def __init__(self, nusc):
        super().__init__()
        self.nusc = nusc
    
    def __len__(self):
        return len(self.nusc.seq_indices)
    
    def __getitem__(self, index):
        frame_instances = self.nusc.get_frame_instances(index)
        boxes = dict()
        for instance_id in frame_instances:
            class_name, obj_to_world, box_size = self.nusc.get_frame_annotation(index, instance_id)
            obj_to_cam_front = self.nusc.get_world_to_cam_front(index) @ obj_to_world
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
    
    def next_frame(self, index):
        return self[index+1] if (index + 1 < len(self.nusc.seq_indices)) and (self.nusc.seq_indices[index][0] == self.nusc.seq_indices[index+1][0]) else None
    
class NuScenesSplats(Dataset):
    def __init__(self, nusc, model_dir, horizontal_range: List[float] = [-100, 100], vertical_range: List[float] = [-15, 5]):
        super().__init__()
        self.nusc = nusc
        self.model_dir = model_dir
        self.horizontal_range = horizontal_range # in CAM_FRONT frame
        self.vertical_range = vertical_range # in CAM_FRONT frame

    def __len__(self):
        return len(self.nusc.seq_indices)
    
    def __getitem__(self, index):
        attributes = self.load_gt(index)
        ret = np.zeros((attributes.shape[0], 14))
        ret[:,:3] = attributes[:,:3]
        ret[:,3:] = normalize_attributes(attributes[:,3:])
        return ret
    
    def next_frame(self, index):
        return self[index+1] if (index + 1 < len(self.nusc.seq_indices)) and (self.nusc.seq_indices[index][0] == self.nusc.seq_indices[index+1][0]) else None
    
    def load_gt(self, index):
        seq_idx, frame_idx = self.nusc.seq_indices[index]
        scene_idx = self.nusc.seqs[seq_idx]
        splats_path = os.path.join(self.model_dir, f'{scene_idx}', f'frame_{frame_idx}.ply')
        splats = PlyData.read(splats_path)
        x = splats['vertex']['x'] # map size
        assert x.min() >= self.horizontal_range[0] and x.max() <= self.horizontal_range[1], f"Scene: {scene_idx}, Frame: {frame_idx}, x: {x.min()} - {x.max()}"
        y = splats['vertex']['y'] # map size
        assert y.min() >= self.vertical_range[0] and y.max() <= self.vertical_range[1], f"Scene: {scene_idx}, Frame: {frame_idx}, y: {y.min()} - {y.max()}"
        z = splats['vertex']['z'] # map size
        assert z.min() >= self.horizontal_range[0] and z.max() <= self.horizontal_range[1], f"Scene: {scene_idx}, Frame: {frame_idx}, z: {z.min()} - {z.max()}"
        f_dc = np.array([splats['vertex'][f'f_dc_{i}'] for i in range(3)]) 
        opacity = splats['vertex']['opacity'] 
        assert opacity.min() >= 0 and opacity.max() <= 1, f"Scene: {scene_idx}, Frame: {frame_idx}, opacity: {opacity.min()} - {opacity.max()}"
        scale = np.array([splats['vertex'][f'scale_{i}'] for i in range(3)]) 
        assert scale.min() >= 0, f"Scene: {scene_idx}, Frame: {frame_idx}, scale: {scale.min()} - {scale.max()}"
        rotation = np.array([splats['vertex'][f'rot_{i}'] for i in range(4)]) 
        assert rotation[0].min() >= 0.5 and rotation[0].max() <= 1, f"Scene: {scene_idx}, Frame: {frame_idx}, rot_0: {rotation[0].min()} - {rotation[0].max()}"
        assert rotation[1:].min() >= -1 and rotation[1:].max() <= 0.5**0.5, f"Scene: {scene_idx}, Frame: {frame_idx}, rot_1-3: {rotation[1:].min()} - {rotation[1:].max()}"
        attributes = np.vstack([x[None,:], y[None,:], z[None,:], f_dc, opacity[None,:], scale, rotation]).T
        return attributes
    
    def render(self, splats_index, pose_index=None, gaussian=None, save_path=None):
        if pose_index is None:
            pose_index = splats_index
        if gaussian is None:
            gaussian = self.load_gt(splats_index)
        if save_path is None:
            save_path = "rendered.png"
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        for i, ax in enumerate(axes.flat):
            cam = cams[i]
            cam_idx = self.nusc.cameras[cam]
            pose_calib_token = self.nusc.camera_calib_tokens[cam_idx][pose_index]
            pose_calib_data = self.nusc.get('calibrated_sensor', pose_calib_token)
            intrinsics = np.array(pose_calib_data['camera_intrinsic'])
            if cam_idx == 0:
                extrinsics = np.eye(4)
            else:
                pose_cam_to_pose_ego = np.eye(4)
                pose_cam_to_pose_ego[:3, :3] = Quaternion(pose_calib_data['rotation']).rotation_matrix
                pose_cam_to_pose_ego[:3, 3] = np.array(pose_calib_data['translation'])

                splats_cam_front_calib_token = self.nusc.camera_calib_tokens[0][splats_index]
                splats_cam_front_calib_data = self.nusc.get('calibrated_sensor', splats_cam_front_calib_token)
                splats_cam_front_to_splats_ego = np.eye(4)
                splats_cam_front_to_splats_ego[:3, :3] = Quaternion(splats_cam_front_calib_data['rotation']).rotation_matrix
                splats_cam_front_to_splats_ego[:3, 3] = np.array(splats_cam_front_calib_data['translation'])

                pose_ego_to_splats_ego = np.eye(4)
                if splats_index != pose_index:
                    splats_ego_pose = self.nusc.get_ego_pose(splats_index)
                    splats_ego_to_world = np.eye(4)
                    splats_ego_to_world[:3, :3] = Quaternion(splats_ego_pose['rotation']).rotation_matrix
                    splats_ego_to_world[:3, 3] = np.array(splats_ego_pose['translation'])
                    
                    pose_ego_pose = self.nusc.get_ego_pose(pose_index)
                    pose_ego_to_world = np.eye(4)
                    pose_ego_to_world[:3, :3] = Quaternion(pose_ego_pose['rotation']).rotation_matrix
                    pose_ego_to_world[:3, 3] = np.array(pose_ego_pose['translation'])
                    pose_ego_to_splats_ego = np.linalg.inv(splats_ego_to_world) @ pose_ego_to_world


                extrinsics = np.linalg.inv(splats_cam_front_to_splats_ego) @ pose_ego_to_splats_ego @ pose_cam_to_pose_ego
            img = render_gaussian(gaussian, extrinsics, intrinsics)
            ax.imshow(img[0].detach().cpu().numpy())
            ax.set_title(cam)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
            
class NuScenesBev(Dataset):
    def __init__(self, nusc):
        super().__init__()
        self.nusc = nusc

    def __len__(self):
        return len(self.nusc.seq_indices)
    
    def __getitem__(self, index):
        seq_idx, frame_idx = self.nusc.seq_indices[index]
        scene_idx = self.nusc.seqs[seq_idx]
        map_mask_path = os.path.join(self.nusc.cache_dir, 'maps', f"{scene_idx}/frame_{frame_idx}.npy")
        map_mask = np.load(map_mask_path)
        return map_mask
    
    def next_frame(self, index):
        return self[index+1] if (index + 1 < len(self.nusc.seq_indices)) and (self.nusc.seq_indices[index][0] == self.nusc.seq_indices[index+1][0]) else None
    
    def vis(self, index, points=None, boxes=None, splats_pos=None):
        plt.figure(figsize=(15, 15))
        map_mask = self[index]
        map_size = self.nusc.map_size
        canvas_size = self.nusc.canvas_size
        assert map_mask.shape == (len(self.nusc.bev_colors), canvas_size, canvas_size)
        map_img = np.ones((canvas_size, canvas_size, 3))
        for i, layer_color in enumerate(self.nusc.bev_colors.values()):
            rgb_color = np.array(mcolors.hex2color(layer_color))
            map_img *= 1 - map_mask[i][:,:, None]
            map_img += map_mask[i][:,:, None] * rgb_color
        plt.imshow(map_img, origin='lower')
        ratio = canvas_size / map_size
        if points is not None:
            # print("\tPoints:", points[:,0].min(), points[:,0].max(), points[:,1].min(), points[:,1].max(), points[:,2].min(), points[:,2].max())
            plt.scatter(points[:, 2] * ratio + canvas_size//2, -points[:, 0] * ratio + canvas_size//2, s=0.05, c='k')
        if splats_pos is not None:
            # print("\tSplats:", splats_pos[:,0].min(), splats_pos[:,0].max(), splats_pos[:,1].min(), splats_pos[:,1].max(), splats_pos[:,2].min(), splats_pos[:,2].max())
            plt.scatter(splats_pos[:, 2] * ratio + canvas_size//2, -splats_pos[:, 0] * ratio + canvas_size//2, s=0.05, c='yellow', alpha=0.5)
        if boxes is not None:
            for (class_name, obj_corners_in_cam_front) in boxes.values():
                obj_corners_in_cam_front = obj_corners_in_cam_front[:, [2, 0]]
                obj_corners_in_cam_front[:, 1] *= -1
                corners = obj_corners_in_cam_front * ratio + canvas_size//2
                center = corners.mean(axis=0)
                color = np.array(self.nusc.object_colors[class_name]) / 255.0
                plt.plot([corners[0,0], corners[2,0], corners[6,0], corners[4,0], corners[0,0]], [corners[0,1], corners[2,1], corners[6,1], corners[4,1], corners[0,1]], c=color, linewidth=2)
                plt.plot([(corners[0,0] + corners[2,0])/2, center[0]], [(corners[0,1] + corners[2,1])/2, center[1]], c=color, linewidth=2)

        plt.xlim(0, canvas_size)
        plt.ylim(0, canvas_size)
        plt.xticks([0, canvas_size/4, canvas_size/2, canvas_size/4*3, canvas_size], [-map_size/2, -map_size/4, 0, map_size/4, map_size/2])
        plt.yticks([0, canvas_size/4, canvas_size/2, canvas_size/4*3, canvas_size], [-map_size/2, -map_size/4, 0, map_size/4, map_size/2])
        plt.savefig("conditions.png", bbox_inches='tight')
        plt.close()

class NuScenesDataset(Dataset):
    def __init__(self, version, dataroot, splats_dir, map_size, split, N=4, keys=['cameras', 'lidar', 'bev', 'boxes', 'splats'], **kwargs):
        super().__init__()
        canvas_size = kwargs.get('canvas_size', map_size)
        self.nusc = NuScenesBase(version=version, dataroot=dataroot, split=split, map_size=map_size, canvas_size=canvas_size, N=N, keys=keys)
        self.splats = NuScenesSplats(self.nusc, model_dir=splats_dir)
        self.cameras = NuScenesCameras(self.nusc)
        self.lidar = NuScenesLidar(self.nusc)
        self.bev = NuScenesBev(self.nusc)
        self.boxes = NuScenesBoxes(self.nusc)
        self.keys = keys

    def __len__(self):
        return len(self.nusc.seq_indices)
    
    def __getitem__(self, index):
        ret = {}
        ret['index'] = index
        if 'cameras' in self.keys:
            ret['cameras'] = self.cameras[index]
        if 'lidar' in self.keys:
            ret['lidar'] = self.lidar[index]
        if 'bev' in self.keys:
            ret['bev'] = self.bev[index]
        if 'boxes' in self.keys:
            ret['boxes'] = self.boxes[index]
        if 'splats' in self.keys:
            ret['splats'] = self.splats[index]
        return ret
    
    def next_frame(self, index):
        return self[index+1] if (index + 1 < len(self.nusc.seq_indices)) and (self.nusc.seq_indices[index][0] == self.nusc.seq_indices[index+1][0]) else None

    def render(self, splats_index, pose_index):
        self.splats.render(splats_index=splats_index, pose_index=pose_index)

    def vis(self, splats_index, pose_index=None):
        if 'cameras' in self.keys:
            self.cameras.vis(splats_index)
        if 'bev' in self.keys:
            points = self.lidar[splats_index] if 'lidar' in self.keys else None
            boxes = self.boxes[splats_index] if 'boxes' in self.keys else None
            splats_pos = self.splats[splats_index] if 'splats' in self.keys else None
            self.bev.vis(splats_index, points, boxes, splats_pos)
        if 'splats' in self.keys:
            self.splats.render(splats_index=splats_index, pose_index=pose_index)
    

if __name__ == "__main__":
    # Sparse conversion
    # from lidiff.utils.collations import splats_and_lidar_to_sparse
    # dataset = NuScenesDataset(version='v1.0-trainval', 
    #                           dataroot='/storage_local/kwang/nuscenes/raw', 
    #                           splats_dir='/mrtstorage/datasets_tmp/nuscenes_3dgs/framewise_splats/180000_-100_100_-8_2',
    #                           split="train", 
    #                           map_size=200, 
    #                           keys=['lidar', 'splats'],)
    # data = dataset[0]
    # lidar = data['lidar']
    # splats = data['splats']
    # sparse = splats_and_lidar_to_sparse(splats, lidar, 18000) #splats_and_lidar_to_sparse(data['splats'], data['lidar'], 18000)
    # breakpoint()

    # Preprocess map
    # parser = argparse.ArgumentParser(description='NuScenes Dataset Preprocessing')
    # parser.add_argument('--start_idx', '-s', type=int, required=True, help='Start index for preprocessing')
    # parser.add_argument('--end_idx', '-e', type=int, required=True, help='End index for preprocessing')
    # args = parser.parse_args()

    # start_idx = args.start_idx
    # end_idx = args.end_idx
    # dataset = NuScenesDataset(version='v1.0-trainval', 
    #                           dataroot='/storage_local/kwang/nuscenes/raw', 
    #                           splats_dir='/mrtstorage/datasets_tmp/nuscenes_3dgs/framewise_splats/180000_-100_100_-8_2',
    #                           split=list(range(start_idx, end_idx+1)),
    #                           map_size=200, 
    #                           canvas_size=200,)

    # Render splats
    dataset = NuScenesDataset(version='v1.0-mini', 
                            dataroot='/storage_local/kwang/nuscenes/raw', 
                            splats_dir='/storage_local/kwang/nuscenes/framewise_mini',
                            split=[0], 
                            map_size=200, 
                            keys=['cameras', 'splats', 'bev'],)
    dataset.vis(60,70)
    breakpoint()