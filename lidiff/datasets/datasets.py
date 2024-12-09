import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from lidiff.datasets.dataloader.SemanticKITTITemporal import TemporalKITTISet
from lidiff.datasets.dataloader.NuScenes import NuScenesDataset
from lidiff.utils.collations import SparseSegmentCollation, LidarSplatsCollation
import warnings

warnings.filterwarnings('ignore')

__all__ = ['TemporalKittiDataModule']

class TemporalKittiDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):
        collate = SparseSegmentCollation()

        data_set = TemporalKITTISet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['train'],
            split=self.cfg['data']['split'],
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            max_range=self.cfg['data']['max_range'],
            dataset_norm=self.cfg['data']['dataset_norm'],
            std_axis_norm=self.cfg['data']['std_axis_norm'])
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def val_dataloader(self, pre_training=True):
        collate = SparseSegmentCollation()

        data_set = TemporalKITTISet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['validation'],
            split='validation',
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            max_range=self.cfg['data']['max_range'],
            dataset_norm=self.cfg['data']['dataset_norm'],
            std_axis_norm=self.cfg['data']['std_axis_norm'])
        loader = DataLoader(data_set, batch_size=1,#self.cfg['train']['batch_size'],
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def test_dataloader(self):
        collate = SparseSegmentCollation()

        data_set = TemporalKITTISet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['validation'],
            split='validation',
            resolution=self.cfg['data']['resolution'],
            num_points=self.cfg['data']['num_points'],
            max_range=self.cfg['data']['max_range'],
            dataset_norm=self.cfg['data']['dataset_norm'],
            std_axis_norm=self.cfg['data']['std_axis_norm'])
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
                             num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

class TemporalNuScenesDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
    
    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):
        assert self.cfg['data']['horizontal_range'][0] + self.cfg['data']['horizontal_range'][1] == 0, 'Horizontal range must be symmetric'
        map_size = self.cfg['data']['horizontal_range'][1] - self.cfg['data']['horizontal_range'][0]
        data_set = NuScenesDataset(version='v1.0-trainval', 
                                   dataroot=self.cfg['data']['data_dir'], 
                                   splats_dir=self.cfg['data']['splats_dir'],
                                   map_size=map_size,
                                   seqs=self.cfg['data']['train'],
                                   keys=['lidar', 'splats'],)

        collate = LidarSplatsCollation(num_lidar_points=self.cfg['data']['num_lidar_points'])
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader
    
    def val_dataloader(self, pre_training=True):
        raise NotImplementedError
    
    def test_dataloader(self):
        raise NotImplementedError


dataloaders = {
    'KITTI': TemporalKittiDataModule,
    'NuScenes': TemporalNuScenesDataModule,
}

