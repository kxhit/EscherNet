import os
import math
from pathlib import Path
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import webdataset as wds
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import sys

class ObjaverseDataLoader():
    def __init__(self, root_dir, batch_size, total_view=12, num_workers=4):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        image_transforms = [torchvision.transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])]
        self.image_transforms = torchvision.transforms.Compose(image_transforms)

    def train_dataloader(self):
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=False,
                                image_transforms=self.image_transforms)
        # sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
                             # sampler=sampler)

    def val_dataloader(self):
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=True,
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

def cartesian_to_spherical(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    z = np.sqrt(xy + xyz[:, 2] ** 2)
    theta = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
    # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    return np.array([theta, azimuth, z])

def get_pose(target_RT):
    target_RT = target_RT[:3, :]
    R, T = target_RT[:3, :3], target_RT[:, -1]
    T_target = -R.T @ T
    theta_target, azimuth_target, z_target = cartesian_to_spherical(T_target[None, :])
    # assert if z_target is out of range
    if z_target.item() < 1.5 or z_target.item() > 2.2:
        # print('z_target out of range 1.5-2.2', z_target.item())
        z_target = np.clip(z_target.item(), 1.5, 2.2)
    # with log scale for radius
    target_T = torch.tensor([theta_target.item(), azimuth_target.item(), (np.log(z_target.item()) - np.log(1.5))/(np.log(2.2)-np.log(1.5)) * torch.pi, torch.tensor(0)])
    assert torch.all(target_T <= torch.pi) and torch.all(target_T >= -torch.pi)
    return target_T.numpy()

class ObjaverseData(Dataset):
    def __init__(self,
                 root_dir='.objaverse/hf-objaverse-v1/views',
                 image_transforms=None,
                 total_view=12,
                 validation=False,
                 T_in=1,
                 T_out=1,
                 fix_sample=False,
                 ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.total_view = total_view
        self.T_in = T_in
        self.T_out = T_out
        self.fix_sample = fix_sample

        self.paths = []
        # # include all folders
        # for folder in os.listdir(self.root_dir):
        #     if os.path.isdir(os.path.join(self.root_dir, folder)):
        #         self.paths.append(folder)
        # load ids from .npy so we have exactly the same ids/order
        self.paths = np.load("../scripts/obj_ids.npy")
        # # only use 100K objects for ablation study
        # self.paths = self.paths[:100000]
        total_objects = len(self.paths)
        assert total_objects == 790152, 'total objects %d' % total_objects
        if validation:
            self.paths = self.paths[math.floor(total_objects / 100. * 99.):]  # used last 1% as validation
        else:
            self.paths = self.paths[:math.floor(total_objects / 100. * 99.)]  # used first 99% as training
        print('============= length of dataset %d =============' % len(self.paths))
        self.tform = image_transforms

        downscale = 512 / 256.
        self.fx = 560. / downscale
        self.fy = 560. / downscale
        self.intrinsic = torch.tensor([[self.fx, 0, 128., 0, self.fy, 128., 0, 0, 1.]], dtype=torch.float64).view(3, 3)

    def __len__(self):
        return len(self.paths)

    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        z = np.sqrt(xy + xyz[:, 2] ** 2)
        theta = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
        # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])

        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond

        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_T

    def get_pose(self, target_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        # assert if z_target is out of range
        if z_target.item() < 1.5 or z_target.item() > 2.2:
            # print('z_target out of range 1.5-2.2', z_target.item())
            z_target = np.clip(z_target.item(), 1.5, 2.2)
        # with log scale for radius
        target_T = torch.tensor([theta_target.item(), azimuth_target.item(), (np.log(z_target.item()) - np.log(1.5))/(np.log(2.2)-np.log(1.5)) * torch.pi, torch.tensor(0)])
        assert torch.all(target_T <= torch.pi) and torch.all(target_T >= -torch.pi)
        return target_T

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        try:
            img = plt.imread(path)
        except:
            print(path)
            sys.exit()
        img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img

    def __getitem__(self, index):
        data = {}
        total_view = 12

        if self.fix_sample:
            if self.T_out > 1:
                indexes = range(total_view)
                index_targets = list(indexes[:2]) + list(indexes[-(self.T_out-2):])
                index_inputs = indexes[1:self.T_in+1]   # one overlap identity
            else:
                indexes = range(total_view)
                index_targets = indexes[:self.T_out]
                index_inputs = indexes[self.T_out-1:self.T_in+self.T_out-1] # one overlap identity
        else:
            assert self.T_in + self.T_out <= total_view
            # training with replace, including identity
            indexes = np.random.choice(range(total_view), self.T_in+self.T_out, replace=True)
            index_inputs = indexes[:self.T_in]
            index_targets = indexes[self.T_in:]
        filename = os.path.join(self.root_dir, self.paths[index])

        color = [1., 1., 1., 1.]

        try:
            input_ims = []
            target_ims = []
            target_Ts = []
            cond_Ts = []
            for i, index_input in enumerate(index_inputs):
                input_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_input), color))
                input_ims.append(input_im)
                input_RT = np.load(os.path.join(filename, '%03d.npy' % index_input))
                cond_Ts.append(self.get_pose(input_RT))
            for i, index_target in enumerate(index_targets):
                target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
                target_ims.append(target_im)
                target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
                target_Ts.append(self.get_pose(target_RT))
        except:
            print('error loading data ', filename)
            filename = os.path.join(self.root_dir, '0a01f314e2864711aa7e33bace4bd8c8')  # this one we know is valid
            input_ims = []
            target_ims = []
            target_Ts = []
            cond_Ts = []
            # very hacky solution, sorry about this
            for i, index_input in enumerate(index_inputs):
                input_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_input), color))
                input_ims.append(input_im)
                input_RT = np.load(os.path.join(filename, '%03d.npy' % index_input))
                cond_Ts.append(self.get_pose(input_RT))
            for i, index_target in enumerate(index_targets):
                target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
                target_ims.append(target_im)
                target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
                target_Ts.append(self.get_pose(target_RT))

        # stack to batch
        data['image_input'] = torch.stack(input_ims, dim=0)
        data['image_target'] = torch.stack(target_ims, dim=0)
        data['pose_out'] = torch.stack(target_Ts, dim=0)
        data['pose_in'] = torch.stack(cond_Ts, dim=0)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)