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

def get_pose(transformation):
    # transformation: 4x4
    return transformation

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

    def get_pose(self, transformation):
        # transformation: 4x4
        return transformation


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
                cond_Ts.append(self.get_pose(np.concatenate([input_RT[:3, :], np.array([[0, 0, 0, 1]])], axis=0)))
            for i, index_target in enumerate(index_targets):
                target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
                target_ims.append(target_im)
                target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
                target_Ts.append(self.get_pose(np.concatenate([target_RT[:3, :], np.array([[0, 0, 0, 1]])], axis=0)))
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
                cond_Ts.append(self.get_pose(np.concatenate([input_RT[:3, :], np.array([[0, 0, 0, 1]])], axis=0)))
            for i, index_target in enumerate(index_targets):
                target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
                target_ims.append(target_im)
                target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
                target_Ts.append(self.get_pose(np.concatenate([target_RT[:3, :], np.array([[0, 0, 0, 1]])], axis=0)))

        # stack to batch
        data['image_input'] = torch.stack(input_ims, dim=0)
        data['image_target'] = torch.stack(target_ims, dim=0)
        data['pose_out'] = np.stack(target_Ts)
        data['pose_out_inv'] = np.linalg.inv(np.stack(target_Ts)).transpose([0, 2, 1])
        data['pose_in'] = np.stack(cond_Ts)
        data['pose_in_inv'] = np.linalg.inv(np.stack(cond_Ts)).transpose([0, 2, 1])
        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)