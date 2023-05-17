import os
import random
import torch

import imageio.v2 as imageio
import numpy as np
from torch.utils.data.dataset import Dataset


class MSMRIDataset(Dataset):
    def __init__(self, directory, transform):
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform
        self.seqtypes = ['flair', 'mprage', 't2', 'pd', 'label1', 'label2']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('_')[0]
                    # print(seqtype)
                    datapoint[seqtype] = os.path.join(root, f)
                    # print(datapoint)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)
        print("Found {} data points".format(len(self.database)), "in directory", directory)

    def _prep_label(self, label):
        label = label[:, :, 0]
        label = label.astype(np.uint8)
        assert all([cl > 240 or cl < 20 for cl in np.unique(label)]), f"{np.unique(label)}"
        label[label < 20] = 0
        label[label > 240] = 1
        # cf. ms_mri_proposed.py: transform + to_tensor
        return label[:, :, None].transpose((2, 0, 1))

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            img = imageio.imread(filedict[seqtype])
            # check: tf.to_tensor or torch.tensor or none?
            out.append(img)
            # out.append(torch.tensor(img))
        # out = np.stack(out)

        image = np.concatenate(out[0:4], axis=2)
        image = image[:, :, ::3] / 255 - 0.5  # normalize to [-0.5, 0.5]
        image = image.astype(np.float32)
        image = image.transpose(2, 0, 1)

        # image = out[0]
        # image = torch.unsqueeze(image, 0)
        # image = torch.cat((image, image, image, image), 0)
        # label = out[random.randint(4, 5)]
        label1 = self._prep_label(out[4])
        label2 = self._prep_label(out[5])
        label = np.concatenate([label1, label2], axis=0)

        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label

    def __len__(self):
        return len(self.database)


def save_npy(data, out_dir):
    stages = ['train', 'val', 'test']
    types = ['images', 'labels']
    for s in stages:
        for t in types:
            if not os.path.exists(os.path.join(out_dir, s, t)):
                os.makedirs(os.path.join(out_dir, s, t))
            cnt = 0
            for index, image in enumerate(data[s][t]):
                fl = os.path.join(out_dir, s, t, str(index) + '.npy')
                if cnt % 100 == 0:
                    print(cnt, ":", fl)
                np.save(fl, image)
                cnt += 1

    return


def load_and_process_data(data_root):
    data = {
        "train": {
            "images": [],
            "labels": []
        },
        "val": {
            "images": [],
            "labels": []
        },
        "test": {
            "images": [],
            "labels": []
        }
    }

    # training
    ds_train = MSMRIDataset(data_root + "/training", None)
    samples = len(ds_train)
    print(f"Training: {samples}")
    for sample in range(samples):
        image, labels = ds_train[sample]
        data["train"]["images"].append(image)
        data["train"]["labels"].append(labels)

    # val
    ds_val = MSMRIDataset(data_root + "/testing", None)
    max_size = 64
    ds_val, _ = torch.utils.data.random_split(ds_val, [max_size, len(ds_val) - max_size], generator=torch.Generator().manual_seed(1))
    samples = len(ds_val)
    print(f"Validation: {samples}")
    assert samples == max_size
    for sample in range(samples):
        image, labels = ds_val[sample]
        data["val"]["images"].append(image)
        data["val"]["labels"].append(labels)

    # testing
    ds_test = MSMRIDataset(data_root + "/testing", None)
    samples = len(ds_test)
    print(f"Testing: {samples}")
    for sample in range(samples):
        image, labels = ds_test[sample]
        data["test"]["images"].append(image)
        data["test"]["labels"].append(labels)

    return data


if __name__ == '__main__':
    # data_root = '../data/data_lidc.pickle'
    # preproc_folder = '../data/preproc'
    # npy_dir = '../data/lidc_npy'
    data_root = '/storage/homefs/lz20w714/aimi_storage/lukaszbinden/datasets/ms_mri/data'
    # data_root = '/home/lukas/datasets/ms_mri/data'
    npy_dir = 'data/msmri_npy'

    save_npy(load_and_process_data(data_root), npy_dir)

