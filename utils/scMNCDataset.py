from cProfile import label
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# code based on that from https://github.com/daifengwanglab/JAMIE/blob/main/examples/notebooks/scMNC-Visual-Cortical.ipynb 
# data from https://github.com/daifengwanglab/scMNC/tree/main/mouse_motor_cortex

class scMNC(Dataset):
    """Multimodal MNIST Dataset."""

    def __init__(self, dir_data, seed, train=True):
        """
        Args:
            unimodal_datapaths (list): list of paths to weakly-supervised unimodal datasets with samples that
            correspond by index. Therefore the numbers of samples of all datapaths should match.
        """
        super().__init__()
        self.dir_data = dir_data
        self.seed = seed
        self.num_modalities = 2  # for scMNC, we have 2 modalities: gene expression and electrophysiology
        self.label_names = "type"
        filename_exp = os.path.join(
            dir_data, "expression_data.csv"
        )
        filename_feat = os.path.join(
            dir_data, "feature_data.csv"
        )
        filename_labels = os.path.join(
            dir_data, "labels.csv"
        )
        # create dataset if it does not exist
        if not os.path.exists(filename_exp) or not os.path.exists(filename_feat) or not os.path.exists(filename_labels):
            scMNC._create_scmnc_dataset(
                dir_data,
                dir_data
            )
        # load partition info
        self.original_dims = [1302, 39]
        self.modality_names = ["exp", "feat"]
        self.exp_data = pd.read_csv(filename_exp)
        num_samples = self.exp_data.shape[0]
        dp = 0 if train else 1
        partition = [
          x in train_test_split(
            range(num_samples), test_size=0.2,
            random_state = seed
            )[dp] for x in range(num_samples)]
        self.exp_data = preprocessing.scale(
          self.exp_data.loc[partition].to_numpy()
          ).astype(np.float32)
        self.feat_data = preprocessing.scale(
          pd.read_csv(filename_feat).loc[partition].to_numpy()
         ).astype(np.float32) 
        self.labels = pd.read_csv(filename_labels).loc[partition].to_numpy().reshape(-1).astype(np.int64)
        self.num_files = len(self.labels)

    @staticmethod
    def _create_scmnc_dataset(
        dir_data, savepath
    ):
        """Structure the scMNC Dataset with labels under 'savepath'.

        Args:
            dir_data (str): path to directory that contains the scMNC data files.
            savepath (str): path to directory that the dataset will be written to. Will be created if it does not
                exist.
        """

        # load MNIST data
        data1 = pd.read_csv(os.path.join(dir_data, "geneExp_filtered.csv"))
        data2 = pd.read_csv(os.path.join(dir_data, "efeature_filtered.csv"))
        sample_names1 = data1.columns[1:]
        sample_names2 = np.array(data2)[:, 0]
        feature_names1 = data1.iloc[:,0]
        feature_names2 = data2.columns[3:]
        assert (sample_names1 == sample_names2).all()
        data1 = np.transpose(np.array(data1)[:, 1:])
        data2 = np.array(data2)[:, 3:]
        meta = pd.read_csv(
          os.path.join(dir_data, "20200711_patchseq_metadata_mouse.csv")
          )
        meta_names = np.array(meta.columns)
        meta_sid = np.argwhere(meta_names == 'transcriptomics_sample_id')[0][0]
        meta_ttype = np.argwhere(meta_names == 't_type')[0][0]
        meta = np.array(meta)
        meta_idx = [
          np.argwhere(meta[:, meta_sid] == sample_names1[i])[0][0] for i in     range(sample_names1.shape[0])
          ]
        type1 = np.array(
          [x.split(' ')[0] for x in meta[meta_idx, meta_ttype]]
          )

        features = [np.array(feature_names1), np.array(feature_names2)]
        feature_dict = {
          'upstroke_downstroke_ratio_short_square':
            'up-downstroke_ratio_short_square',
            'upstroke_downstroke_ratio_long_square':
              'up-downstroke_ratio_long_square'
              }
        # Preprocessing
        data1 = preprocessing.scale(data1, axis=0)
        data2 = preprocessing.scale(data2, axis=0)
        data1[np.isnan(data1)] = 0  # Replace NaN with average
        data2[np.isnan(data2)] = 0
        dataset = [data1, data2]

        # Replace NULL feature names
        for i in range(len(features)):
            if features[i] is None:
                features[i] = np.array([f'Feature {i}' for i in range(dataset[i].shape[1])])
        
        # One-hot encode labels
        labels = np.unique(type1, return_inverse=True)[1]
        # save labels as csv
        with open(os.path.join(savepath, "labels.csv"), "w") as f:
            writer = pd.DataFrame(data=labels)
            writer.to_csv(f, index=False)
        # save expression data
        with open(os.path.join(savepath, "expression_data.csv"), "w") as f:
            writer = pd.DataFrame(data=dataset[0], columns=features[0])
            writer.to_csv(f, index=False)
        # save feature data
        with open(os.path.join(savepath, "feature_data.csv"), "w") as f:
            writer = pd.DataFrame(data=dataset[1], columns=features[1])
            writer.to_csv(f, index=False) 
        

    def __getitem__(self, index):
        """
        Returns a tuple (data, labels) where data is a dict containing expression and feature data
        for the given index and labels is the corresponding cell type label.
        """

        data_dict = {"exp": torch.from_numpy(self.exp_data[index]),
                     "feat": torch.from_numpy(self.feat_data[index])}
        return (
            data_dict,
            self.labels[index],
        )  # NOTE: for scMNC, labels are shared across modalities, so can take one value

    def __len__(self):
        return self.num_files

class full_dataset_scMNC(scMNC):
    def __init__(self, cfg, training=False):
        super().__init__(cfg.dataset.dir_data, cfg.model.seed,train=training)
        self.data_loader = torch.utils.data.DataLoader(
            self,
            batch_size=3654 if training else 731,
            shuffle=False,
            num_workers=cfg.dataset.num_workers,
            drop_last=False,
        )
        self.batch = next(iter(self.data_loader))
        self.exp_data = self.batch[0]["exp"].numpy() # expression data
        self.feat_data = self.batch[0]["feat"].numpy() # feature data
        self.labels = self.batch[1].numpy() # tensor of labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--savepath-train", type=str, required=True)
    parser.add_argument("--savepath-test", type=str, required=True)
    args = parser.parse_args()  # use vars to convert args into a dict
    print("\nARGS:\n", args)

    # create dataset
    scMNC._create_scmnc_dataset(
        args.dir_data,
        args.savepath_train
    )
    print("Done.")