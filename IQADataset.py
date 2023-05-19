import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset

from PIL import Image


class IQA_dataloader(Dataset):
    def __init__(self, data_dir, csv_path, transform, database):
        self.database = database
        if self.database == 'Koniq10k':
            column_names = ['image_name','c1','c2','c3','c4','c5','c_total','MOS','SD','MOS_zscore']
            tmp_df = pd.read_csv(csv_path,header= 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            self.X_train = tmp_df[['image_name']]
            self.Y_train = tmp_df['MOS_zscore']

        elif self.database == 'FLIVE' or  self.database == 'FLIVE_patch':
            column_names = ['name','mos']
            tmp_df = pd.read_csv(csv_path,header= 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            self.X_train = tmp_df[['name']]
            self.Y_train = tmp_df['mos']

        elif self.database == 'LIVE_challenge':
            column_names = ['image','mos','std']
            tmp_df = pd.read_csv(csv_path,header= 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            self.X_train = tmp_df[['image']]
            self.Y_train = tmp_df['mos']

        elif self.database == 'SPAQ':
            column_names = ['name','mos','brightness','colorfulness','contrast','noisiness','sharpness']
            tmp_df = pd.read_csv(csv_path,header= 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            self.X_train = tmp_df[['name']]
            self.Y_train = tmp_df['mos']
            
        elif self.database == 'BID':
            column_names = ['name','mos']
            tmp_df = pd.read_csv(csv_path,header= 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            self.X_train = tmp_df[['name']]
            self.Y_train = tmp_df['mos']

        self.data_dir = data_dir
        self.transform = transform
        self.length = len(self.X_train)

    def __getitem__(self, index):
        path = os.path.join(self.data_dir,self.X_train.iloc[index,0])

        img = Image.open(path)
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        y_mos = self.Y_train.iloc[index]
        if self.database == 'BID':        
            y_label = torch.FloatTensor(np.array(float(y_mos*20)))
        elif self.database == 'FLIVE' or self.database == 'FLIVE_patch':
            y_label = torch.FloatTensor(np.array(float(y_mos-50)*2))
        else:
            y_label = torch.FloatTensor(np.array(float(y_mos)))

        return img, y_label


    def __len__(self):
        return self.length