import numpy as np
import pandas as pd
import torch
import torchvision
from utils import utils_pytorch

class FER2013PyTorch(torch.utils.data.Dataset):
    def __init__(self, df, cfg_OnsuNet = False, transforms=None):
        self.df = df
        self.cfg_OnsuNet = cfg_OnsuNet
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.loc[index]
        image, label = np.array([x.split() for x in self.df.loc[index, ['pixels']]]), row['emotion']
        image = np.asarray(image).astype(np.uint8).reshape(48,48)
        if self.cfg_OnsuNet == False:
            image = np.stack((image,)*3, axis=-1)
       
        if self.transforms:
            image = self.transforms(image)
            
        return image.clone().detach(), label

def load_fer2013_pytorch(filepath, device, upsample = None, batch_size=64, stats=([0.5],[0.5]), cfg_OnsuNet = False):
    df = pd.read_csv(filepath)

    train_df = df[df['Usage']=='Training']
    valid_df = df[df['Usage']=='PublicTest']
    test_df = df[df['Usage']=='PrivateTest']

    valid_df = valid_df.reset_index(drop=True) 
    test_df = test_df.reset_index(drop = True)

    train_transformations = torchvision.transforms.Compose([   
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((48 if upsample == None else upsample * 48, 48 if upsample == None else upsample * 48)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(*stats,inplace=True)
    ])
    valid_transformations = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((48 if upsample == None else upsample * 48, 48 if upsample == None else upsample * 48)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(*stats,inplace=True)
    ])

    train_ds = FER2013PyTorch(train_df, cfg_OnsuNet, train_transformations)
    valid_ds = FER2013PyTorch(valid_df, cfg_OnsuNet, valid_transformations)
    test_ds = FER2013PyTorch(test_df, cfg_OnsuNet, valid_transformations)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size*2, num_workers=2, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size*2, num_workers=2, pin_memory=True)

    torch.cuda.empty_cache()

    train_dl = utils_pytorch.DeviceDataLoader(train_dl, device)
    valid_dl = utils_pytorch.DeviceDataLoader(valid_dl, device)
    test_dl = utils_pytorch.DeviceDataLoader(test_dl, device)

    return train_dl, valid_dl, test_dl