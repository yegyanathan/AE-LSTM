import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import create_dataset


class TehranDataset(Dataset):

    """
    Tehran Stock Exchange (TSE) dataset
    :Param X_scaled: normalized x
    :Param y_scaled: normalized y
    :Param timestep: number of days to look behind.
    :Param k_days: k days forecast
    
    """
    
    def __init__(self, 
                    X_scaled,
                    y_scaled, 
                    timestep,
                    k_days):
        
        self.X_scaled = X_scaled
        self.y_scaled = y_scaled
        self.timestep = timestep
        self.k_days = k_days

        self.data_X, self.data_Y = create_dataset(X_scaled,
                                                    y_scaled,
                                                    timestep,
                                                    k_days)


    def __len__(self):
        
        return len(self.data_X)


    def __getitem__(self, index):

        timestep_X = self.data_X[index]
        timestep_y = self.data_Y[index]

        timestep_X = torch.tensor(timestep_X)
        timestep_y = torch.tensor(timestep_y)

        return timestep_X.float(), timestep_y.float()




def get_loader(dataset, batch_size, num_workers = 2, shuffle = False, drop_last = True):

    loader = DataLoader(dataset = dataset,
                         batch_size = batch_size, 
                         num_workers = num_workers,
                         shuffle = shuffle, 
                         drop_last = drop_last)

    return loader