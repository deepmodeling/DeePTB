import torch
from torch.utils.data import Dataset

class EnvSet(Dataset):
    def __init__(self, projenv, type, device='cpu', dtype=torch.float32):
        super(EnvSet, self).__init__()
        self.projenv = torch.tensor(projenv, dtype=dtype).to(device) # np.array([f,i,j,R])
        self.type = type

    def __getitem__(self, item):
        return self.projenv[item]

    def __len__(self):
        return self.projenv.shape[0]