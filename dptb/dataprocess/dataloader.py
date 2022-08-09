from torch.utils.data import DataLoader, Dataset


class structDataset(Dataset):
    def __init__(self, structs):
        super(structDataset, self).__init__()
        self.structs = structs

    def __getitem__(self, item):
        '''

        Parameters
        ----------
        item

        Returns
        -------
            {env:(1, Na, Ne, 4), bond: onsite_bond:}
        '''
        pass

    def __len__(self):
        return len(self.structs)