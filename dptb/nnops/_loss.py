import torch.nn as nn
import torch
from dptb.utils.register import Register

"""this is the register class for descriptors

all descriptors inplemendeted should be a instance of nn.Module class, and provide a forward function that
takes AtomicData class as input, and give AtomicData class as output.

"""
class Loss:
    _register = Register()

    def register(target):
        return Loss._register.register(target)
    
    def __new__(cls, method: str, **kwargs):
        if method in Loss._register.keys():
            return Loss._register[method](**kwargs)
        else:
            raise Exception(f"Loss method: {method} is not registered!")


@Loss.register("eig")
class EigLoss(nn.Module):
    def __init__(self, **kwargs):
        super(EigLoss, self).__init__()
        self.loss = nn.MSELoss()