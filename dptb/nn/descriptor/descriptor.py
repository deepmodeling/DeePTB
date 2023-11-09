import torch.nn as nn
import torch
from dptb.utils.register import Register

"""this is the register class for descriptors

all descriptors inplemendeted should be a instance of nn.Module class, and provide a forward function that
takes AtomicData class as input, and give AtomicData class as output.

"""
class Descriptor:
    _register = Register()

    def register(target):
        return Descriptor._register.register(target)
    
    def __new__(cls, mode: str, **kwargs):
        if mode in Descriptor._register.keys():
            return Descriptor._register[mode](**kwargs)
        else:
            raise Exception(f"Descriptor mode: {mode} is not registered!")
        
    



