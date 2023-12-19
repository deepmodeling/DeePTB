import torch.nn as nn
import torch
from dptb.utils.register import Register

"""this is the register class for descriptors

all descriptors inplemendeted should be a instance of nn.Module class, and provide a forward function that
takes AtomicData class as input, and give AtomicData class as output.

"""
class Embedding:
    _register = Register()

    def register(target):
        return Embedding._register.register(target)
    
    def __new__(cls, method: str, **kwargs):
        if method in Embedding._register.keys():
            return Embedding._register[method](**kwargs)
        else:
            raise Exception(f"Descriptor mode: {method} is not registered!")
        

        
    



