# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from dptb.data import AtomicDataDict
# from typing import Optional, Any, Union, Callable, OrderedDict, List
# from torch import Tensor

# class AtomicPrediction(torch.nn.Module):
#     def __init__(
#         self,
#         config: List[dict],
#         in_field: AtomicDataDict.NODE_FEATURES_KEY,
#         out_field: AtomicDataDict.NODE_FEATURES_KEY,
#         activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
#         if_batch_normalized: bool = False,
#         device: Union[str, torch.device] = torch.device('cpu'), 
#         dtype: Union[str, torch.dtype] = torch.float32,
#         **kwargs
#         ):
        