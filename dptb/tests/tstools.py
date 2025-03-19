import torch
import numpy as np
from collections import Counter

def compare_tensors_as_sets(tensor1, tensor2):
    """
    Compare whether two tensors contain the same rows, regardless of order.
    Args:
    tensor1 (torch.Tensor or array-like): The first tensor with shape (n, m).
    tensor2 (torch.Tensor or array-like): The second tensor with shape (n, m).
    Returns:
    bool: True if the two tensors contain the same set of rows, otherwise False.
    """
    if not isinstance(tensor1, torch.Tensor):
        tensor1 = torch.tensor(tensor1)
    # elif has grad:
    elif tensor1.requires_grad:
        tensor1 = tensor1.detach()
    if not isinstance(tensor2, torch.Tensor):
        tensor2 = torch.tensor(tensor2)
    elif tensor2.requires_grad:
        tensor2 = tensor2.detach()

    if tensor1.dim() == 1:
        tensor1 = tensor1.unsqueeze(1)
    elif tensor1.dim() > 2:
        raise ValueError('The first tensor must have rank <=2')
    if tensor2.dim() == 1:
        tensor2 = tensor2.unsqueeze(1)
    elif tensor2.dim() > 2:
        raise ValueError('The second tensor must have rank <=2')
    
    if tensor1.shape!= tensor2.shape:
        return False
    
    set1 = {tuple(row.tolist()) for row in tensor1}
    set2 = {tuple(row.tolist()) for row in tensor2}
    
    return set1 == set2

def compare_tensors_as_sets_float(tensor1, tensor2, precision=6):
    """
    Compare whether two tensors contain the same rows, regardless of order.
    Args:
    tensor1 (torch.Tensor or array-like): The first tensor with shape (n, m).
    tensor2 (torch.Tensor or array-like): The second tensor with shape (n, m).
    Returns:
    bool: True if the two tensors contain the same set of rows, otherwise False.
    """
    # Ensure that the input is a tensor
    if not isinstance(tensor1, torch.Tensor):
        tensor1 = torch.tensor(tensor1)
    elif tensor1.requires_grad:
        tensor1 = tensor1.detach()
    if not isinstance(tensor2, torch.Tensor):
        tensor2 = torch.tensor(tensor2)
    elif tensor2.requires_grad:
        tensor2 = tensor2.detach()
    # Ensure that the input is rank 2
    if tensor1.dim() == 1:
        tensor1 = tensor1.unsqueeze(1)
    elif tensor1.dim() > 2:
        raise ValueError('The first tensor must have rank <=2')
    if tensor2.dim() == 1:
        tensor2 = tensor2.unsqueeze(1)
    elif tensor2.dim() > 2:
        raise ValueError('The second tensor must have rank <=2')
    
    if tensor1.shape != tensor2.shape:
        return False
    
    # transform tensors to NumPy arrays and round to the specified precision
    arr1 = np.round(tensor1.numpy(), precision)
    arr2 = np.round(tensor2.numpy(), precision)
    
    # use a structured array for more efficient comparison
    dtype = [(f'f{i}', 'float64') for i in range(arr1.shape[1])]
    
    # Convert arrays to structured arrays
    structured_arr1 = np.array([tuple(row) for row in arr1], dtype=dtype)
    structured_arr2 = np.array([tuple(row) for row in arr2], dtype=dtype)
    
    # sort and compare unique values
    unique1 = np.unique(structured_arr1, return_counts=True)
    unique2 = np.unique(structured_arr2, return_counts=True)
    
    # compare the unique values and their counts
    
    return (unique1[0].shape == unique2[0].shape and 
            np.all(unique1[0] == unique2[0]) and 
            np.all(unique1[1] == unique2[1]))