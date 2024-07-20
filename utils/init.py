#  Copyright (C) 2021 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


# the function has been modified to handle more the 2-D masks, in our case, handling the mask for the convolutional layers(3-D), 
# if this function is to be used for conv2d layers then further modifications is to be done 


import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init

# TODO: Expand to support tensors larger than 2 dimensions (Done) 
def random_restrict_fanin(mask: Tensor, fan_in: int) -> Tensor:
    """
    'random_restrict_fanin':  is a standalone function that can be applied to any tensor to impose a sparsity pattern.

    Restrict the fan-in of a given mask tensor by setting a specified number of elements to 1 in each slice.

    Parameters:
    mask (Tensor): The mask tensor to be modified. Can be 2D or 3D.
    fan_in (int): The number of elements to set to 1 in each vector (or 1D slice) of the mask.

    Returns:
    Tensor: The modified mask tensor with the specified sparsity pattern.
    """
    dimensions = mask.dim()
    init.constant_(mask, 0.0)  # Initialize all elements of the mask to 0.
    
    if dimensions == 2:
        # If the mask is 2D, set fan_in elements in each row to 1.
        vector_size, num_vectors = mask.shape
        for i in range(num_vectors):
            indices = torch.randperm(vector_size)[:fan_in]  # select indices to set to 1 (Done Randomly).
            mask[i][indices] = 1

    elif dimensions == 3:
        # If the mask is 3D, set fan_in elements in each 1D slice of the mask. Modified case to allow the function to be used for the conv layers 
        out_channels, in_channels, kernel_size = mask.shape
        for i in range(out_channels):
            for j in range(in_channels):
                indices = torch.randperm(kernel_size)[:fan_in]  # select indices to set to 1 (Done Randomly).
                mask[i, j, indices] = 1

    else:
        raise ValueError("Unsupported mask shape: %s" % (str(mask.shape)))
    
    return mask

