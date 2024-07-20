# Imports 
import torch
from torch import nn 
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter


# defining the 2D mask for linear layers (creating a tensor with all values =1)
class DenseMask2D(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(DenseMask2D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mask = Parameter(torch.Tensor(self.out_features, self.in_features), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.constant_(self.mask, 1.0)

    def forward(self):
        return self.mask

    def print_mask_size(self):
        print(f"Mask size in DenseMask2D: {self.mask.size()}")

    
# Sparsity mask for linear layers (the fanin argument defined the number of nonzero elements) 
class RandomFixedSparsityMask2D(nn.Module):
    def __init__(self, in_features: int, out_features: int, fan_in: int) -> None:
        super(RandomFixedSparsityMask2D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fan_in = fan_in
        self.mask = Parameter(torch.Tensor(self.out_features, self.in_features), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.constant_(self.mask, 0.0)
        for i in range(self.out_features):
            x = torch.randperm(self.in_features)[:self.fan_in]
            self.mask[i][x] = 1

    def forward(self):
        return self.mask
    
    def print_mask_size(self):
        print(f"Mask size in RandomFixedSparsityMask2D: {self.mask.size()}")
    
    def count_zero_elements(self):
        zero_count = torch.sum(self.mask == 0).item()
        print(f"Number of zero elements in the mask: {zero_count}")
        return zero_count

    def count_total_elements(self):
        total_elements = self.mask.numel()
        print(f"Total number of elements in the mask: {total_elements}")
        return total_elements


# Dense mask for 1D convolutional layers
class Conv1DMask(nn.Module):
    def __init__(self, out_channels: int, in_channels: int, kernel_size: int) -> None:
        super(Conv1DMask, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.mask = Parameter(torch.Tensor(self.out_channels, self.in_channels, kernel_size), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.constant_(self.mask, 1.0)

    def forward(self):
        return self.mask
    
    def print_mask_size(self):
        print(f"Mask size in Conv1DMask: {self.mask.size()}")


# Sparsoty mask for convolutional layers 
# (The fanin argument has a maximum value when fanin == kernel size and in that case sparsity is 0%)  
class RandomFixedSparsityConv1DMask(nn.Module):
    def __init__(self, out_channels: int, in_channels: int, kernel_size: int, fan_in: int) -> None:
        super(RandomFixedSparsityConv1DMask, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.fan_in = fan_in
        self.mask = Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.constant_(self.mask, 0.0)
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                x = torch.randperm(self.kernel_size)[:self.fan_in]
                self.mask[i][j][x] = 1

    def forward(self):
        return self.mask

    def print_mask_size(self):
        print(f"Mask size in RandomFixedSparsityConv1DMask: {self.mask.size()}")
    
    def count_zero_elements(self):
        zero_count = torch.sum(self.mask == 0).item()
        print(f"Number of zero elements in the mask: {zero_count}")
        return zero_count

    def count_total_elements(self):
        total_elements = self.mask.numel()
        print(f"Total number of elements in the mask: {total_elements}")
        return total_elements