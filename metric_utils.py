import torch
import gc
from typing import Dict, Any, Optional, Callable
import torch.utils.checkpoint as checkpoint
import types
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

class MetricCalculator:
    def __init__(self):
        self.tensor = None
        self.metrics = {}

    def __call__(self, *args, **kwargs):
        return self.metric_fn(*args, **kwargs)

    def calculate_metrics(self, tensor):
        tensor = tensor.permute(1, 2, 3, 0) # shape: [21, 60, 104, 16]
        self.tensor = tensor
        frames, height, width, channels = self.tensor.shape

        self.metrics = {}

        ### Motion Appearance Mean
        motion_appearance_mean_tensor = self.tensor.mean(dim=0).mean(dim=-1) # Shape: [60, 104]
        self.metrics['mean_motion_appearance_mean_tensor'] = motion_appearance_mean_tensor.mean() # to log
        self.metrics['max_motion_appearance_mean_tensor'] = motion_appearance_mean_tensor.max() # to log

        ### Motion Appearance Variance
        motion_appearance_variance_tensor = torch.var(self.tensor, dim=0, unbiased=False).mean(dim=-1) # Shape: [60, 104, 16]
        self.metrics['mean_motion_appearance_variance_tensor'] = motion_appearance_variance_tensor.mean() # to log
        self.metrics['max_motion_appearance_variance_tensor'] = motion_appearance_variance_tensor.max() # to log

        motion_frames_tensor = torch.diff(self.tensor, dim=0)  # Shape: [20, 60, 104, 16]
        abs_motion_frames_tensor = torch.abs(motion_frames_tensor)  # Shape: [20, 60, 104, 16]

        ### Motion Mean
        mean_motion_frames_tensor = motion_frames_tensor.mean(dim=0).mean(dim=-1) # Shape: [60, 104]
        self.metrics['mean_motion_mean_tensor'] = mean_motion_frames_tensor.mean() # to log
        self.metrics['max_motion_mean_tensor'] = mean_motion_frames_tensor.max() # to log

        ### Abs Motion Mean
        mean_abs_motion_frames_tensor = abs_motion_frames_tensor.mean(dim=0).mean(dim=-1) # Shape: [60, 104]
        self.metrics['mean_abs_motion_mean_tensor'] = mean_abs_motion_frames_tensor.mean() # to log
        self.metrics['max_abs_motion_mean_tensor'] = mean_abs_motion_frames_tensor.max() # to log

        ### Motion Variance 
        motion_variance_tensor = torch.var(motion_frames_tensor, dim=0, unbiased=False) # Shape: [60, 104, 16]
        motion_variance_mean_channels = motion_variance_tensor.mean(dim=-1) # Shape: [60, 104]
        self.metrics['mean_motion_variance_tensor'] = motion_variance_mean_channels.mean() # to log
        self.metrics['max_motion_variance_tensor'] = motion_variance_mean_channels.max() # to log

        ### Abs Motion Variance
        abs_motion_variance_tensor = torch.var(abs_motion_frames_tensor, dim=0, unbiased=False) # Shape: [60, 104, 16]
        abs_motion_variance_mean_channels = abs_motion_variance_tensor.mean(dim=-1) # Shape: [60, 104]
        self.metrics['mean_abs_motion_variance_tensor'] = abs_motion_variance_tensor.mean() # to log
        self.metrics['max_abs_motion_variance_tensor'] = abs_motion_variance_mean_channels.max() # to log
        self.metrics['argmax_abs_motion_variance_tensor'] = abs_motion_variance_mean_channels.argmax()

        # quantiles
        abs_motion_variance_mean_channels_flattened = abs_motion_variance_mean_channels.view(-1) # shape: [60*104]
        self.metrics['q90_abs_motion_variance_tensor'] = torch.quantile(abs_motion_variance_mean_channels_flattened, 0.9, dim=-1) # shape: [1]
        self.metrics['q99_abs_motion_variance_tensor'] = torch.quantile(abs_motion_variance_mean_channels_flattened, 0.99, dim=-1) # shape: [1]
        self.metrics['q95_abs_motion_variance_tensor'] = torch.quantile(abs_motion_variance_mean_channels_flattened, 0.95, dim=-1) # shape: [1]
        self.metrics['q75_abs_motion_variance_tensor'] = torch.quantile(abs_motion_variance_mean_channels_flattened, 0.75, dim=-1) # shape: [1]
        

        # top p abs variance
        for p in [0.01, 0.05, 0.1, 0.25]:
            k = int(abs_motion_variance_mean_channels_flattened.shape[0] * p)
            top_k_values, _ = torch.topk(abs_motion_variance_mean_channels_flattened, k)
            self.metrics[f'top{int(p*100)}_abs_variance'] = torch.mean(top_k_values)

        # euler diff
        real_ij_var = motion_variance_tensor[1:-1, 1:-1, :] # [w-2, h-2, c]
        expected_ij_var = torch.zeros_like(real_ij_var) # [w-2, h-2, c]
        for i in range(real_ij_var.shape[0]):
            for j in range(real_ij_var.shape[1]):
                expected_ij_var[i, j, :] = (motion_variance_tensor[i+1, j, :] + \
                                            motion_variance_tensor[i+1, j+2, :] + \
                                            motion_variance_tensor[i, j+1, :] + \
                                            motion_variance_tensor[i+2, j+1, :]) / 4
        
        self.metrics['all_channel_deuler_differentiation_mse'] = torch.mean((expected_ij_var - real_ij_var)**2)
        
        abs_motion_variance_flattened = abs_motion_variance_tensor.view(-1, channels) # shape: [60*104, 16]

        for i in range(16):
            ci_variance = motion_variance_tensor[:, :, i] # shape: [60, 104]
            self.metrics[f'c{i}_nonabs_max_variance'] = ci_variance.max() # to log
            self.metrics[f'c{i}_nonabs_mean_variance'] = ci_variance.mean() # to log
            ci_abs_variance = abs_motion_variance_tensor[:, :, 7] # shape: [60, 104]
            self.metrics[f'c{i}_max_variance'] = ci_abs_variance.max() # to log
            self.metrics[f'c{i}_mean_variance'] = ci_abs_variance.mean() # to log
            self.metrics[f'c{i}_q75_variance'] = torch.quantile(ci_abs_variance, 0.75) # to log
            self.metrics[f'c{i}_q90_variance'] = torch.quantile(ci_abs_variance, 0.9) # to log
            self.metrics[f'c{i}_q95_variance'] = torch.quantile(ci_abs_variance, 0.95) # to log
            self.metrics[f'c{i}_q99_variance'] = torch.quantile(ci_abs_variance, 0.99) # to log
            for p in [0.01, 0.05, 0.1, 0.25]:
                k = int(abs_motion_variance_flattened.shape[0] * p)
                top_k_values, _ = torch.topk(abs_motion_variance_flattened[:,i], k)
                self.metrics[f'c{i}_top{int(p*100)}_abs_variance'] = torch.mean(top_k_values)
            # compute the MSE between the expected and real variance over time
            self.metrics[f'dc{i}dt_euler_differentiation'] = torch.mean((expected_ij_var[:, :, i] - real_ij_var[:, :, i])**2)

        return self.metrics
        
    def get_metrics(self):
        return self.metrics

    def get_metric(self, key):
        return self.metrics.get(key, None)