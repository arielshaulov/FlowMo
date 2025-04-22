import torch
import gc
from typing import Dict, Any, Optional, Callable
import torch.utils.checkpoint as checkpoint
import types
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"


class ManualGradientCheckpointer:
    """
    Adds gradient checkpointing to models that don't have native support.
    Ensures proper gradient flow.
    """
    def __init__(self, model):
        self.model = model
        self.original_forward = model.forward
        self.is_active = False
    
    def enable(self):
        """Enable gradient checkpointing by patching the forward method"""
        if self.is_active:
            return
            
        def checkpointed_forward(self_model, latent_model_input, t, **kwargs):
            """Wrapped forward function with gradient checkpointing"""
            # Get the first element of latent_model_input
            latent = latent_model_input[0]
            
            def custom_forward(x, timestep):
                # Create list format expected by original forward
                latent_list = [x]
                
                # Call original forward
                output = self.original_forward(latent_list, t=timestep, **kwargs)
                
                # Properly handle output format
                if isinstance(output, (list, tuple)):
                    result = output[0]
                else:
                    result = output
                    
                return result
                
            # Apply checkpointing with proper preserve_rng_state flag
            return checkpoint.checkpoint(
                custom_forward, 
                latent,
                t,
                preserve_rng_state=True  # Important for consistent results
            )
            
        # Replace the forward method
        self.model.forward = types.MethodType(checkpointed_forward, self.model)
        self.is_active = True
        
    def disable(self):
        """Restore original forward method"""
        if not self.is_active:
            return
            
        # Restore original forward method
        self.model.forward = self.original_forward
        self.is_active = False


class MotionVarianceOptimizer:
    """
    Memory-efficient implementation of motion variance optimization during diffusion sampling.
    Uses manual gradient checkpointing for models without native support.
    """
    def __init__(self, 
                 iterations=3,
                 lr=0.001,
                 start_after_steps=15, 
                 apply_frequency=5,
                 use_softmax_mean=True,
                 temperature=10.0):
        """
        Args:
            iterations: Number of optimization iterations per step
            lr: Learning rate for optimization
            start_after_steps: Only start optimization after this many denoising steps
            apply_frequency: Apply optimization every N steps
            use_softmax_mean: Use softmax weighting for mean calculation
            temperature: Temperature for softmax weighting
        """
        self.iterations = iterations
        self.lr = lr
        self.start_after_steps = start_after_steps
        self.apply_frequency = apply_frequency
        self.use_softmax_mean = use_softmax_mean
        self.temperature = temperature
        
    def calculate_motion_variance(self, noise_pred):
        """
        Calculate motion variance in a differentiable way from noise prediction.
        Handles 3D tensors with shape [C, F, H*W].
        
        Args:
            noise_pred: Tensor of shape [C, F, H*W] - model's noise prediction
            
        Returns:
            motion_variance_metric: Scalar tensor for optimization
        """
        # Get dimensions
        if len(noise_pred.shape) == 3:  # [C, F, H*W]
            channels, frames, flattened_hw = noise_pred.shape
            
            # Reshape to [F, H*W, C] for easier frame manipulation
            latent_frames = noise_pred.permute(1, 2, 0)  # Shape: [frames, flattened_hw, channels]
            
            # Calculate frame-by-frame differences
            motion_frames = []
            for frame_idx in range(1, frames):
                current_frame = latent_frames[frame_idx]       # Shape: [flattened_hw, channels]
                previous_frame = latent_frames[frame_idx - 1]  # Shape: [flattened_hw, channels]
                motion_frame = torch.abs(current_frame - previous_frame)
                motion_frames.append(motion_frame)
            
            if not motion_frames:
                # No motion to calculate for single-frame generation
                return torch.tensor(0.0, device=noise_pred.device, requires_grad=True)
            
            motion_frames_tensor = torch.stack(motion_frames, dim=0)  # Shape: [frames-1, flattened_hw, channels]
            
            # Calculate variance across frames dimension
            motion_variance = torch.var(motion_frames_tensor, dim=0, unbiased=False)  # Shape: [flattened_hw, channels]
            
            # Average across channels
            mean_motion_variance = motion_variance.mean(dim=-1)  # Shape: [flattened_hw]
        
        elif len(noise_pred.shape) == 4:  # [C, F, H, W]
            # Original implementation for 4D tensors
            latent_frames = noise_pred.permute(1, 2, 3, 0)  # Shape: [frames, height, width, channels]
            frames, height, width, channels = latent_frames.shape
            
            # Calculate frame-by-frame differences
            motion_frames = []
            for frame_idx in range(1, frames):
                current_frame = latent_frames[frame_idx]
                previous_frame = latent_frames[frame_idx - 1]
                motion_frame = torch.abs(current_frame - previous_frame)
                motion_frames.append(motion_frame)
            
            if not motion_frames:
                # No motion to calculate for single-frame generation
                return torch.tensor(0.0, device=noise_pred.device, requires_grad=True)
            
            motion_frames_tensor = torch.stack(motion_frames, dim=0)
            motion_variance = torch.var(motion_frames_tensor, dim=0, unbiased=False)
            
            # Average across channels
            mean_motion_variance = motion_variance.mean(dim=-1)  # [height, width]
        else:
            raise ValueError(f"Unsupported tensor shape: {noise_pred.shape}. Expected 3D or 4D tensor.")
        
        motion_variance_metric = torch.max(mean_motion_variance)
        
        return motion_variance_metric
    
    def _save_model_state(self, model):
        """Save model parameter states to restore later"""
        state = {
            'training': model.training,
            'requires_grad': {name: param.requires_grad for name, param in model.named_parameters()}
        }
        return state
    
    def _restore_model_state(self, model, state):
        """Restore model parameter states"""
        model.train(state['training'])
        for name, param in model.named_parameters():
            if name in state['requires_grad']:
                param.requires_grad_(state['requires_grad'][name])
    
    def optimize_noise_prediction(self, model, sample, timestep, arg_c, arg_null, guide_scale, curr_step, total_steps):
        """
        Memory-efficient sample optimization to minimize motion variance.
        
        Args:
            model: The transformer-based diffusion model
            sample: Current latent sample tensor
            timestep: Current timestep tensor
            arg_c: Arguments for conditional model forward pass
            arg_null: Arguments for unconditional model forward pass
            guide_scale: Classifier-free guidance scale
            curr_step: Current denoising step index
            total_steps: Total number of denoising steps
            
        Returns:
            Optimized sample tensor
        """
        logfile = 'test_logs.txt'

        with open(logfile, 'a') as log:
            log.write(f'timestep={curr_step}, start_after_steps={self.start_after_steps}, apply_frequency={self.apply_frequency}...')

        # Only apply after certain steps and at specified frequency
        if curr_step < self.start_after_steps or curr_step % self.apply_frequency != 0:
            with open(logfile, 'a') as log:
                log.write(f'\tSkipped!\n')
            return sample
        
        with open(logfile, 'a') as log:
                log.write(f'\tOptimizing!\n')
                log.write(f'\titerations={self.iterations}, lr={self.lr}, use_softmax_mean={self.use_softmax_mean}, temperature={self.temperature}\n')
        
        # Save model state
        model_state = self._save_model_state(model)
        
        # Setup gradient checkpointing
        checkpointer = ManualGradientCheckpointer(model)
        
        # Prepare model for inference but allow gradient propagation
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        
        # Enable gradient checkpointing
        checkpointer.enable()
        
        # Make a copy with gradients explicitly enabled
        sample_opt = sample.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([sample_opt], lr=self.lr)
        
        best_sample = sample.clone()
        lowest_variance = float('inf')
        
        
        # Enable grad tracking
        with torch.enable_grad():
            for i in range(self.iterations):
                optimizer.zero_grad()
                
                # Free memory before forward pass
                torch.cuda.empty_cache()
                gc.collect()
                
                # Verify sample_opt has gradients enabled
                assert sample_opt.requires_grad, "sample_opt must require gradients"
                
                # Run forward passes with gradient checkpointing
                latent_model_input = [sample_opt]
                
                # Forward pass for conditional prediction 
                outputs_cond = model(latent_model_input, t=timestep, **arg_c)
                
                # Ensure we have proper tensor outputs
                if isinstance(outputs_cond, (list, tuple)):
                    noise_pred_cond = outputs_cond[0]
                else:
                    noise_pred_cond = outputs_cond
                
                # Free memory between forward passes
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
                
                outputs_uncond = model(latent_model_input, t=timestep, **arg_null)
                
                if isinstance(outputs_uncond, (list, tuple)):
                    noise_pred_uncond = outputs_uncond[0]
                else:
                    noise_pred_uncond = outputs_uncond
                
                noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
                
                motion_variance = self.calculate_motion_variance(noise_pred)
                                
                loss = motion_variance 
                
                # Backward pass
                # print_gpu_memory_status()
                loss.backward(retain_graph=False)
                
                # Check if gradients were properly computed
                if sample_opt.grad is None or sample_opt.grad.abs().sum().item() == 0:
                    print("WARNING: No gradients computed for sample_opt!")
                    continue
                    
                # print(f"Gradient magnitude: {sample_opt.grad.abs().sum().item()}")
                
                # Step optimizer
                optimizer.step()
                
                # Track best result
                current_variance = motion_variance.item()
                if current_variance < lowest_variance:
                    lowest_variance = current_variance
                    best_sample = sample_opt.detach().clone()
                
                # Explicit cleanup
                del noise_pred_cond, noise_pred_uncond, noise_pred, motion_variance, loss
                torch.cuda.empty_cache()
                gc.collect()
        
        # Disable gradient checkpointing and restore original forward
        checkpointer.disable()
        
        del sample_opt, optimizer, checkpointer
        torch.cuda.empty_cache()
        gc.collect()
        
        self._restore_model_state(model, model_state)
        
        return best_sample


class ModelStateCheckpointer:
    """
    Utility class to efficiently checkpoint model state during optimization.
    """
    def __init__(self, device="cuda:0"):
        self.device = device
        self.storage = {}
    
    def save_state(self, model, key="default"):
        """
        Save the current state of model parameters without keeping computation graph.
        
        Args:
            model: Model to save state from
            key: Identifier for this saved state
        """
        state_dict = {}
        for name, param in model.named_parameters():
            # Only store parameters that require gradients to save memory
            if param.requires_grad:
                state_dict[name] = param.detach().cpu().clone()
        
        self.storage[key] = {
            'state_dict': state_dict,
            'training': model.training
        }
        
        return key
    
    def load_state(self, model, key="default"):
        """
        Restore saved state to model.
        
        Args:
            model: Model to restore state to
            key: Identifier for the saved state
        """
        if key not in self.storage:
            raise ValueError(f"No checkpoint found with key: {key}")
        
        saved_state = self.storage[key]
        
        # Restore training/eval mode
        model.train(saved_state['training'])
        
        # Restore parameter values
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in saved_state['state_dict']:
                    param_state = saved_state['state_dict'][name].to(self.device)
                    param.copy_(param_state)
        
        return model
    
    def clear(self, key="default"):
        """Clear a specific saved state to free memory"""
        if key in self.storage:
            del self.storage[key]
            
    def clear_all(self):
        """Clear all saved states"""
        self.storage.clear()


def print_gpu_memory_status():
    """Print the memory status of all available GPUs"""
    print("\n=== GPU Memory Status optimization===")
    for i in range(torch.cuda.device_count()):
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        reserved_memory = torch.cuda.memory_reserved(i) / 1024**3
        allocated_memory = torch.cuda.memory_allocated(i) / 1024**3
        free_memory = total_memory - allocated_memory
        
        print(f"GPU {i}: "
              f"Total: {total_memory:.2f} GB | "
              f"Reserved: {reserved_memory:.2f} GB | "
              f"Allocated: {allocated_memory:.2f} GB | "
              f"Free: {free_memory:.2f} GB")
    print("========================\n")