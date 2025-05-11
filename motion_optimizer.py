import torch
import gc
from typing import Dict, Any, Optional, Callable
import torch.utils.checkpoint as checkpoint
import types
import os
from metric_utils import MetricCalculator

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
                 lr=0.005,
                 start_after_steps=15, 
                 apply_frequency=5,
                 use_softmax_mean=True,
                 temperature=10.0,
                 metric='max_variance',
                 optimizer_tensor='x0_pred'):
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
        self.metric = metric
        self.optimizer_tensor = optimizer_tensor
        self.metric_calculator = MetricCalculator()

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
    
    def optimize_noise_prediction(self, model, sample, timestep, arg_c, arg_null, guide_scale, curr_step, total_steps, sample_scheduler, prompt, seed):
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
            sample_scheduler: Scheduler for sample generation
        Returns:
            Optimized sample tensor
        """
        
        # Only apply after certain steps and at specified frequency
        if curr_step < self.start_after_steps or curr_step % self.apply_frequency != 0:
            return sample
        
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
                
                noise_pred_cond = model(
                    latent_model_input, t=timestep, **arg_c)
                
                # Free memory between forward passes
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
                
                noise_pred_uncond = model(
                    latent_model_input, t=timestep, **arg_null)

                ############### noise_pred is basically x1-x0 = model_output (with classifier-free guidance) ##################
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)
                
                model_output = noise_pred.unsqueeze(0)
                sample = sample.unsqueeze(0)

                # Regular scheduler step without optimization
                x0_pred = sample_scheduler.convert_model_output(
                    prompt,
                    seed,
                    model_output, 
                    sample=sample,
                    timestep=timestep)[0]

                if self.optimizer_tensor == 'x0_pred':
                    self.metric_calculator.calculate_metrics(x0_pred)
                elif self.optimizer_tensor == 'model_output':
                    self.metric_calculator.calculate_metrics(model_output)
                elif self.optimizer_tensor == 'sample':
                    self.metric_calculator.calculate_metrics(sample)
                else:
                    raise ValueError(f"Unknown optimizer_tensor: {self.optimizer_tensor}")

                loss = self.metric_calculator.get_metric(self.metric)
                
                # Backward pass
                # print_gpu_memory_status()
                loss.backward(retain_graph=False)
                
                # Check if gradients were properly computed
                if sample_opt.grad is None or sample_opt.grad.abs().sum().item() == 0:
                    print("WARNING: No gradients computed for sample_opt!")
                    continue
                    
                print(f"Gradient magnitude: {sample_opt.grad.abs().sum().item()}")
                
                # Step optimizer
                optimizer.step()
                
                # Track best result
                current_variance = loss.item()
                if current_variance < lowest_variance:
                    lowest_variance = current_variance
                    best_sample = sample_opt.detach().clone()
                
                # Explicit cleanup
                del noise_pred_cond, noise_pred_uncond, noise_pred, x0_pred, loss
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
