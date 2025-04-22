# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
import torch.nn.functional as F
from .attention import flash_attention

__all__ = ['WanModel']

import subprocess

def printGPUState():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            check=True
        )

        lines = result.stdout.strip().split("\n")
        for line in lines:
            index, name, mem_total, mem_used, util_gpu, util_mem = [item.strip() for item in line.split(",")]
            print(f"GPU {index}: {name}")
            print(f"  Memory Used: {mem_used} MB / {mem_total} MB")
            print(f"  GPU Utilization: {util_gpu}%")
            print(f"  Memory Utilization: {util_mem}%")
            print()

    except subprocess.CalledProcessError as e:
        print("Failed to run nvidia-smi:", e)
    except FileNotFoundError:
        print("nvidia-smi not found. Make sure NVIDIA drivers are installed and accessible.")

def get_optimal_device(tensor_shape=None):
    """
    Select the GPU with the most available memory to balance workload.
    Returns the device with most available memory.
    """
    
    # printGPUState()
    
    max_free = 0
    optimal_device = "cuda:0"  # Default fallback
    
    for i in range(torch.cuda.device_count()):
        device = f"cuda:{i}"
        free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
        
        # If we know the tensor shape, we can be more precise in our decision
        if tensor_shape is not None:
            # Approximate memory needed for this tensor
            element_size = 4  # Assume float32 (4 bytes)
            tensor_size = math.prod(tensor_shape) * element_size
            
            # If this device doesn't have enough free memory, skip it
            if free_memory < tensor_size * 1.5:  # 1.5x buffer for operations
                continue
                
        if free_memory > max_free:
            max_free = free_memory
            optimal_device = device

    return optimal_device


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    device = position.device
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half, device=device).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """
    Optimized version of rope_apply function with improved device management.
    """
    # Get the device of input tensor x
    device = x.device
    
    # Ensure grid_sizes and freqs are on the same device
    grid_sizes = grid_sizes.to(device)
    freqs = freqs.to(device)
    
    n, c = x.size(2), x.size(3) // 2
    
    # Split freqs but ensure they stay on the right device
    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    
    # Loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        
        # Get slice of x
        x_slice = x[i, :seq_len]
        
        # Convert to complex
        x_i = torch.view_as_complex(x_slice.to(torch.float64).reshape(
            seq_len, n, -1, 2))
        
        # Create freqs_i with explicit device management
        freqs_0 = freqs_split[0][:f]
        freqs_1 = freqs_split[1][:h]
        freqs_2 = freqs_split[2][:w]
        
        # Create views
        freqs_0_view = freqs_0.view(f, 1, 1, -1).expand(f, h, w, -1)
        freqs_1_view = freqs_1.view(1, h, 1, -1).expand(f, h, w, -1)
        freqs_2_view = freqs_2.view(1, 1, w, -1).expand(f, h, w, -1)
        
        # Concatenate
        freqs_i = torch.cat([
            freqs_0_view, 
            freqs_1_view,
            freqs_2_view
        ], dim=-1).reshape(seq_len, 1, -1)
        
        # Apply rotary embedding
        result = torch.view_as_real(x_i * freqs_i).flatten(2)
        
        # Get remaining slice of x
        remaining = x[i, seq_len:]
        
        # Concatenate
        x_i = torch.cat([result, remaining])
        
        # Append to output
        output.append(x_i)
    
    # Stack results
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """Device-aware norm implementation"""
        # Use the device of the input tensor
        device = x.device
        weight = self.weight.to(device)
        
        # Call the norm method
        return self._norm(x.float()).type_as(x) * weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        """Device-aware layer norm implementation"""
        # Use the device of the input tensor for all operations
        device = x.device
        
        if self.weight is not None:
            weight = self.weight.to(device)
        else:
            weight = None
            
        if self.bias is not None:
            bias = self.bias.to(device)
        else:
            bias = None
        
        # Call layer_norm with device-consistent tensors
        normalized = F.layer_norm(
            x.float(), 
            self.normalized_shape, 
            weight, 
            bias,
            self.eps
        )
        
        # Return result with original dtype
        return normalized.type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        """
        Device-balanced attention implementation
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        device = x.device

        # Move attention parameters to input device if needed
        q_weight = self.q.weight.to(device)
        q_bias = self.q.bias.to(device) if self.q.bias is not None else None
        k_weight = self.k.weight.to(device)
        k_bias = self.k.bias.to(device) if self.k.bias is not None else None
        v_weight = self.v.weight.to(device)
        v_bias = self.v.bias.to(device) if self.v.bias is not None else None
        o_weight = self.o.weight.to(device)
        o_bias = self.o.bias.to(device) if self.o.bias is not None else None

        # query, key, value function
        def qkv_fn(x):
            # Perform operations with consistent device
            q_linear = F.linear(x, q_weight, q_bias)
            q_norm = self.norm_q(q_linear)
            q = q_norm.view(b, s, n, d)
            
            k_linear = F.linear(x, k_weight, k_bias)
            k_norm = self.norm_k(k_linear)
            k = k_norm.view(b, s, n, d)
            
            v_linear = F.linear(x, v_weight, v_bias)
            v = v_linear.view(b, s, n, d)
            
            return q, k, v

        q, k, v = qkv_fn(x)

        # Ensure all inputs to flash_attention are on the same device
        seq_lens = seq_lens.to(device)
        grid_sizes = grid_sizes.to(device)
        freqs = freqs.to(device)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = F.linear(x, o_weight, o_bias)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens):
        """
        Device-balanced cross-attention implementation
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        device = x.device

        # Move parameters to input device
        q_weight = self.q.weight.to(device)
        q_bias = self.q.bias.to(device) if self.q.bias is not None else None
        k_weight = self.k.weight.to(device)
        k_bias = self.k.bias.to(device) if self.k.bias is not None else None
        v_weight = self.v.weight.to(device)
        v_bias = self.v.bias.to(device) if self.v.bias is not None else None
        o_weight = self.o.weight.to(device)
        o_bias = self.o.bias.to(device) if self.o.bias is not None else None

        # Ensure context is on the same device
        context = context.to(device)
        if context_lens is not None:
            context_lens = context_lens.to(device)

        # compute query, key, value
        q = self.norm_q(F.linear(x, q_weight, q_bias)).view(b, -1, n, d)
        k = self.norm_k(F.linear(context, k_weight, k_bias)).view(b, -1, n, d)
        v = F.linear(context, v_weight, v_bias).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = F.linear(x, o_weight, o_bias)
        return x


class WanI2VCrossAttention(WanSelfAttention):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        """
        Device-balanced image-to-video cross-attention implementation
        """
        device = x.device
        
        # Move all model parameters to the input device
        q_weight = self.q.weight.to(device)
        q_bias = self.q.bias.to(device) if self.q.bias is not None else None
        k_weight = self.k.weight.to(device)
        k_bias = self.k.bias.to(device) if self.k.bias is not None else None
        v_weight = self.v.weight.to(device)
        v_bias = self.v.bias.to(device) if self.v.bias is not None else None
        o_weight = self.o.weight.to(device)
        o_bias = self.o.bias.to(device) if self.o.bias is not None else None
        k_img_weight = self.k_img.weight.to(device)
        k_img_bias = self.k_img.bias.to(device) if self.k_img.bias is not None else None
        v_img_weight = self.v_img.weight.to(device)
        v_img_bias = self.v_img.bias.to(device) if self.v_img.bias is not None else None
        
        # Ensure context is on the input device
        context = context.to(device)
        if context_lens is not None:
            context_lens = context_lens.to(device)
            
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(F.linear(x, q_weight, q_bias)).view(b, -1, n, d)
        k = self.norm_k(F.linear(context, k_weight, k_bias)).view(b, -1, n, d)
        v = F.linear(context, v_weight, v_bias).view(b, -1, n, d)
        k_img = self.norm_k_img(F.linear(context_img, k_img_weight, k_img_bias)).view(b, -1, n, d)
        v_img = F.linear(context_img, v_img_weight, v_img_bias).view(b, -1, n, d)
        
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = F.linear(x, o_weight, o_bias)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):
    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        """
        Device-balanced attention block implementation
        """
        # Use the device of the input tensor
        device = x.device
        
        # Move modulation to the input device
        modulation = self.modulation.to(device)
        
        # Ensure e is on the input device
        e = e.to(device)
        
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e_chunks = (modulation + e).chunk(6, dim=1)
        assert e_chunks[0].dtype == torch.float32

        # Ensure all tensors for self-attention are on the input device
        norm_x = self.norm1(x).float()
        e0 = e_chunks[0]
        e1 = e_chunks[1]
        e2 = e_chunks[2]
        
        # Self-attention with device-consistent tensors
        y = self.self_attn(norm_x * (1 + e1) + e0, seq_lens, grid_sizes, freqs)

        with amp.autocast(dtype=torch.float32):
            x = x + y * e2

        # Cross-attention with device-consistent tensors
        x = self._cross_attn_ffn(x, context, context_lens, e_chunks)
        return x
        
    def _cross_attn_ffn(self, x, context, context_lens, e):
        """
        Device-consistent cross-attention and FFN processing
        """
        # Use the device of the input tensor
        device = x.device
        
        # Move all tensors to the input device
        context = context.to(device) if context is not None else None
        context_lens = context_lens.to(device) if context_lens is not None else None
        e3 = e[3].to(device)
        e4 = e[4].to(device)
        e5 = e[5].to(device)
        
        # Move FFN parameters to device
        ffn_params = []
        for module in self.ffn:
            if hasattr(module, 'weight'):
                ffn_params.append((module.weight.to(device), 
                                 module.bias.to(device) if module.bias is not None else None))
        
        # Cross-attention processing
        norm3_x = self.norm3(x)
        cross_attn_result = self.cross_attn(norm3_x, context, context_lens)
        x = x + cross_attn_result
        
        # FFN processing
        norm2_x = self.norm2(x).float()
        ffn_input = norm2_x * (1 + e4) + e3
        
        # Apply FFN manually with device-consistent parameters
        hidden = F.linear(ffn_input, ffn_params[0][0], ffn_params[0][1])
        hidden = F.gelu(hidden, approximate='tanh')
        y = F.linear(hidden, ffn_params[1][0], ffn_params[1][1])
        
        # Final addition
        with amp.autocast(dtype=torch.float32):
            x = x + y * e5
        
        return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        """
        Device-consistent head implementation
        """
        # Use the device of the input tensor
        device = x.device
        
        # Move modulation and parameters to the input device
        modulation = self.modulation.to(device)
        head_weight = self.head.weight.to(device)
        head_bias = self.head.bias.to(device) if self.head.bias is not None else None
        
        # Ensure e is on the input device
        e = e.to(device)
        
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e_unsqueezed = e.unsqueeze(1)
            e_chunks = (modulation + e_unsqueezed).chunk(2, dim=1)
            e0 = e_chunks[0]
            e1 = e_chunks[1]
            
            # Process with norm
            norm_x = self.norm(x)
            transformed_x = norm_x * (1 + e1) + e0
            
            # Apply head
            output = F.linear(transformed_x, head_weight, head_bias)
        
        return output


class MLPProj(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), 
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), 
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim)
        )

    def forward(self, image_embeds):
        # Use the device of input tensor
        device = image_embeds.device
        
        # Process sequentially with device management
        x = image_embeds
        
        # Process through each layer with device management
        for layer in self.proj:
            if hasattr(layer, 'weight') and hasattr(layer, 'bias'):
                # Linear layer
                weight = layer.weight.to(device)
                bias = layer.bias.to(device) if layer.bias is not None else None
                x = F.linear(x, weight, bias)
            elif isinstance(layer, torch.nn.LayerNorm):
                # LayerNorm
                if layer.elementwise_affine:
                    weight = layer.weight.to(device)
                    bias = layer.bias.to(device) if layer.bias is not None else None
                else:
                    weight = None
                    bias = None
                x = F.layer_norm(x, layer.normalized_shape, weight, bias, layer.eps)
            elif isinstance(layer, torch.nn.GELU):
                # GELU
                x = F.gelu(x, approximate='tanh')
            else:
                # Other layers
                x = layer(x)
        
        return x


class WanModel(ModelMixin, ConfigMixin):
    """
    Wan diffusion backbone with improved GPU distribution
    """
    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.has_multiple_gpus = torch.cuda.device_count() > 1

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
            
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), 
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), 
            nn.SiLU(), 
            nn.Linear(dim, dim)
        )
        
        self.time_projection = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(dim, dim * 6)
        )

        # Create attention blocks without directly specifying device
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(
                cross_attn_type, dim, ffn_dim, num_heads,
                window_size, qk_norm, cross_attn_norm, eps
            ) for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # Create freqs tensor
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()
        
        # Print available GPUs and their status
        # self._print_gpu_info()
        
    def _print_gpu_info(self):
        """Print information about available GPUs"""
        print("\n=== GPU Configuration ===")
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPU(s)")
        
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated(i) / 1024**3
            free_memory = total_memory - allocated_memory
            
            print(f"GPU {i}: {props.name}")
            print(f"  - Total Memory: {total_memory:.2f} GB")
            print(f"  - Allocated Memory: {allocated_memory:.2f} GB")
            print(f"  - Free Memory: {free_memory:.2f} GB")
        
        print("========================\n")

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
    ):
        """
        Forward pass with improved device distribution
        """
        # Track which GPU has most free memory for large operations
        primary_device = get_optimal_device()
        # print(f"Using {primary_device} as primary device for large operations")
        
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
            
        # Process each input on its own device, then move to appropriate compute device
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # Get patch embeddings
        # print("Processing patch embeddings...")
        patch_emb_device = self.patch_embedding.weight.device
        # print(f"Patch embedding weight device: {patch_emb_device}")
        
        # Find best device for patch embedding
        best_device = get_optimal_device()
        # print(f"Best device for patch embedding: {best_device}")
        
        # Apply patch embedding with device management
        x_processed = []
        for idx, u in enumerate(x):
            # Choose device for this batch item
            if self.has_multiple_gpus:
                # Alternate between available GPUs for better distribution
                item_device = f"cuda:{idx % torch.cuda.device_count()}"
            else:
                item_device = "cuda:0"
                
            # Process this item
            u = u.to(item_device)
            weight = self.patch_embedding.weight.to(item_device)
            bias = self.patch_embedding.bias.to(item_device) if self.patch_embedding.bias is not None else None
            
            # Apply convolution manually
            u = u.unsqueeze(0)  # Add batch dimension
            u = F.conv3d(u, weight, bias, stride=self.patch_size)
            x_processed.append(u)
        
        x = x_processed
        # print("Finished patch embeddings")
        
        # Continue with grid sizes and shape processing
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long).to(u.device) for u in x])
        
        x = [u.flatten(2).transpose(1, 2) for u in x]
        
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long).to(primary_device)
        assert seq_lens.max() <= seq_len
        
        # Move each tensor to a suitable device before concat
        for idx, tensor in enumerate(x):
            if self.has_multiple_gpus:
                target_device = f"cuda:{idx % torch.cuda.device_count()}"
                x[idx] = tensor.to(target_device)
        
        # Concatenate with padding - keep each tensor on its original device
        x_padded = []
        for u in x:
            device = u.device
            padded = torch.cat([
                u, 
                torch.zeros(1, seq_len - u.size(1), u.size(2), device=device)
            ], dim=1)
            x_padded.append(padded)
        
        # Now move all to primary device for concat
        x = torch.cat([tensor.to(primary_device) for tensor in x_padded])

        # Process time embeddings on least used GPU
        time_device = get_optimal_device()
        # print(f"Processing time embeddings on {time_device}")
        
        with amp.autocast(dtype=torch.float32):
            # Process time embedding with direct parameter access
            t = t.to(time_device)
            sinusoidal_t = sinusoidal_embedding_1d(self.freq_dim, t).float()
            
            # Get weight references for time embedding
            te_linear1 = self.time_embedding[0]
            te_linear1_weight = te_linear1.weight.to(time_device)
            te_linear1_bias = te_linear1.bias.to(time_device) if te_linear1.bias is not None else None
            
            # There's an activation function in between, so access the second linear layer carefully
            te_linear2 = self.time_embedding[2]  # Skip the activation function at index 1
            te_linear2_weight = te_linear2.weight.to(time_device)
            te_linear2_bias = te_linear2.bias.to(time_device) if te_linear2.bias is not None else None
            
            # Get the linear layer weights from time_projection
            # Note: time_projection[0] is SiLU, time_projection[1] is Linear
            tp_linear = self.time_projection[1]
            tp_linear_weight = tp_linear.weight.to(time_device)
            tp_linear_bias = tp_linear.bias.to(time_device) if tp_linear.bias is not None else None
            
            # Apply operations
            e = F.linear(sinusoidal_t, te_linear1_weight, te_linear1_bias)
            e = F.silu(e)
            e = F.linear(e, te_linear2_weight, te_linear2_bias)
            
            # Time projection
            e = F.silu(e)
            e0 = F.linear(e, tp_linear_weight, tp_linear_bias).unflatten(1, (6, self.dim))
            
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # Process context on a balanced device
        text_device = get_optimal_device()
        # print(f"Processing text embeddings on {text_device}")
        
        # Get text embedding parameters
        te_seq = self.text_embedding
        
        # First linear layer
        te_linear1 = te_seq[0]
        te_linear1_weight = te_linear1.weight.to(text_device)
        te_linear1_bias = te_linear1.bias.to(text_device) if te_linear1.bias is not None else None
        
        # Second linear layer (index 2 because index 1 is GELU)
        te_linear2 = te_seq[2]
        te_linear2_weight = te_linear2.weight.to(text_device)
        te_linear2_bias = te_linear2.bias.to(text_device) if te_linear2.bias is not None else None
        
        # Move context to text device
        context = [c.to(text_device) for c in context]
        
        # Stack and pad context
        context_stacked = torch.stack([
            torch.cat([
                u, 
                torch.zeros(self.text_len - u.size(0), u.size(1), device=text_device)
            ]) for u in context
        ])
        
        # Apply text embedding
        context = F.linear(context_stacked, te_linear1_weight, te_linear1_bias)
        context = F.gelu(context, approximate='tanh')
        context = F.linear(context, te_linear2_weight, te_linear2_bias)

        # Process CLIP features if provided
        if clip_fea is not None:
            clip_device = get_optimal_device()
            context_clip = self.img_emb(clip_fea.to(clip_device))
            context = torch.concat([context_clip.to(context.device), context], dim=1)

        # Move freqs to device with most memory
        freqs_device = get_optimal_device()
        freqs = self.freqs.to(freqs_device)
        
        # Set up for processing blocks
        context_lens = None
        
        # Forward pass through blocks with balanced device assignment
        # print("Starting forward pass through blocks...")
        
        for i, block in enumerate(self.blocks):
            # Choose optimal device for this block
            if self.has_multiple_gpus:
                block_device = f"cuda:{i % torch.cuda.device_count()}"
            else:
                block_device = "cuda:0"
                
            # print(f"Processing block {i+1}/{len(self.blocks)} on {block_device}")
            
            # Move inputs to block's device
            x_block = x.to(block_device)
            e0_block = e0.to(block_device)
            seq_lens_block = seq_lens.to(block_device)
            grid_sizes_block = grid_sizes.to(block_device)
            freqs_block = freqs.to(block_device)
            context_block = context.to(block_device)
            context_lens_block = context_lens.to(block_device) if context_lens is not None else None
            
            # Process block
            x = block(
                x_block, 
                e=e0_block,
                seq_lens=seq_lens_block,
                grid_sizes=grid_sizes_block,
                freqs=freqs_block,
                context=context_block,
                context_lens=context_lens_block
            )
            
            # Print memory status after each block
            # if (i+1) % 4 == 0:
            #     self._print_memory_status()

        # Head processing on optimal device
        head_device = get_optimal_device()
        # print(f"Processing head on {head_device}")
        x = self.head(x.to(head_device), e.to(head_device))

        # Unpatchify on optimal device for final results
        unpatching_device = get_optimal_device()
        # print(f"Unpatchifying on {unpatching_device}")
        x = self.unpatchify(x.to(unpatching_device), grid_sizes.to(unpatching_device))
        
        # print("Forward pass complete")
        return [u.float() for u in x]
        
    def _print_memory_status(self):
        """Print current GPU memory status"""
        print("\n=== GPU Memory Status ===")
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            free = total_memory - allocated
            
            print(f"GPU {i}: Total: {total_memory:.2f}GB | "
                  f"Reserved: {reserved:.2f}GB | "
                  f"Allocated: {allocated:.2f}GB | "
                  f"Free: {free:.2f}GB")
        print("=========================")

    def unpatchify(self, x, grid_sizes):
        """
        Device-consistent unpatchify implementation
        """
        device = x.device
        grid_sizes = grid_sizes.to(device)
        c = self.out_dim
        out = []
        
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
            
        return out

    def init_weights(self):
        """
        Initialize model parameters
        """
        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)