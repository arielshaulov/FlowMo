# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
print('import argparse')
from datetime import datetime
print('from datetime import datetime')
import logging
print('import logging')
import os
print('import os')
import sys
print('import sys')
import warnings
print('import warnings')
import gc
print('import gc')

warnings.filterwarnings('ignore')

import torch, random
print('import torch, random')
import torch.distributed as dist
print('import torch.distributed as dist')

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

from PIL import Image
print('from PIL import Image')

import matplotlib.pyplot as plt
print('import matplotlib.pyplot as plt')
import pandas as pd
print('import pandas as pd')    
import numpy as np
print('import numpy as np')

import wan
print('import wan')
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
print('from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES')
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
print('from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander')
from wan.utils.utils import cache_video, cache_image, str2bool
print('from wan.utils.utils import cache_video, cache_image, str2bool')

from motion_optimizer import MotionVarianceOptimizer
print('from motion_optimizer import MotionVarianceOptimizer')

import os
print('import os')

import torch.nn.functional as F
print('import torch.nn.functional as F')
original_linear = F.linear

print('====== Finishing imports ======')

def device_safe_linear(input, weight, bias=None):
    target_device = weight.device
    if input.device != target_device:
        input = input.to(target_device)
    if bias is not None and bias.device != target_device:
        bias = bias.to(target_device)
    return original_linear(input, weight, bias)
F.linear = device_safe_linear

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I frame_num check
    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="ch",
        choices=["ch", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")
    
    parser.add_argument(
        "--optimize",
        type=str2bool,
        default=None,)
    parser.add_argument(
        "--optimizer_metrics",
        type=str,
        nargs="+",  # Allows multiple metrics
        default=['max_abs_motion_variance_tensor'],)
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,)
    parser.add_argument(
        "--optimizer_lr",
        type=float,
        default=0.005,)
    parser.add_argument(
        "--optimizer_iterations",
        type=int,
        default=1,)
    parser.add_argument(
        "--optimizer_tensor",
        type=str,
        default='x0_pred',)
    parser.add_argument(
        "--optimizer_timesteps",
        type=int,
        nargs="+",  # Allows multiple timesteps
        default=[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 ],
    )

    parser.add_argument(
        "--change_prompt",
        type=str2bool,
        default=None,)
    parser.add_argument(
        "--new_prompt",
        type=str,
        default=None,)
    parser.add_argument(
        "--new_prompt_timestep",
        type=int,
        default=1,)


    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",  # Allows multiple prompts
        default=[],
        help="The prompts to generate images or videos from (provide multiple prompts separated by spaces)."
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",  # Allows multiple seeds
        default=[],
        help="The seeds for generating images or videos (provide multiple seeds separated by spaces)."
    )

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (initialize_model_parallel,
                                             init_distributed_environment)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model, is_vl="i2v" in args.task)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`num_heads` must be divisible by `ulysses_size`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    if "t2v" in args.task or "t2i" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        logging.info(f"Input prompt: {args.prompt}")
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    seed=args.base_seed)
                if prompt_output.status == False:
                    logging.info(
                        f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        formatted_prompt = args.prompt.replace(" ", "_").replace("/","_").replace("'","_")[:50]
        formatted_new_prompt = "" if not args.new_prompt else args.new_prompt.replace(" ", "_").replace("/","_").replace("'","_")[:10]

        change_prompt_string = "" if not args.change_prompt else f"changeprompt_t{args.new_prompt_timestep}_{formatted_new_prompt}"

        is_optimized_suffix = "" if not args.optimize else "_optimized"
        optimized_string = "" if not args.optimize else f"C_{args.optimizer_tensor}_{args.optimizer_metric}_t{min(args.optimizer_timesteps)}-{max(args.optimizer_timesteps)}_it{args.optimizer_iterations}_lr{args.optimizer_lr}"

        experiment_name = "" if not args.experiment_name else args.experiment_name+'_'

        full_experiment_suffix = f'test_{experiment_name}{optimized_string}{change_prompt_string}'

        log_file_name = f'{args.ring_size}_{formatted_prompt}_{args.base_seed}{is_optimized_suffix}.{full_experiment_suffix}'

        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
            log_file_name=log_file_name,
            optimizer_timesteps=args.optimizer_timesteps,
            change_prompt=args.change_prompt,
            new_prompt=args.new_prompt,
            new_prompt_timestep=args.new_prompt_timestep
        )


        motion_optimizer = None
        if args.optimize:
            motion_optimizer = MotionVarianceOptimizer(
                iterations=args.optimizer_iterations,
                lr=args.optimizer_lr,
                start_after_steps=int(args.sample_steps * 0.01),  # Start after 20% of steps
                apply_frequency=1,
                metric=args.optimizer_metric,
                optimizer_tensor=args.optimizer_tensor,
            )


        logging.info(
            f"Generating {'image' if 't2i' in args.task else 'video'} ...")
        video = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            motion_optimizer=motion_optimizer  # Pass the optimizer
        )
    

        # video = wan_t2v.generate(
        #     args.prompt,
        #     size=SIZE_CONFIGS[args.size],
        #     frame_num=args.frame_num,
        #     shift=args.sample_shift,
        #     sample_solver=args.sample_solver,
        #     sampling_steps=args.sample_steps,
        #     guide_scale=args.sample_guide_scale,
        #     seed=args.base_seed,
        #     offload_model=args.offload_model)

    else:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.image is None:
            args.image = EXAMPLE_PROMPT[args.task]["image"]
        logging.info(f"Input prompt: {args.prompt}")
        logging.info(f"Input image: {args.image}")

        img = Image.open(args.image).convert("RGB")
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    image=img,
                    seed=args.base_seed)
                if prompt_output.status == False:
                    logging.info(
                        f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        logging.info("Generating video ...")
        video = wan_i2v.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = '.png' if "t2i" in args.task else '.mp4'
            args.save_file = f"generated_videos/{log_file_name}{suffix}"

        if "t2i" in args.task:
            logging.info(f"Saving generated image to {args.save_file}")
            cache_image(
                tensor=video.squeeze(1)[None],
                save_file=args.save_file,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
        else:
            logging.info(f"Saving generated video to {args.save_file}")
            cache_video(
                tensor=video[None],
                save_file=args.save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
    logging.info("Finished.")

if __name__ == "__main__":

    args = _parse_args()
    print("Parsed arguments: ", args)

    if not args.optimizer_metrics:
        args.optimizer_metrics = ['unopt']

    for prompt in args.prompts:
        for seed in args.seeds:
            for metric in args.optimizer_metrics:
                args.prompt = prompt
                args.base_seed = seed
                args.save_file = None
                args.optimizer_metric = metric

                print(f"Generating video for prompt: '{prompt}' with seed: {seed} and metric: {metric}")
                generate(args)
                torch.cuda.empty_cache()
                gc.collect()
