#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import logging
import math
import os
import shutil
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, DistributedDataParallelKwargs
from dataset import ObjaverseData
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from CN_encoder import CN_encoder

import diffusers
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    # UNet2DConditionModel,
)
from unet_2d_condition import UNet2DConditionModel
from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.training_utils import EMAModel
import torchvision
import itertools

# metrics
import cv2
from skimage.metrics import structural_similarity as calculate_ssim
import lpips
LPIPS = lpips.LPIPS(net='alex', version='0.1')

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.19.0.dev0")

logger = get_logger(__name__)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

@torch.no_grad()
def log_validation(validation_dataloader, vae, image_encoder, feature_extractor, unet, args, accelerator, weight_dtype, split="val"):
    logger.info("Running {} validation... ".format(split))

    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    pipeline = Zero1to3StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae).eval(),
        image_encoder=accelerator.unwrap_model(image_encoder).eval(),
        feature_extractor=feature_extractor,
        unet=accelerator.unwrap_model(unet).eval(),
        scheduler=scheduler,
        safety_checker=None,
        torch_dtype=weight_dtype,
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    image_logs = []
    val_lpips = 0
    val_ssim = 0
    val_psnr = 0
    val_loss = 0
    val_num = 0
    T_out = args.T_out  # fix to be 1?
    for T_in_val in [1, args.T_in_val//2, args.T_in_val]:   # eval different number of given views
        for valid_step, batch in tqdm(enumerate(validation_dataloader)):
            if args.num_validation_batches is not None and valid_step >= args.num_validation_batches:
                break
            T_in = T_in_val
            gt_image = batch["image_target"].to(dtype=weight_dtype)
            input_image = batch["image_input"].to(dtype=weight_dtype)[:, :T_in]
            pose_in = batch["pose_in"].to(dtype=weight_dtype)[:, :T_in]   # BxTx4
            pose_out = batch["pose_out"].to(dtype=weight_dtype)  # BxTx4
            pose_in_inv = batch["pose_in_inv"].to(dtype=weight_dtype)[:, :T_in]  # BxTx4
            pose_out_inv = batch["pose_out_inv"].to(dtype=weight_dtype)  # BxTx4

            gt_image = einops.rearrange(gt_image, 'b t c h w -> (b t) c h w', t=T_out)
            input_image = einops.rearrange(input_image, 'b t c h w -> (b t) c h w', t=T_in) # T_in

            images = []
            h, w = input_image.shape[2:]
            for _ in range(args.num_validation_images):
                with torch.autocast("cuda"):
                    image = pipeline(input_imgs=input_image, prompt_imgs=input_image, poses=[[pose_out, pose_out_inv], [pose_in, pose_in_inv]], height=h, width=w, T_in=T_in, T_out=pose_out.shape[1],
                                     guidance_scale=args.guidance_scale, num_inference_steps=50, generator=generator, output_type="numpy").images

                pred_image = torch.from_numpy(image * 2. - 1.).permute(0, 3, 1, 2)
                images.append(pred_image)

                pred_np = (image * 255).astype(np.uint8) # [0,1]
                gt_np = (gt_image / 2 + 0.5).clamp(0, 1)
                gt_np = (gt_np.cpu().permute(0, 2, 3, 1).float().numpy()*255).astype(np.uint8)
                # for 1 image
                # pixel loss
                loss = F.mse_loss(pred_image[0], gt_image[0].cpu()).item()
                # LPIPS
                lpips = LPIPS(pred_image[0], gt_image[0].cpu()).item()    # [-1, 1] torch tensor
                # SSIM
                ssim = calculate_ssim(pred_np[0], gt_np[0], channel_axis=2)
                # PSNR
                psnr = cv2.PSNR(gt_np[0], pred_np[0])

                val_loss += loss
                val_lpips += lpips
                val_ssim += ssim
                val_psnr += psnr

                val_num += 1

            image_logs.append(
                {"gt_image": gt_image, "pred_images": images, "input_image": input_image}
            )

        pixel_loss = val_loss / val_num
        pixel_lpips= val_lpips / val_num
        pixel_ssim = val_ssim / val_num
        pixel_psnr = val_psnr / val_num

        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                # need to use table, wandb doesn't allow more than 108 images
                assert args.num_validation_images == 2
                table = wandb.Table(columns=["Input", "GT", "Pred1", "Pred2"])

                for log_id, log in enumerate(image_logs):
                    formatted_images = [[], [], []]  # [[input], [gt], [pred]]
                    pred_images = log["pred_images"]  # pred
                    input_image = log["input_image"]    # input
                    gt_image = log["gt_image"]  # GT

                    formatted_images[0].append(wandb.Image(input_image, caption="{}_input".format(log_id)))
                    formatted_images[1].append(wandb.Image(gt_image, caption="{}_gt".format(log_id)))

                    for sample_id, pred_image in enumerate(pred_images): # n_samples
                        pred_image = wandb.Image(pred_image, caption="{}_pred_{}".format(log_id, sample_id))
                        formatted_images[2].append(pred_image)

                    table.add_data(*formatted_images[0], *formatted_images[1], *formatted_images[2])


                tracker.log({split: table,  # formatted_images
                             "{}_T{}_pixel_loss".format(split, T_in_val): pixel_loss,
                             "{}_T{}_lpips".format(split, T_in_val): pixel_lpips,
                             "{}_T{}_ssim".format(split, T_in_val): pixel_ssim,
                             "{}_T{}_psnr".format(split, T_in_val): pixel_psnr})
            else:
                logger.warn(f"image logging not implemented for {tracker.name}")

    # del pipeline
    # torch.cuda.empty_cache()
    # after validation, set the pipeline back to training mode
    unet.train()
    vae.eval()
    image_encoder.train()

    return image_logs


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a Zero123 training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="lambdalabs/sd-image-variations-diffusers",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eschernet-6dof",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--T_in", type=int, default=1, help="Number of input views"
    )
    parser.add_argument(
        "--T_in_val", type=int, default=10, help="Number of input views"
    )
    parser.add_argument(
        "--T_out", type=int, default=1, help="Number of output views"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=100000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="unconditional guidance scale, if guidance_scale>1.0, do_classifier_free_guidance"
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.05,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=20,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=1000, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=0.5, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",    # log_image currently only for wandb
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", default=True, help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        default=True,
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")

    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=2,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=2000,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--num_validation_batches",
        type=int,
        default=20,
        help=(
            "Number of batches to use for validation. If `None`, use all batches."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_zero123_hf",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images."
        )

    return args

ConvNextV2_preprocess = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def _encode_image(feature_extractor, image_encoder, image, device, dtype, do_classifier_free_guidance):
    # [-1, 1] -> [0, 1]
    image = (image + 1.) / 2.
    image = ConvNextV2_preprocess(image)
    image_embeddings = image_encoder(image)   # bt, 768, 12, 12

    if do_classifier_free_guidance:
        negative_prompt_embeds = torch.zeros_like(image_embeddings)
        image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

    return image_embeddings     #.detach() # !we need keep image encoder gradient


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token, private=True
            ).repo_id


    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", revision=args.revision)
    image_encoder = CN_encoder.from_pretrained("facebook/convnextv2-tiny-22k-224")
    feature_extractor = None
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)

    T_in = args.T_in
    T_in_val = args.T_in_val
    T_out = args.T_out

    vae.eval()
    vae.requires_grad_(False)

    image_encoder.train()
    image_encoder.requires_grad_(True)

    unet.requires_grad_(True)
    unet.train()


    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            vae.enable_slicing()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"UNet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW


    optimizer = optimizer_class(
        [{"params": unet.parameters(), "lr": args.learning_rate},
         {"params": image_encoder.parameters(), "lr": args.learning_rate}],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )

    # print model info, learnable parameters, non-learnable parameters, total parameters, model size, all in billion
    def print_model_info(model):
        print("="*20)
        # print model class name
        print("model name: ", type(model).__name__)
        print("learnable parameters(M): ", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)
        print("non-learnable parameters(M): ", sum(p.numel() for p in model.parameters() if not p.requires_grad) / 1e6)
        print("total parameters(M): ", sum(p.numel() for p in model.parameters()) / 1e6)
        print("model size(MB): ", sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024)

    print_model_info(unet)
    print_model_info(vae)
    print_model_info(image_encoder)

    # Init Dataset
    image_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((args.resolution, args.resolution)),  # 256, 256
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    )
    train_dataset = ObjaverseData(root_dir=args.train_data_dir, image_transforms=image_transforms, validation=False, T_in=T_in, T_out=T_out)
    train_log_dataset = ObjaverseData(root_dir=args.train_data_dir, image_transforms=image_transforms, validation=False, T_in=T_in_val, T_out=T_out, fix_sample=True)
    validation_dataset = ObjaverseData(root_dir=args.train_data_dir, image_transforms=image_transforms, validation=True, T_in=T_in_val, T_out=T_out, fix_sample=True)
    # for training
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    # for validation set logs
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=1,
    )
    # for training set logs
    train_log_dataloader = torch.utils.data.DataLoader(
        train_log_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=1,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True


    def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
        """Warmup the learning rate"""
        lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
        """Decay the learning rate"""
        lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Prepare everything with our `accelerator`.
    unet, image_encoder, optimizer, train_dataloader, validation_dataloader, train_log_dataloader = accelerator.prepare(
        unet, image_encoder, optimizer, train_dataloader, validation_dataloader, train_log_dataloader
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, image_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        run_name = args.output_dir.split("logs_")[1]
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs={"wandb":{"name":run_name}})

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    do_classifier_free_guidance = args.guidance_scale > 1.0
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f" do_classifier_free_guidance = {do_classifier_free_guidance}")
    logger.info(f" conditioning_dropout_prob = {args.conditioning_dropout_prob}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        loss_epoch = 0.0
        num_train_elems = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, image_encoder):
                gt_image = batch["image_target"].to(dtype=weight_dtype) # BxTx3xHxW
                gt_image = einops.rearrange(gt_image, 'b t c h w -> (b t) c h w', t=T_out)
                input_image = batch["image_input"].to(dtype=weight_dtype)    # Bx3xHxW
                input_image = einops.rearrange(input_image, 'b t c h w -> (b t) c h w', t=T_in)
                pose_in = batch["pose_in"].to(dtype=weight_dtype)  # BxTx4
                pose_out = batch["pose_out"].to(dtype=weight_dtype)  # BxTx4
                pose_in_inv = batch["pose_in_inv"].to(dtype=weight_dtype)  # BxTx4
                pose_out_inv = batch["pose_out_inv"].to(dtype=weight_dtype)  # BxTx4

                gt_latents = vae.encode(gt_image).latent_dist.sample().detach()
                gt_latents = gt_latents * vae.config.scaling_factor # follow zero123, only target image latent is scaled

                # Sample noise that we'll add to the latents
                bsz = gt_latents.shape[0] // T_out
                noise = torch.randn_like(gt_latents)
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=gt_latents.device)
                timesteps = timesteps.long()
                timesteps = einops.repeat(timesteps, 'b -> (b t)', t=T_out)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(gt_latents.to(dtype=torch.float32), noise.to(dtype=torch.float32), timesteps).to(dtype=gt_latents.dtype)

                if do_classifier_free_guidance:  #support classifier-free guidance, randomly drop out 5%
                    # Conditioning dropout to support classifier-free guidance during inference. For more details
                    # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                    random_p = torch.rand(bsz, device=gt_latents.device)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1, 1)

                    img_prompt_embeds = _encode_image(feature_extractor, image_encoder, input_image, gt_latents.device, gt_latents.dtype, False)

                    # Final text conditioning.
                    img_prompt_embeds = einops.rearrange(img_prompt_embeds, '(b t) l c -> b t l c', t=T_in)
                    null_conditioning = torch.zeros_like(img_prompt_embeds).detach()
                    img_prompt_embeds = torch.where(prompt_mask, null_conditioning, img_prompt_embeds)
                    img_prompt_embeds = einops.rearrange(img_prompt_embeds, 'b t l c -> (b t) l c', t=T_in)
                    prompt_embeds = torch.cat([img_prompt_embeds], dim=-1)
                else:
                    # Get the image_with_pose embedding for conditioning
                    prompt_embeds = _encode_image(feature_extractor, image_encoder, input_image, gt_latents.device, gt_latents.dtype, False)

                prompt_embeds = einops.rearrange(prompt_embeds, '(b t) l c -> b (t l) c', t=T_in)

                # noisy_latents (b T_out)
                latent_model_input = torch.cat([noisy_latents], dim=1)

                # Predict the noise residual
                model_pred = unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,    # (bxT_in) l 768
                    pose=[[pose_out, pose_out_inv], [pose_in, pose_in_inv]],  # (bxT_in) 4, pose_out - self-attn, pose_in - cross-attn
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(gt_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = (loss.mean([1, 2, 3])).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = itertools.chain(unet.parameters(), image_encoder.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                # cosine
                if global_step <= args.lr_warmup_steps:
                    warmup_lr_schedule(optimizer, global_step, args.lr_warmup_steps, 1e-5, args.learning_rate)
                else:
                    cosine_lr_schedule(optimizer, global_step, args.max_train_steps, args.learning_rate, 1e-5)
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                        # save pipeline
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            pipelines = os.listdir(args.output_dir)
                            pipelines = [d for d in pipelines if d.startswith("pipeline")]
                            pipelines = sorted(pipelines, key=lambda x: int(x.split("-")[1]))

                            # before we save the new pipeline, we need to have at _most_ `checkpoints_total_limit - 1` pipeline
                            if len(pipelines) >= args.checkpoints_total_limit:
                                num_to_remove = len(pipelines) - args.checkpoints_total_limit + 1
                                removing_pipelines = pipelines[0:num_to_remove]

                                logger.info(
                                    f"{len(pipelines)} pipelines already exist, removing {len(removing_pipelines)} pipelines"
                                )
                                logger.info(f"removing pipelines: {', '.join(removing_pipelines)}")

                                for removing_pipeline in removing_pipelines:
                                    removing_pipeline = os.path.join(args.output_dir, removing_pipeline)
                                    shutil.rmtree(removing_pipeline)

                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())

                        pipeline = Zero1to3StableDiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            vae=accelerator.unwrap_model(vae),
                            image_encoder=accelerator.unwrap_model(image_encoder),
                            feature_extractor=feature_extractor,
                            unet=accelerator.unwrap_model(unet),
                            scheduler=noise_scheduler,
                            safety_checker=None,
                            torch_dtype=torch.float32,
                        )
                        pipeline_save_path = os.path.join(args.output_dir, f"pipeline-{global_step}")
                        pipeline.save_pretrained(pipeline_save_path)
                        # del pipeline

                        if args.push_to_hub:
                            print("Pushing to the hub ", repo_id)
                            upload_folder(
                                repo_id=repo_id,
                                folder_path=pipeline_save_path,
                                commit_message=global_step,
                                ignore_patterns=["step_*", "epoch_*"],
                                run_as_future=True,
                            )

                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())

                    if validation_dataloader is not None and global_step % args.validation_steps == 0:
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        image_logs = log_validation(
                            validation_dataloader,
                            vae,
                            image_encoder,
                            feature_extractor,
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            'val',
                        )
                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())
                    if train_log_dataloader is not None and (global_step % args.validation_steps == 0 or global_step == 1):
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        train_image_logs = log_validation(
                            train_log_dataloader,
                            vae,
                            image_encoder,
                            feature_extractor,
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            'train',
                        )
                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())
            loss_epoch += loss.detach().item()
            num_train_elems += 1

            logs = {"loss": loss.detach().item(), "lr": optimizer.param_groups[0]['lr'],
                    "loss_epoch": loss_epoch / num_train_elems,
                    "epoch": epoch}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break



    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = Zero1to3StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=accelerator.unwrap_model(vae),
            image_encoder=accelerator.unwrap_model(image_encoder),
            feature_extractor=feature_extractor,
            unet=unet,
            scheduler=noise_scheduler,
            safety_checker=None,
            torch_dtype=torch.float32,
        )
        pipeline_save_path = os.path.join(args.output_dir, f"pipeline-{global_step}")
        pipeline.save_pretrained(pipeline_save_path)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=pipeline_save_path,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    # torch.multiprocessing.set_sharing_strategy("file_system")
    args = parse_args()
    main(args)
