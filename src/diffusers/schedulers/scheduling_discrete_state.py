# Copyright 2024 UC Berkeley Team and The HuggingFace Team. All rights reserved.
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
# limitations under the License.

# DISCLAIMER: This file is strongly influenced by https://github.com/ermongroup/ddim

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Literal
from functools import cached_property

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from ..utils.torch_utils import unsqueeze_as
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin


@dataclass
class DiscreteStateSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: Tensor
    pred_original_sample: Optional[Tensor] = None
    prev_logits: Optional[Tensor] = None
    pred_original_logits: Optional[Tensor] = None


def betas_for_alpha_bar(
        num_diffusion_timesteps,
        max_beta=0.999,
        alpha_transform_type="cosine",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


# Copied from diffusers.schedulers.scheduling_ddim.rescale_zero_terminal_snr
def rescale_zero_terminal_snr(betas):
    """
    Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)


    Args:
        betas (`torch.FloatTensor`):
            the betas that the scheduler is being initialized with.

    Returns:
        `torch.FloatTensor`: rescaled betas with zero terminal SNR
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


class DiscreteStateScheduler(SchedulerMixin, ConfigMixin):
    """
    `DDPMScheduler` explores the connections between denoising score matching and Langevin dynamics sampling.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        beta_schedule (`str`, defaults to `"linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, *optional*):
            An array of betas to pass directly to the constructor without using `beta_start` and `beta_end`.
        variance_type (`str`, defaults to `"fixed_small"`):
            Clip the variance when adding noise to the denoised sample. Choose from `fixed_small`, `fixed_small_log`,
            `fixed_large`, `fixed_large_log`, `learned` or `learned_range`.
        clip_sample (`bool`, defaults to `True`):
            Clip the predicted sample for numerical stability.
        clip_sample_range (`float`, defaults to 1.0):
            The maximum magnitude for sample clipping. Valid only when `clip_sample=True`.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        thresholding (`bool`, defaults to `False`):
            Whether to use the "dynamic thresholding" method. This is unsuitable for latent-space diffusion models such
            as Stable Diffusion.
        dynamic_thresholding_ratio (`float`, defaults to 0.995):
            The ratio for the dynamic thresholding method. Valid only when `thresholding=True`.
        sample_max_value (`float`, defaults to 1.0):
            The threshold value for dynamic thresholding. Valid only when `thresholding=True`.
        timestep_spacing (`str`, defaults to `"leading"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps, as required by some model families.
        rescale_betas_zero_snr (`bool`, defaults to `False`):
            Whether to rescale the betas to have zero terminal SNR. This enables the model to generate very bright and
            dark samples instead of limiting it to samples with medium brightness. Loosely related to
            [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506).
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
            self,
            num_classes: int = 2,
            num_train_timesteps: int = 1000,
            beta_start: float = 0.0001,
            beta_end: float = 0.02,
            beta_schedule: str = "squaredcos_cap_v2",
            trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
            timestep_spacing: str = "leading",
            steps_offset: int = 0,
            rescale_betas_zero_snr: int = False,
            transition_mat_type: Optional[Literal["uniform", "gaussian", "absorbing"]] = None,
            transition_bands: Optional[int] = None,
            implementation: str = "simple"
    ):
        self.implementation = implementation
        if transition_mat_type is not None:
            self.implementation = "matrix"
            if transition_mat_type == "uniform":
                beta_schedule = "squaredcos_cap_v2"
            elif transition_mat_type == "gaussian":
                beta_schedule = "linear"
                beta_start = 0.02
                beta_end  = 1.0
            elif transition_mat_type == "absorbing":
                beta_schedule = "jsd"

        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float64)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float64)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps,
                                        dtype=torch.float64) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        elif beta_schedule == "sigmoid":
            # GeoDiff sigmoid schedule
            betas = torch.linspace(-6, 6, num_train_timesteps, dtype=torch.float64)
            self.betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        elif beta_schedule == "jsd":
            self.betas = 1. / torch.linspace(num_train_timesteps, 1, num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__.__name__}")

        # Rescale for zero SNR
        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0
        self.eps = 1e-6

        # setable values
        self.custom_timesteps = False
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

    @cached_property
    def log_alphas(self) -> Tensor:
        return torch.log(self.alphas)

    @cached_property
    def log_alphas_cumprod(self) -> Tensor:
        return torch.cumsum(self.log_alphas, dim=0)

    @cached_property
    def alphas_mats(self) -> Tensor:
        matrix_type = self.config.transition_mat_type
        if matrix_type is None or matrix_type == 'uniform':
            return self.uniform_transition_mats()
        elif matrix_type == 'gaussian':
            return self.gaussian_transition_mats()
        elif matrix_type == 'absorbing':
            return self.absorbing_transition_mats()
        else:
            raise ValueError(f"Unsupported transition matrix type: {self.config.transition_mat_type}")

    def uniform_transition_mats(self) -> Tensor:
        num_classes = self.config.num_classes
        transition_bands = self.config.transition_bands
        alpha_mats = []
        for beta in self.betas:
            if transition_bands is None:
                mat = torch.ones(num_classes, num_classes, dtype=torch.float64) * beta / num_classes
                mat.diagonal().fill_(1 - (num_classes - 1) * beta / num_classes)
            else:
                mat = torch.zeros(num_classes, num_classes, dtype=torch.float64)
                off_diag = torch.full((num_classes - 1,), beta / float(num_classes), dtype=torch.float64)
                for k in range(1, transition_bands + 1):
                    mat += torch.diag(off_diag, k=k)
                    mat += torch.diag(off_diag, k=-k)
                    off_diag = off_diag[:-1]
                diag = 1. - mat.sum(1)
                mat += torch.diag(diag, k=0)
            alpha_mats.append(mat)
        return torch.stack(alpha_mats)

    def gaussian_transition_mats(self) -> Tensor:
        num_classes = self.config.num_classes
        transition_bands = self.config.transition_bands or num_classes - 1
        alpha_mats = []
        for t, beta in enumerate(self.betas.numpy()):
            mat = np.zeros((num_classes, num_classes), dtype=np.float64)
            values = np.linspace(0, num_classes - 1, num_classes, endpoint=True, dtype=np.float64)
            values = values * 2. / (num_classes - 1.)
            values = values[:transition_bands + 1]
            values = -values * values / beta
            values = np.concatenate([values[:0:-1], values], axis=0)
            values = F.softmax(torch.from_numpy(values), dim=0).numpy()
            values = values[transition_bands:]
            for k in range(1, transition_bands + 1):
                off_diag = np.full((num_classes - k,), values[k], dtype=np.float64)
                mat += np.diag(off_diag, k=k)
                mat += np.diag(off_diag, k=-k)
            diag = 1. - mat.sum(axis=1)
            mat += np.diag(diag, k=0)
            alpha_mats.append(mat)
        return torch.stack([torch.from_numpy(mat) for mat in alpha_mats])

    def absorbing_transition_mats(self) -> Tensor:
        num_classes = self.config.num_classes
        alpha_mats = []
        for beta in self.betas.numpy():
            diag = np.full((num_classes,), 1. - beta, dtype=np.float64)
            mat = np.diag(diag, k=0)
            mat[:, num_classes // 2] += beta
            alpha_mats.append(mat)
        return torch.stack([torch.from_numpy(mat) for mat in alpha_mats])

    @cached_property
    def alphas_cumprod_mats(self) -> Tensor:
        alpha_mat_t = self.alphas_mats[0]
        alpha_bar_mats = [alpha_mat_t]
        for idx in range(1, self.config.num_train_timesteps):
            alpha_mat_t = alpha_mat_t @ self.alphas_mats[idx]
            alpha_bar_mats.append(alpha_mat_t)
        return torch.stack(alpha_bar_mats)

    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        return sample

    def set_timesteps(
            self,
            num_inference_steps: Optional[int] = None,
            device: Union[str, torch.device] = None,
            timesteps: Optional[List[int]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of equal spacing between timesteps is used. If `timesteps` is passed,
                `num_inference_steps` must be `None`.

        """
        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `custom_timesteps`.")

        if timesteps is not None:
            for i in range(1, len(timesteps)):
                if timesteps[i] >= timesteps[i - 1]:
                    raise ValueError("`custom_timesteps` must be in descending order.")

            if timesteps[0] >= self.config.num_train_timesteps:
                raise ValueError(
                    f"`timesteps` must start before `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps}."
                )

            timesteps = np.array(timesteps, dtype=np.int64)
            self.custom_timesteps = True
        else:
            if num_inference_steps > self.config.num_train_timesteps:
                raise ValueError(
                    f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                    f" maximal {self.config.num_train_timesteps} timesteps."
                )

            self.num_inference_steps = num_inference_steps
            self.custom_timesteps = False

            # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
            if self.config.timestep_spacing == "linspace":
                timesteps = (
                    np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
                    .round()[::-1]
                    .copy()
                    .astype(np.int64)
                )
            elif self.config.timestep_spacing == "leading":
                step_ratio = self.config.num_train_timesteps // self.num_inference_steps
                # creates integer timesteps by multiplying by ratio
                # casting to int to avoid issues when num_inference_step is power of 3
                timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
                timesteps += self.config.steps_offset
            elif self.config.timestep_spacing == "trailing":
                step_ratio = self.config.num_train_timesteps / self.num_inference_steps
                # creates integer timesteps by multiplying by ratio
                # casting to int to avoid issues when num_inference_step is power of 3
                timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
                timesteps -= 1
            else:
                raise ValueError(
                    f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
                )

        self.timesteps = torch.from_numpy(timesteps).to(device)

    def sample(self,
               probs_or_logits: Tensor,
               noise: Optional[Union[bool, Tensor]] = None,
               temperature: float = 1.0) -> Tensor:
        is_logits = torch.any(probs_or_logits < -self.eps) or torch.any(probs_or_logits > 1 + self.eps)

        if noise is None:
            if temperature != 1.0:
                logits = probs_or_logits
                if not is_logits:
                    logits = torch.log(probs_or_logits.clamp(min=self.eps))
                sample = F.gumbel_softmax(logits, tau=temperature, hard=False, eps=self.eps, dim=-1)
                return sample.argmax(dim=-1).type(probs_or_logits.dtype)

            probs = probs_or_logits
            if is_logits:
                probs = torch.softmax(probs_or_logits, dim=-1)
            flat_probs = probs.view(-1, probs.size(-1))
            if probs.size(-1) == 2:
                return torch.bernoulli(flat_probs[..., 1]).view(probs.shape[:-1])
            return torch.multinomial(flat_probs, 1).view(probs.shape[:-1])

        if not torch.is_tensor(noise) and noise:
            noise = torch.rand_like(probs_or_logits)

        gumbel_noise = 0
        if torch.is_tensor(noise) and not torch.all(noise == 0):
            gumbel_noise = -torch.log(-torch.log(noise.clamp(self.eps, 1.0)))

        if is_logits:
            log_probs = torch.log_softmax(probs_or_logits, dim=-1)
            perturbed_logits = (log_probs + gumbel_noise) / temperature
            return perturbed_logits.argmax(dim=-1).type(probs_or_logits.dtype)

        log_probs = torch.log(probs_or_logits.clamp(min=self.eps))
        perturbed_logits = (log_probs + gumbel_noise) / temperature
        return perturbed_logits.argmax(dim=-1).type(probs_or_logits.dtype)

    def log1mexp(self, a: Tensor) -> Tensor:
        return torch.log((1 - a.exp()).clamp(min=self.eps))

    @staticmethod
    def log_add_exp(a, b):
        maximum = torch.max(a, b)
        return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

    def _prepare_step(self,
                      model_output: Tensor,
                      sample: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if model_output.size(1) == 1:
            pred_original_sample = (model_output > 0).type(model_output.dtype)
            probs = torch.sigmoid(model_output)
            x_start = torch.stack((1 - probs, probs), dim=-1)
        else:
            pred_original_sample = model_output.argmax(dim=1, keepdim=True).type(model_output.dtype)
            x_start = torch.softmax(model_output, dim=1).unsqueeze(-1).transpose(1, -1)

        if not torch.is_floating_point(model_output):
            x_start = F.one_hot(model_output.long(), self.config.num_classes).float()

        x_t_one_hot = F.one_hot(sample.long(), self.config.num_classes).float()

        assert x_start.shape == x_t_one_hot.shape, (x_start.shape, x_t_one_hot.shape)

        return x_start, x_t_one_hot, pred_original_sample

    def _step_simple(self,
                     model_output: Tensor,
                     timestep: Tensor,
                     sample: Tensor,
                     noise: Optional[Tensor] = None,
                     return_dict: bool = True) -> Union[DiscreteStateSchedulerOutput, Tuple]:
        num_classes = self.config.num_classes
        t = timestep.to(model_output.device)
        prev_t = self.previous_timestep(t).to(model_output.device)
        prev_t[prev_t < 0] = 0

        x_start, x_t_one_hot, pred_original_sample = self._prepare_step(model_output, sample)

        self.alphas = self.alphas.to(model_output.device)
        self.alphas_cumprod = self.alphas_cumprod.to(model_output.device)

        alpha = unsqueeze_as(self.alphas[t], x_t_one_hot)
        prev_alpha_prod = unsqueeze_as(self.alphas_cumprod[prev_t], x_start)

        probs1 = alpha * x_t_one_hot + (1 - alpha) / num_classes
        probs2 = prev_alpha_prod * x_start + (1 - prev_alpha_prod) / num_classes

        probs_prod = probs1 * probs2
        probs = probs_prod / probs_prod.sum(dim=-1, keepdim=True)

        pred_prev_sample = self.sample(probs, noise=True if noise is None else noise)
        pred_prev_sample = torch.where(unsqueeze_as(t, pred_prev_sample) == 0,
                                       x_start.argmax(dim=-1).type(x_start.dtype),
                                       pred_prev_sample)

        if not return_dict:
            return (pred_prev_sample,)

        return DiscreteStateSchedulerOutput(prev_sample=pred_prev_sample.type(sample.dtype),
                                            pred_original_sample=pred_original_sample.to(model_output.dtype))

    def _step_log(self,
                  model_output: Tensor,
                  timestep: Tensor,
                  sample: Tensor,
                  noise: Optional[Tensor] = None,
                  return_dict: bool = True) -> Union[DiscreteStateSchedulerOutput, Tuple]:
        num_classes = self.config.num_classes
        t = timestep.to(model_output.device)
        prev_t = self.previous_timestep(t).to(model_output.device)
        prev_t[prev_t < 0] = 0

        x_start, x_t_one_hot, pred_original_sample = self._prepare_step(model_output, sample)

        self.log_alphas = self.log_alphas.to(model_output.device)
        self.log_alphas_cumprod = self.log_alphas_cumprod.to(model_output.device)

        log_x_t = torch.log(x_t_one_hot.clamp(min=self.eps))
        log_alpha_t = unsqueeze_as(self.log_alphas[t], log_x_t)
        log_1_min_alpha_t = self.log1mexp(log_alpha_t)
        logits1 = self.log_add_exp(log_x_t + log_alpha_t, log_1_min_alpha_t - np.log(num_classes))

        log_x_start = torch.log(x_start.clamp(min=self.eps))
        log_alpha_prod = unsqueeze_as(self.log_alphas_cumprod[prev_t], log_x_start)
        log_1_min_alpha_prod = self.log1mexp(log_alpha_prod)
        logits2 = self.log_add_exp(log_x_start + log_alpha_prod, log_1_min_alpha_prod - np.log(num_classes))

        logits = logits1 + logits2
        logits = torch.where(unsqueeze_as(t, logits) == 0, log_x_start, logits)
        log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        pred_prev_sample = self.sample(log_probs, noise=True if noise is None else noise)
        pred_prev_sample = torch.where(unsqueeze_as(t, pred_prev_sample) == 0,
                                       x_start.argmax(dim=-1).type(x_start.dtype),
                                       pred_prev_sample)

        if not return_dict:
            return (pred_prev_sample,)

        return DiscreteStateSchedulerOutput(prev_sample=pred_prev_sample.type(sample.dtype),
                                            pred_original_sample=pred_original_sample.type(model_output.dtype),
                                            prev_logits=logits.type(sample.dtype),
                                            pred_original_logits=log_x_start.type(model_output.dtype))

    def _step_matrix(self,
                     model_output: Tensor,
                     timestep: Tensor,
                     sample: Tensor,
                     noise: Optional[Tensor] = None,
                     return_dict: bool = True) -> Union[DiscreteStateSchedulerOutput, Tuple]:
        t = timestep.to(model_output.device)
        prev_t = self.previous_timestep(t).to(model_output.device)
        prev_t[prev_t < 0] = 0

        x_start, x_t_one_hot, pred_original_sample = self._prepare_step(model_output, sample)

        self.alpha_mats = self.alphas_mats.to(model_output.device)
        self.alphas_cumprod_mats = self.alphas_cumprod_mats.to(model_output.device)

        probs1 = torch.bmm(x_t_one_hot.view(x_t_one_hot.size(0), -1, x_t_one_hot.size(-1)),
                           self.alphas_mats[t].transpose(1, 2).to(x_t_one_hot.dtype)).view_as(x_t_one_hot)
        probs2 = torch.bmm(x_start.view(x_start.size(0), -1, x_start.size(-1)),
                           self.alphas_cumprod_mats[prev_t].to(x_start.dtype)).view_as(x_start)

        probs_prod = probs1 * probs2
        probs = probs_prod / probs_prod.sum(dim=-1, keepdim=True)

        pred_prev_sample = self.sample(probs, noise=True if noise is None else noise)
        pred_prev_sample = torch.where(unsqueeze_as(t, pred_prev_sample) == 0,
                                       x_start.argmax(dim=-1).type(x_start.dtype),
                                       pred_prev_sample)

        if not return_dict:
            return (pred_prev_sample,)

        return DiscreteStateSchedulerOutput(prev_sample=pred_prev_sample.type(sample.dtype),
                                            pred_original_sample=pred_original_sample.to(model_output.dtype))

    def step(self,
             model_output: Tensor,
             timestep: Union[int, Tensor],
             sample: Tensor,
             noise: Optional[Tensor] = None,
             return_dict: bool = True,
             implementation: Optional[str] = None) -> Union[DiscreteStateSchedulerOutput, Tuple]:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep])
        if (implementation or self.implementation) == "simple":
            return self._step_simple(model_output, timestep, sample, noise, return_dict)
        elif (implementation or self.implementation) == "log":
            return self._step_log(model_output, timestep, sample, noise, return_dict)
        elif (implementation or self.implementation) == "matrix":
            return self._step_matrix(model_output, timestep, sample, noise, return_dict)
        else:
            raise NotImplementedError(f"Implementation {self.config.implementation} not supported yet")

    def _prepare_add_noise(self,
                           original_samples: Tensor,
                           noise: Optional[Tensor]) -> Tensor:
        num_classes = self.config.num_classes
        classes = torch.arange(num_classes, device=original_samples.device)
        unique_vals = torch.unique(original_samples)
        assert torch.all(torch.isin(unique_vals, classes)), f"`original_samples` must be in {num_classes}"

        x_start = F.one_hot(original_samples.long(), num_classes).float()

        if noise is not None:
            assert noise.min().item() >= 0 and noise.max().item() <= 1, "`noise` must be in [0, 1]"
            assert noise.size() == x_start.size(), \
                f"Expected noise to have shape {(*x_start.size(),)}, got {(*noise.size(),)}"

        return x_start

    def _add_noise_simple(self,
                          original_samples: Tensor,
                          noise: Optional[Tensor],
                          timesteps: Tensor) -> Tuple[Tensor, Tensor]:
        x_start = self._prepare_add_noise(original_samples, noise)

        self.alphas_cumprod = self.alphas_cumprod.to(original_samples.device)
        timesteps = timesteps.to(original_samples.device)
        alpha_prod = unsqueeze_as(self.alphas_cumprod[timesteps], x_start)

        probs = alpha_prod * x_start + (1 - alpha_prod) / self.config.num_classes

        return self.sample(probs, noise).type(original_samples.dtype), probs

    def _add_noise_log(self,
                       original_samples: Tensor,
                       noise: Optional[Tensor],
                       timesteps: Tensor) -> Tuple[Tensor, Tensor]:
        x_start = self._prepare_add_noise(original_samples, noise)
        log_x_start = torch.log(x_start.clamp(min=self.eps))

        self.log_alphas_cumprod = self.log_alphas_cumprod.to(original_samples.device)
        timesteps = timesteps.to(original_samples.device)
        log_alpha_prod = unsqueeze_as(self.log_alphas_cumprod[timesteps], log_x_start)
        log_1_min_alpha_prod = self.log1mexp(log_alpha_prod)

        log_probs = self.log_add_exp(log_x_start + log_alpha_prod,
                                     log_1_min_alpha_prod - np.log(self.config.num_classes))

        return self.sample(log_probs, noise).type(original_samples.dtype), log_probs

    def _add_noise_matrix(self,
                          original_samples: Tensor,
                          noise: Optional[Tensor],
                          timesteps: Tensor) -> Tuple[Tensor, Tensor]:
        x_start = self._prepare_add_noise(original_samples, noise)

        self.alphas_cumprod_mats = self.alphas_cumprod_mats.to(original_samples.device)
        timesteps = timesteps.to(original_samples.device)
        alpha_prod_mat = self.alphas_cumprod_mats[timesteps]

        probs = torch.bmm(x_start.view(x_start.size(0), -1, x_start.size(-1)),
                          alpha_prod_mat.to(x_start.dtype)).view_as(x_start)

        return self.sample(probs, noise).type(original_samples.dtype), probs

    def add_noise(self,
                  original_samples: Tensor,
                  noise: Optional[Tensor],
                  timesteps: Tensor,
                  implementation: Optional[str] = None) -> Tensor:
        if (implementation or self.implementation) == "simple":
            return self._add_noise_simple(original_samples, noise, timesteps)[0]
        elif (implementation or self.implementation) == "log":
            return self._add_noise_log(original_samples, noise, timesteps)[0]
        elif (implementation or self.implementation) == "matrix":
            return self._add_noise_matrix(original_samples, noise, timesteps)[0]
        else:
            raise NotImplementedError(f"Unsupported implementation: {implementation}")

    def __len__(self):
        return self.config.num_train_timesteps

    def previous_timestep(self, timestep):
        if self.custom_timesteps:
            index = (self.timesteps == timestep).nonzero(as_tuple=True)[0][0]
            if index == self.timesteps.shape[0] - 1:
                prev_t = torch.tensor(-1)
            else:
                prev_t = self.timesteps[index + 1]
        else:
            num_inference_steps = (
                self.num_inference_steps if self.num_inference_steps else self.config.num_train_timesteps
            )
            prev_t = timestep - self.config.num_train_timesteps // num_inference_steps

        return prev_t
