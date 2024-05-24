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
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from ..utils.torch_utils import randn_tensor
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

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None


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
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
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
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        variance_type: str = "fixed_small",
        clip_sample: bool = True,
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "leading",
        steps_offset: int = 0,
        rescale_betas_zero_snr: int = False,
    ):
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        elif beta_schedule == "sigmoid":
            # GeoDiff sigmoid schedule
            betas = torch.linspace(-6, 6, num_train_timesteps)
            self.betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        # Rescale for zero SNR
        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.log_alphas = torch.log(self.alphas)
        self.log_alphas_cumprod = torch.cumsum(self.log_alphas, dim=0)

        alpha_mats = []
        for beta in self.betas:
            mat = torch.ones(num_classes, num_classes) * beta / num_classes
            mat.diagonal().fill_(1 - (num_classes - 1) * beta / num_classes)
            alpha_mats.append(mat)
        self.alpha_mats = torch.stack(alpha_mats)

        alpha_mat_t = alpha_mats[0]
        alpha_bar_mats = [alpha_mat_t]
        for idx in range(1, self.num_train_timesteps):
            alpha_mat_t = alpha_mat_t @ alpha_mats[idx]
            alpha_bar_mats.append(alpha_mat_t)
        self.alpha_bar_mats = torch.stack(alpha_bar_mats)

        self.eps = 1e-6
        self.one = torch.tensor(1.0)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.custom_timesteps = False
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

        self.variance_type = variance_type

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

    def _at(self, a: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Extract coefficients at specified timesteps t and conditioning data x.

        Args:
          a: torch.Tensor: 3D tensor of constants indexed by time.
          t: torch.Tensor: 1D tensor of time indices, shape = (batch_size,).
          x: torch.Tensor: tensor of shape (batch_size, ...).

        Returns:
          torch.Tensor: Extracted coefficients.
        """
        t_broadcast = t.view(-1, *([1] * (x.ndim - 1)))
        return a[t_broadcast, x]

    def _at_onehot(self, a: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Extract coefficients at specified timesteps t and conditioning data x.

        Args:
          a: torch.Tensor: 3D tensor of constants indexed by time.
          t: torch.Tensor: 1D tensor of time indices, shape = (batch_size,).
          x: torch.Tensor: tensor of shape (batch_size, ..., num_pixel_vals).

        Returns:
          torch.Tensor: Output of dot(x, a[t], axis=[[-1], [1]]).
        """
        a_t = a[t]
        return torch.matmul(x, a_t)

    def q_posterior_logits(self,
                           x_start: Tensor,
                           x_t: Tensor,
                           t: Tensor,
                           x_start_logits: bool) -> Tensor:
        """
        Compute logits of q(x_{t-1} | x_t, x_start).

        Args:
          x_start: torch.Tensor: The starting data.
          x_t: torch.Tensor: The data at time t.
          t: torch.Tensor: Time indices.
          x_start_logits: bool: Whether x_start is in logits format.

        Returns:
          torch.Tensor: Logits of q(x_{t-1} | x_t, x_start).
        """
        if x_start_logits:
            assert x_start.shape == x_t.shape + (self.config.num_classes,), (x_start.shape, x_t.shape)
        else:
            assert x_start.shape == x_t.shape, (x_start.shape, x_t.shape)

        prev_t = self.previous_timestep(t)  # TODO: Or t - 1?

        fact1 = self._at(self.alpha_mats.transpose(1, 2), t, x_t)
        if x_start_logits:
            fact2 = self._at_onehot(self.alpha_bar_mats, prev_t, F.softmax(x_start, dim=-1))
            tzero_logits = x_start
        else:
            fact2 = self._at(self.alpha_bar_mats, prev_t, x_start)
            tzero_logits = torch.log(F.one_hot(x_start, num_classes=self.config.num_classes).float() + self.eps)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        t_broadcast = t.view(-1, *([1] * (out.ndim - 1)))
        return torch.where(t_broadcast == 0, tzero_logits, out)


    def p_logits(self, x_start: Tensor, t: Tensor, x_t: Tensor) -> Tensor:
        """
        Compute logits of p(x_{t-1} | x_t).

        Args:
          model_fn: Callable that returns the logits for x_start.
          x: torch.Tensor: The data at time t.
          t: torch.Tensor: Time indices.

        Returns:
          torch.Tensor: Logits of p(x_{t-1} | x_t).
        """
        t_broadcast = t.view(-1, *([1] * (x_start.ndim - 1)))
        model_logits = torch.where(
            t_broadcast == 0,
            x_start,
            self.q_posterior_logits(x_start, x_t, t, x_start_logits=True)
        )
        return model_logits

    def step(self,
             model_output: Tensor,
             timestep: int,
             sample: Tensor) -> DiscreteStateSchedulerOutput:
        self.alpha_mats = self.alpha_mats.to(device=model_output.device)
        self.alpha_bar_mats = self.alpha_bar_mats.to(device=model_output.device)

        probs = torch.sigmoid(model_output)
        x_0_one_hot = torch.stack([1 - probs, probs], dim=-1)
        log_x_start = torch.log(x_0_one_hot + self.eps)

        logits = self.p_logits(log_x_start.squeeze(1), timestep, sample.squeeze(1).long()).unsqueeze(1)
        if timestep > 0:
            noise = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(noise + self.eps) + self.eps)
            logits += gumbel_noise

        return DiscreteStateSchedulerOutput(prev_sample=logits.argmax(dim=-1).to(sample.dtype),
                                            pred_original_sample=(model_output > 0).to(model_output.dtype))

    def log_1_min_a(self, a):
        return torch.log(1 - a.exp() + self.eps)

    @staticmethod
    def log_add_exp(a, b):
        maximum = torch.max(a, b)
        return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

    def index_to_log_onehot(self, x, num_classes):
        assert x.max().item() < num_classes, \
            f'Error: {x.max().item()} >= {num_classes}'
        x_onehot = F.one_hot(x, num_classes)

        permute_order = (0, -1) + tuple(range(1, len(x.size())))

        x_onehot = x_onehot.permute(permute_order)

        log_x = torch.log(x_onehot.float().clamp(min=self.eps))

        return log_x

    def log_sample_categorical(self, logits):
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + self.eps) + self.eps)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = self.index_to_log_onehot(sample, self.config.num_classes)
        return log_sample

    def q_pred_one_timestep(self, log_x_t, t):
        # log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_alpha_t = self.log_alphas[t].view(-1, *((1,) * (log_x_t.ndim - 1)))
        # log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = self.log_1_min_a(self.log_alphas[t]).view(-1, *((1,) * (log_x_t.ndim - 1)))

        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        log_probs = self.log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - np.log(self.config.num_classes)
        )

        return log_probs

    def q_pred(self, log_x_start, t):
        # log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_cumprod_alpha_t = self.log_alphas_cumprod[t].view(-1, *((1,) * (log_x_start.ndim - 1)))
        # log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = self.log_1_min_a(self.log_alphas_cumprod[t]).view(-1, *((1,) * (log_x_start.ndim - 1)))

        log_probs = self.log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.config.num_classes)
        )

        return log_probs

    def q_posterior(self, log_x_start, log_x_t, t):
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).

        # t_minus_1 = t - 1
        t_minus_1 = self.previous_timestep(t)  # TODO: Check if this is correct

        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0)


        # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
        # Not very easy to see why this is true. But it is :)
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)

        log_EV_xtmin_given_xt_given_xstart = \
            unnormed_logprobs \
            - torch.logsumexp(unnormed_logprobs, dim=1, keepdim=True)

        return log_EV_xtmin_given_xt_given_xstart

    def step1(self, model_output, t, sample):
        self.log_alphas = self.log_alphas.to(device=model_output.device)
        self.log_alphas_cumprod = self.log_alphas_cumprod.to(device=model_output.device)

        probs = torch.sigmoid(model_output)
        x_0_one_hot = torch.cat([1 - probs, probs], dim=1)
        log_x_start = torch.log(x_0_one_hot + self.eps)
        x_t_one_hot = F.one_hot(sample.long().view(sample.size(0), -1), self.config.num_classes).float().transpose(1, 2)
        log_x_t= torch.log(x_t_one_hot + self.eps)  # FIXME: Check if this is correct

        out = self.q_posterior(log_x_start=log_x_start, log_x_t=log_x_t, t=t)
        if t > 0:
            out = self.log_sample_categorical(out)

        return DiscreteStateSchedulerOutput(prev_sample=out.argmax(dim=1, keepdim=True).to(sample.dtype),
                                            pred_original_sample=(model_output > 0).to(model_output.dtype))

    def step2(
        self,
        model_output: Tensor,
        timestep: int,
        sample: Tensor,
        return_dict: bool = True
    ) -> Union[DiscreteStateSchedulerOutput, Tuple]:
        t = timestep
        prev_t = self.previous_timestep(t)

        num_classes = self.config.num_classes

        if model_output.size(1) == 1 and num_classes == 2:
            pred_original_sample = (model_output > 0).to(dtype=model_output.dtype)
            probs = torch.sigmoid(model_output)
            x_0_one_hot = torch.cat([1 - probs, probs], dim=1)
        else:
            pred_original_sample = model_output.argmax(dim=1).to(dtype=model_output.dtype)
            x_0_one_hot = torch.softmax(model_output, dim=1)

        x_t_one_hot = F.one_hot(sample.long().view(sample.size(0), -1), num_classes).float().transpose(1, 2)

        self.alphas = self.alphas.to(device=model_output.device)
        self.alphas_cumprod = self.alphas_cumprod.to(device=model_output.device)

        alpha_t = self.alphas[t].view(-1, *((1,) * (x_t_one_hot.ndim - 1)))
        alpha_cumprod_t_minus_1 = self.alphas_cumprod[prev_t].view(-1, *((1,) * (x_0_one_hot.ndim - 1))) if t > 0 else torch.ones_like(x_0_one_hot)

        theta_t = alpha_t * x_t_one_hot + (1 - alpha_t) / num_classes
        theta_t_minus_1 = alpha_cumprod_t_minus_1 * x_0_one_hot + (1 - alpha_cumprod_t_minus_1) / num_classes

        posterior_numerator = theta_t * theta_t_minus_1
        posterior_denominator = posterior_numerator.sum(dim=1, keepdim=True)

        p = posterior_numerator / posterior_denominator
        pred_prev_sample = self.sample(p, noise=(t > 0).item())

        if not return_dict:
            return (pred_prev_sample,)

        return DiscreteStateSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)

    def sample(self,
               probs_or_logits: Tensor,
               noise: Optional[Union[bool, Tensor]] = None,
               temperature: float = 1.0) -> Tensor:
        is_logits = torch.any(probs_or_logits < -self.eps) or torch.any(probs_or_logits > 1 + self.eps)

        if noise is None:
            if is_logits:
                probs_or_logits = torch.softmax(probs_or_logits, dim=1)
            return torch.multinomial(probs_or_logits, 1)

        if not torch.is_tensor(noise) and noise:
            noise = torch.rand_like(probs_or_logits)

        gumbel_noise = 0
        if torch.is_tensor(noise):
            gumbel_noise = -torch.log(-torch.log(noise + self.eps) + self.eps)

        if is_logits:
            return (probs_or_logits + gumbel_noise).argmax(dim=1, keepdim=True).to(dtype=probs_or_logits.dtype)

        logits = torch.log(probs_or_logits + self.eps)
        return (logits + gumbel_noise).argmax(dim=1, keepdim=True).to(dtype=probs_or_logits.dtype)
        # return F.gumbel_softmax(logits).argmax(dim=-1)

    def add_noise(self,
                  original_samples: Tensor,
                  noise: Optional[Tensor],
                  timesteps: Tensor) -> Tensor:
        num_classes = self.config.num_classes
        classes = torch.arange(num_classes, device=original_samples.device)
        unique_vals = torch.unique(original_samples)

        assert torch.all(torch.isin(unique_vals, classes)), f"`original_samples` must be in {num_classes}"
        # assert torch.all(torch.isin(torch.unique(noise), torch.arange(num_classes))), \
        #     f"`noise` must be in {num_classes}"
        if noise is not None:
            assert noise.min().item() >= 0 and noise.max().item() <= 1, "`noise` must be in [0, 1]"
        # assert original_samples.size() == noise.size(), \
        #     "`original_samples` and `noise` must have the same shape"

        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        # alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        x_0_one_hot = F.one_hot(original_samples.long().view(original_samples.size(0), -1), num_classes).float().transpose(1, 2)
        alpha_prod = self.alphas_cumprod[timesteps].view(-1, *((1,) * (x_0_one_hot.ndim - 1))).float()

        p = alpha_prod * x_0_one_hot + (1 - alpha_prod) / num_classes

        # noise_one_hot = F.one_hot(noise.long(), num_classes).float()
        return self.sample(p, noise).to(dtype=original_samples.dtype)

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
