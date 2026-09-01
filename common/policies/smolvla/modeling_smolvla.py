#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
python lerobot/scripts/train.py \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
python lerobot/scripts/train.py \
--policy.type=smolvla \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

import math
import copy
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoProcessor

from lerobot.common.constants import ACTION, OBS_EFFORT, OBS_STATE
from lerobot.common.policies.normalize import (
    Normalize,
    Unnormalize,
)
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from lerobot.common.policies.tactile.encoder import TactileTokenEncoder
from lerobot.common.policies.utils import (
    populate_queues,
)
from lerobot.common.utils.utils import get_safe_dtype
try:
    from lerobot.force_vqvae.models import ForceVQVAE, ForceVQVAEConfig
except ModuleNotFoundError:  # Optional legacy component; not used by TacForce-VLA.
    ForceVQVAE = None
    ForceVQVAEConfig = None
from lerobot.tacforce_wm.models.dynamics.frozen import FrozenTacForceDynamics

OBS_TACTILE = "observation.tactile"


class TacForceCrossAttention(nn.Module):
    """Fuse frozen current and world-model-predicted tactile latent tokens."""

    def __init__(self, dim: int, heads: int, max_steps: int, dropout: float):
        super().__init__()
        if dim % heads:
            raise ValueError(f"TacForce token dim {dim} must be divisible by heads={heads}.")
        self.max_steps = int(max_steps)
        self.pos = nn.Parameter(torch.randn(1, self.max_steps, dim) * 0.02)
        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)
        self.v_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * dim, dim), nn.Dropout(dropout),
        )

    def forward(self, current: Tensor, future: Tensor) -> Tensor:
        if current.shape != future.shape or current.ndim != 3:
            raise ValueError(f"Expected matching [B,T,D] TacForce latents, got {current.shape}/{future.shape}.")
        steps = current.shape[1]
        if steps > self.max_steps:
            raise ValueError(f"TacForce history {steps} exceeds max_steps={self.max_steps}.")
        pos = self.pos[:, :steps].to(dtype=current.dtype)
        context, _ = self.attn(
            self.q_norm(current + pos), self.k_norm(future + pos), self.v_norm(future + pos)
        )
        fused = current + context
        return fused + self.ff(self.ff_norm(fused))


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def pad_sequence(vector, new_len):
    """Pads or trims the sequence dimension to keep the most recent values."""
    if vector.shape[1] == new_len:
        return vector
    if vector.shape[1] > new_len:
        return vector[:, -new_len:, :]

    shape = list(vector.shape)
    current_len = shape[1]
    shape[1] = new_len
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[:, -current_len:, :] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class SmolVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLAConfig
    name = "smolvla"

    def __init__(
        self,
        config: SmolVLAConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.language_tokenizer = AutoProcessor.from_pretrained(self.config.vlm_model_name).tokenizer
        self.model = VLAFlowMatching(config)
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.effort_type.endswith("_his_c") or self.config.effort_type.endswith("_his_t"):
            effort_queue_len = (
                self.config.force_vqvae_window
                if self.config.effort_tokenizer == "force_vqvae"
                else self.config.effort_history_steps
            )
            self._queues[self.config.effort_key] = deque(maxlen=effort_queue_len)
            for effort_key in (OBS_EFFORT, "effort", "force"):
                if effort_key != self.config.effort_key:
                    self._queues[effort_key] = deque(maxlen=effort_queue_len)
        self._force_refine_state = None
        self._tacforce_tactile_queues = {
            key: deque(maxlen=self.config.tacforce_wm_history_steps)
            for key in (self.config.tactile_features or [])
        }
        self._tacforce_force_queue = deque(maxlen=self.config.tacforce_wm_history_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _raw_effort_for_tokenizer(self, batch: dict[str, Tensor]) -> tuple[str, Tensor] | None:
        if self.config.effort_tokenizer != "force_vqvae":
            return None
        for candidate_key in (self.config.effort_key, "force", OBS_EFFORT, "effort"):
            if candidate_key in batch:
                return candidate_key, batch[candidate_key].clone()
        return None

    def _restore_raw_effort_for_tokenizer(
        self, batch: dict[str, Tensor], raw_effort: tuple[str, Tensor] | None
    ) -> None:
        if raw_effort is None:
            return
        effort_key, effort = raw_effort
        batch[effort_key] = effort

    def _unnormalize_effort_target(self, effort: Tensor, effort_key: str | None) -> Tensor:
        if effort_key is None:
            return effort
        buffer_name = "buffer_" + effort_key.replace(".", "_")
        buffer = getattr(self.normalize_inputs, buffer_name, None)
        if buffer is None or "mean" not in buffer or "std" not in buffer:
            return effort
        mean = buffer["mean"].to(device=effort.device, dtype=effort.dtype)
        std = buffer["std"].to(device=effort.device, dtype=effort.dtype)
        if torch.isinf(mean).any() or torch.isinf(std).any():
            return effort
        return effort * (std + 1e-8) + mean

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()
        raw_tactile, raw_force = self._raw_tacforce_inputs(batch, inference=True)

        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        raw_effort = self._raw_effort_for_tokenizer(batch)
        batch = self.normalize_inputs(batch)
        self._restore_raw_effort_for_tokenizer(batch, raw_effort)

        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._queues[ACTION]) == 0:
            for k in batch:
                if k in self._queues:
                    batch[k] = torch.stack(list(self._queues[k]), dim=1)
            images, img_masks = self.prepare_images(batch)
            state = self.prepare_state(batch)
            effort = self.prepare_effort(batch)
            tactile_data = self._extract_tactile_data(batch)
            tactile_tokens = self.model.encode_tacforce_refine_tokens(
                raw_tactile if self.config.tacforce_wm_enabled else tactile_data, raw_force
            )
            lang_tokens, lang_masks = self.prepare_language(batch)

            if self.config.force_refine_enabled:
                actions, self._force_refine_state = self.model.sample_actions_for_force_refine(
                    images,
                    img_masks,
                    lang_tokens,
                    lang_masks,
                    state,
                    effort=effort,
                    tactile_tokens=tactile_tokens,
                    noise=noise,
                )
            else:
                actions = self.model.sample_actions(
                    images, img_masks, lang_tokens, lang_masks, state, effort=effort, noise=noise
                )
            # Unpad actions
            original_action_dim = self.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]

            actions = self.unnormalize_outputs({"action": actions})["action"]

            if self.config.adapt_to_pi_aloha:
                actions = self._pi_aloha_encode_actions(actions)

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()

    @torch.no_grad
    def refine_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Refine the not-yet-executed actions in the current chunk with fresh high-rate force readings."""
        if not self.config.force_refine_enabled:
            raise RuntimeError("`refine_action_chunk` requires `policy.force_refine_enabled=True`.")
        if self._force_refine_state is None:
            raise RuntimeError("No force-refine cache is available. Call `select_action` once to start a chunk.")
        if len(self._queues[ACTION]) == 0:
            raise RuntimeError("No queued actions remain to refine.")

        self.eval()
        raw_tactile, raw_force = self._raw_tacforce_inputs(batch, inference=True)
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
        raw_effort = self._raw_effort_for_tokenizer(batch)
        batch = self.normalize_inputs(batch)
        self._restore_raw_effort_for_tokenizer(batch, raw_effort)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        effort = self.prepare_effort(batch)
        tactile_data = self._extract_tactile_data(batch)
        tactile_tokens = self.model.encode_tacforce_refine_tokens(
            raw_tactile if self.config.tacforce_wm_enabled else tactile_data, raw_force
        )
        refined_actions = self.model.refine_actions_from_force(
            self._force_refine_state,
            effort=effort,
            tactile_tokens=tactile_tokens,
        )

        original_action_dim = self.config.action_feature.shape[0]
        refined_actions = refined_actions[:, :, :original_action_dim]
        refined_actions = self.unnormalize_outputs({"action": refined_actions})["action"]
        if self.config.adapt_to_pi_aloha:
            refined_actions = self._pi_aloha_encode_actions(refined_actions)

        executed_steps = self.config.n_action_steps - len(self._queues[ACTION])
        remaining_actions = refined_actions[:, executed_steps : self.config.n_action_steps]
        self._queues[ACTION].clear()
        self._queues[ACTION].extend(remaining_actions.transpose(0, 1))
        return remaining_actions

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss"""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])
        raw_effort = self._raw_effort_for_tokenizer(batch)
        raw_tactile, raw_force = self._raw_tacforce_inputs(batch, inference=False)

        effort_key = self.config.effort_key
        if effort_key in batch and not hasattr(self, "_debug_raw_effort_printed"):
            e = batch[effort_key].detach()
            reduce_dim = (0, 1) if e.ndim == 3 else 0
            print("RAW? batch effort mean:", e.mean(dim=reduce_dim).cpu())
            print("RAW? batch effort min:", e.amin(dim=reduce_dim).cpu())
            print("RAW? batch effort max:", e.amax(dim=reduce_dim).cpu())
            self._debug_raw_effort_printed = True

        raw_future_effort = self.prepare_future_effort(batch)
        batch = self.normalize_inputs(batch)
        self._restore_raw_effort_for_tokenizer(batch, raw_effort)
        batch = self.normalize_targets(batch)
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        effort = self.prepare_effort(batch)
        tactile_data = self._extract_tactile_data(batch)
        tactile_tokens = self.model.encode_tacforce_refine_tokens(
            raw_tactile if self.config.tacforce_wm_enabled else tactile_data, raw_force
        )
        future_effort = raw_future_effort
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")
        loss_dict = {}
        model_losses = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            effort,
            noise=noise,
            time=time,
            future_effort=future_effort,
            tactile_tokens=tactile_tokens,
        )
        force_prediction_loss = None
        if isinstance(model_losses, tuple):
            if len(model_losses) == 3:
                losses, force_refine_losses, force_prediction_loss = model_losses
            else:
                losses, force_refine_losses = model_losses
        else:
            losses = model_losses
            force_refine_losses = None
        loss_dict["losses_after_forward"] = losses.clone()
        if force_refine_losses is not None:
            loss_dict["force_refine_losses_after_forward"] = force_refine_losses.clone()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()
            if force_refine_losses is not None:
                force_refine_losses = force_refine_losses * in_episode_bound.unsqueeze(-1)
                loss_dict["force_refine_losses_after_in_ep_bound"] = force_refine_losses.clone()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone()
        if force_refine_losses is not None:
            force_refine_losses = force_refine_losses[:, :, : self.config.max_action_dim]
            loss_dict["force_refine_losses_after_rm_padding"] = force_refine_losses.clone()

        # For backward pass
        action_loss = losses.mean()
        loss_dict["action_loss"] = action_loss.item()
        loss = action_loss
        if force_refine_losses is not None:
            force_refine_loss = force_refine_losses.mean()
            loss_dict["force_refine_loss"] = force_refine_loss.item()
            loss = loss + self.config.force_refine_loss_weight * force_refine_loss
        if force_prediction_loss is not None:
            loss_dict["force_prediction_loss"] = force_prediction_loss.item()
            loss = loss + self.config.force_prediction_loss_weight * force_prediction_loss
        if self.config.force_prediction_enabled:
            loss_dict["force_prediction_has_target"] = float(force_prediction_loss is not None)
        # For backward pass
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """Tokenize the text input"""
        device = batch[OBS_STATE].device
        tasks = batch["task"]
        if len(tasks) == 1:
            tasks = [tasks[0] for _ in range(batch[OBS_STATE].shape[0])]

        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]
        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding=self.config.pad_language_to,
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def _extract_tactile_data(self, batch: dict[str, Tensor]) -> list[Tensor] | None:
        """Extract left/right Tac3D distributed-force tensors for force refinement."""
        if not self.config.use_tactile:
            return None
        tactile_keys = self.config.tactile_features if self.config.tactile_features else [OBS_TACTILE]
        tactile_data = [self._prepare_tactile_tensor(batch[k]) for k in tactile_keys if k in batch]
        if len(tactile_data) == 0:
            raise ValueError(
                f"`use_tactile=True` but none of the tactile keys were found in the batch: {tactile_keys}."
            )
        return tactile_data

    def _raw_tacforce_inputs(
        self, batch: dict[str, Tensor], *, inference: bool
    ) -> tuple[list[Tensor] | None, Tensor | None]:
        """Return raw contiguous histories; inference queues use first-frame left padding."""
        if not self.config.tacforce_wm_enabled:
            return None, None
        tactile_keys = self.config.tactile_features or []
        if len(tactile_keys) != 2 or any(key not in batch for key in tactile_keys):
            raise KeyError(f"TacForce-WM requires both tactile keys {tactile_keys}.")
        force_key = next(
            (key for key in (self.config.effort_key, "force", OBS_EFFORT, "effort") if key in batch),
            None,
        )
        if force_key is None:
            raise KeyError(f"TacForce-WM requires raw force input `{self.config.effort_key}`.")

        if not inference:
            tactile = [batch[key] for key in tactile_keys]
            force = batch[force_key]
            expected = self.config.tacforce_wm_history_steps
            if any(x.ndim != 4 or x.shape[1] != expected for x in tactile):
                raise ValueError(f"Training tactile inputs must contain exactly {expected} contiguous frames.")
            if force.ndim != 3 or force.shape[1] != expected:
                raise ValueError(f"Training force input must be [B,{expected},6], got {force.shape}.")
            return tactile, force

        for key in tactile_keys:
            value = batch[key]
            if value.ndim == 4:
                value = value[:, -1]
            queue = self._tacforce_tactile_queues[key]
            if not queue:
                queue.extend([value] * queue.maxlen)
            else:
                queue.append(value)
        force = batch[force_key]
        if force.ndim == 3:
            force = force[:, -1]
        if not self._tacforce_force_queue:
            self._tacforce_force_queue.extend([force] * self._tacforce_force_queue.maxlen)
        else:
            self._tacforce_force_queue.append(force)
        return (
            [torch.stack(list(self._tacforce_tactile_queues[key]), dim=1) for key in tactile_keys],
            torch.stack(list(self._tacforce_force_queue), dim=1),
        )

    def _prepare_tactile_tensor(self, tactile: Tensor) -> Tensor:
        """Use the latest tactile frame and keep the Tac3D raw shape for the encoder."""
        raw_ndim = len(self.config.tactile_raw_shape)
        if self.config.tacforce_wm_enabled and tactile.ndim == raw_ndim + 2:
            return tactile
        if tactile.ndim == raw_ndim + 2 and not self.config.tacforce_wm_enabled:
            tactile = tactile[:, -1]
        if tactile.ndim != raw_ndim + 1:
            raise ValueError(
                "Tac3D tactile tensors must have shape "
                f"(B, *{self.config.tactile_raw_shape}) or (B, T, *{self.config.tactile_raw_shape}); "
                f"got {tactile.shape}."
            )
        return tactile

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        if self.config.effort_type == "state":
            effort = self.prepare_effort(batch)
            state = torch.cat([state, effort[:, -1, :]], dim=-1)
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_effort(self, batch):
        """Prepare end-effector force/effort history as (batch_size, history, effort_dim)."""
        if self.config.effort_type in {"none", "no"}:
            return None

        effort_key = None
        for candidate_key in (self.config.effort_key, "force", OBS_EFFORT, "effort"):
            if candidate_key in batch:
                effort_key = candidate_key
                break
        if effort_key is None:
            raise ValueError(
                f"`effort_type={self.config.effort_type}` requires one of "
                f"`{self.config.effort_key}`, `force`, `{OBS_EFFORT}`, or `effort` in the batch."
            )

        effort = batch[effort_key]
        if effort.ndim == 2:
            effort = effort[:, None, :]
        elif effort.ndim != 3:
            raise ValueError(f"Effort is expected to have shape (B, D) or (B, T, D), got {effort.shape}.")
        effort_pad_key = f"{effort_key}_is_pad"
        if effort_pad_key in batch:
            effort = effort.masked_fill(batch[effort_pad_key].to(device=effort.device).unsqueeze(-1), 0.0)

        effort = pad_vector(effort, self.config.effort_dim)
        history_steps = self.config.effort_history_steps
        if self.config.effort_tokenizer == "force_vqvae":
            history_steps = self.config.force_vqvae_window
        elif self.config.effort_type in {"llm", "expert", "state"}:
            history_steps = 1
        effort = effort[:, :history_steps, :]
        return pad_sequence(effort, history_steps)

    def prepare_future_effort(self, batch):
        """Prepare future force label for force prediction auxiliary training."""
        if not self.config.force_prediction_enabled:
            return None

        effort_key = None
        for candidate_key in (self.config.effort_key, "force", OBS_EFFORT, "effort"):
            if candidate_key in batch:
                effort_key = candidate_key
                break

        future_effort = None
        if effort_key is not None:
            effort = batch[effort_key]
            if effort.ndim == 3:
                history_steps = self.config.effort_history_steps
                if self.config.effort_tokenizer == "force_vqvae":
                    history_steps = self.config.force_vqvae_window
                elif self.config.effort_type in {"llm", "expert", "state"}:
                    history_steps = 1
                if effort.shape[1] > history_steps - 1:
                    start = history_steps - 1
                    end = start + self.config.chunk_size
                    future_effort = effort[:, start:end, :]
            elif effort.ndim != 2:
                raise ValueError(f"Effort is expected to have shape (B, D) or (B, T, D), got {effort.shape}.")

        if future_effort is None:
            key = self.config.force_prediction_target_key
            if key not in batch:
                return None
            future_effort = batch[key]
            if future_effort.ndim == 2:
                future_effort = future_effort[:, None, :]
            elif future_effort.ndim != 3:
                raise ValueError(
                    f"Future effort is expected to have shape (B, D) or (B, T, D), got {future_effort.shape}."
                )
            future_effort = self._unnormalize_effort_target(future_effort, effort_key)

        if future_effort.ndim == 2:
            future_effort = future_effort[:, None, :]
        elif future_effort.ndim != 3:
            raise ValueError(
                f"Future effort is expected to have shape (B, D) or (B, T, D), got {future_effort.shape}."
            )
        future_effort = pad_vector(future_effort, self.config.effort_dim)
        future_effort = pad_sequence(future_effort, self.config.chunk_size)
        return future_effort

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class VLAFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.effort_type = self.config.effort_type
        self.effort_tokenizer = self.config.effort_tokenizer

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        if self.config.use_tactile and not self.config.tacforce_wm_enabled:
            self.tactile_encoder = TactileTokenEncoder(
                encoder_type=self.config.tactile_encoder_type,
                input_shape=self.config.tactile_input_shape,
                input_channels=self.config.tactile_input_channels,
                raw_shape=self.config.tactile_raw_shape,
                feature_dim=self.config.tactile_feature_dim,
                n_tokens=self.config.n_tactile_tokens,
                dropout=self.config.tactile_dropout,
            )
            self.tactile_proj = nn.Linear(
                self.config.tactile_feature_dim,
                self.vlm_with_expert.expert_hidden_size,
            )
        else:
            self.tactile_encoder = None
            self.tactile_proj = None
        if self.config.tacforce_wm_enabled:
            self.tacforce_dynamics = FrozenTacForceDynamics(
                {
                    "config_path": self.config.tacforce_wm_config_path,
                    "ckpt_path": self.config.tacforce_wm_ckpt_path,
                    "normalize_keys": ["tactile", "force_4x"],
                },
                device="cpu",
            )
            latent_dim = self.tacforce_dynamics.latent_dim
            expert_dim = self.vlm_with_expert.expert_hidden_size
            self.tacforce_current_proj = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, expert_dim), nn.SiLU())
            self.tacforce_future_proj = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, expert_dim), nn.SiLU())
            self.tacforce_cross_attention = TacForceCrossAttention(
                expert_dim,
                self.config.tacforce_wm_cross_attention_heads,
                self.config.tacforce_wm_history_steps,
                self.config.tacforce_wm_cross_attention_dropout,
            )
        else:
            self.tacforce_dynamics = None
            self.tacforce_current_proj = None
            self.tacforce_future_proj = None
            self.tacforce_cross_attention = None
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)
        if self.config.force_expert_enabled:
            self.force_expert = copy.deepcopy(self.vlm_with_expert.lm_expert)
        else:
            self.force_expert = None
        if self.config.force_prediction_enabled and self.config.force_prediction_expert_enabled:
            self.force_prediction_expert = copy.deepcopy(self.vlm_with_expert.lm_expert)
        else:
            self.force_prediction_expert = None
        if self.force_prediction_expert is not None:
            prediction_effort_in_dim = self.config.effort_dim
            if self.config.force_prediction_effort_type == "expert_his_c":
                prediction_effort_in_dim = self.config.effort_dim * self.config.effort_history_steps
            self.force_prediction_effort_proj_in = nn.Linear(
                prediction_effort_in_dim, 2 * self.vlm_with_expert.expert_hidden_size
            )
            self.force_prediction_effort_proj_out = nn.Linear(
                2 * self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
            )
        else:
            self.force_prediction_effort_proj_in = None
            self.force_prediction_effort_proj_out = None
        if self.config.force_refine_enabled:
            self.force_refine_out_proj = nn.Linear(
                self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim
            )
        else:
            self.force_refine_out_proj = None
        if self.config.force_prediction_enabled:
            self.force_pred_head = nn.Sequential(
                nn.LayerNorm(self.vlm_with_expert.expert_hidden_size),
                nn.Linear(self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size),
                nn.SiLU(),
                nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.effort_dim),
            )
        else:
            self.force_pred_head = None
        self._last_predicted_force = None
        self.force_vqvae = None
        self.force_code_embedder = None
        if self.effort_tokenizer == "force_vqvae":
            force_vqvae_cfg, force_vqvae_stats = self._load_force_vqvae()
            self.force_vqvae_window = force_vqvae_cfg.window
            self.config.force_vqvae_window = force_vqvae_cfg.window
            if force_vqvae_cfg.force_dim != self.config.effort_dim:
                raise ValueError(
                    f"ForceVQVAE checkpoint expects force_dim={force_vqvae_cfg.force_dim}, "
                    f"but `effort_dim={self.config.effort_dim}`."
                )
            self.force_code_embedder = nn.Embedding(
                force_vqvae_cfg.codebook_size, self.vlm_with_expert.expert_hidden_size
            )
            self.register_buffer(
                "force_vqvae_min", torch.as_tensor(force_vqvae_stats["force_min"], dtype=torch.float32)
            )
            self.register_buffer(
                "force_vqvae_max", torch.as_tensor(force_vqvae_stats["force_max"], dtype=torch.float32)
            )
            self.register_buffer(
                "force_vqvae_mask", torch.as_tensor(force_vqvae_stats["force_mask"], dtype=torch.bool)
            )
        if self.effort_tokenizer == "raw" and self.effort_type in {"llm", "llm_his_c", "llm_his_t"}:
            effort_in_dim = self.config.effort_dim * self.config.effort_history_steps
            if self.effort_type == "llm_his_t":
                effort_in_dim = self.config.effort_dim
            self.effort_proj_in = nn.Linear(
                effort_in_dim, 2 * self.vlm_with_expert.config.text_config.hidden_size
            )
            self.effort_proj_out = nn.Linear(
                2 * self.vlm_with_expert.config.text_config.hidden_size,
                self.vlm_with_expert.config.text_config.hidden_size,
            )
        elif self.effort_tokenizer == "raw" and self.effort_type in {"expert", "expert_his_c", "expert_his_t"}:
            effort_in_dim = self.config.effort_dim * self.config.effort_history_steps
            if self.effort_type == "expert_his_t":
                effort_in_dim = self.config.effort_dim
            self.effort_proj_in = nn.Linear(effort_in_dim, 2 * self.vlm_with_expert.expert_hidden_size)
            self.effort_proj_out = nn.Linear(
                2 * self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
            )
        else:
            self.effort_proj_in = None
            self.effort_proj_out = None

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj
        if self.effort_proj_in is not None:
            for params in self.effort_proj_in.parameters():
                params.requires_grad = self.config.train_effort_proj
            for params in self.effort_proj_out.parameters():
                params.requires_grad = self.config.train_effort_proj
        if self.force_code_embedder is not None:
            for params in self.force_code_embedder.parameters():
                params.requires_grad = self.config.train_force_code_embedder
        if self.force_expert is not None:
            for params in self.force_expert.parameters():
                params.requires_grad = self.config.train_force_expert
        if self.force_prediction_expert is not None:
            for params in self.force_prediction_expert.parameters():
                params.requires_grad = self.config.train_force_prediction_expert
        if self.force_prediction_effort_proj_in is not None:
            for params in self.force_prediction_effort_proj_in.parameters():
                params.requires_grad = self.config.train_force_prediction_expert
            for params in self.force_prediction_effort_proj_out.parameters():
                params.requires_grad = self.config.train_force_prediction_expert
        if self.force_pred_head is not None:
            for params in self.force_pred_head.parameters():
                params.requires_grad = self.config.force_prediction_enabled
        if self.tactile_encoder is not None:
            for params in self.tactile_encoder.parameters():
                params.requires_grad = self.config.force_refine_enabled
            for params in self.tactile_proj.parameters():
                params.requires_grad = self.config.force_refine_enabled
        if self.tacforce_dynamics is not None:
            for params in self.tacforce_dynamics.parameters():
                params.requires_grad = False
            train_bridge = self.config.train_tacforce_cross_attention
            for module in (self.tacforce_current_proj, self.tacforce_future_proj, self.tacforce_cross_attention):
                for params in module.parameters():
                    params.requires_grad = train_bridge

    def _load_force_vqvae(self) -> tuple[ForceVQVAEConfig, dict]:
        if ForceVQVAE is None or ForceVQVAEConfig is None:
            raise ImportError("`effort_tokenizer='force_vqvae'` requires the optional lerobot.force_vqvae package.")
        ckpt_path = Path(self.config.force_vqvae_ckpt).expanduser()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"`force_vqvae_ckpt` does not exist: {ckpt_path}")
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        force_vqvae_cfg = ForceVQVAEConfig.from_dict(blob["config"])
        self.force_vqvae = ForceVQVAE(force_vqvae_cfg)
        self.force_vqvae.load_state_dict(blob["model_state"])
        self.force_vqvae.eval()
        for param in self.force_vqvae.parameters():
            param.requires_grad = False
        return force_vqvae_cfg, blob["stats"]

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None, effort: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        effort_embs, effort_pad_masks, effort_att_masks = self._process_effort_tokens(effort, mode="prefix")
        embs.extend(effort_embs)
        pad_masks.extend(effort_pad_masks)
        att_masks.extend(effort_att_masks)

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def _project_effort(self, effort: torch.Tensor) -> torch.Tensor:
        effort_hidden = self.effort_proj_in(effort)
        effort_hidden = F.silu(effort_hidden)
        return self.effort_proj_out(effort_hidden)

    def _project_force_prediction_effort(self, effort: torch.Tensor) -> torch.Tensor:
        effort_hidden = self.force_prediction_effort_proj_in(effort)
        effort_hidden = F.silu(effort_hidden)
        return self.force_prediction_effort_proj_out(effort_hidden)

    def _normalize_force_for_vqvae(self, effort: torch.Tensor) -> torch.Tensor:
        effort = effort.to(device=self.force_vqvae_min.device, dtype=torch.float32)
        denom = (self.force_vqvae_max - self.force_vqvae_min) + 1e-8
        normed = torch.clamp(2.0 * (effort - self.force_vqvae_min) / denom - 1.0, -1.0, 1.0)
        return torch.where(self.force_vqvae_mask, normed, effort)

    def _process_force_code_tokens(self, effort: torch.Tensor | None, mode: str):
        embs = []
        pad_masks = []
        att_masks = []
        suffix_effort_types = {"expert", "expert_his_c", "expert_his_t"}
        if mode != "suffix" or self.effort_type not in suffix_effort_types:
            return embs, pad_masks, att_masks
        if effort is None:
            raise ValueError("`effort_tokenizer='force_vqvae'` requires an effort tensor.")
        if self.force_vqvae is None or self.force_code_embedder is None:
            raise RuntimeError("Force VQ-VAE tokenizer is not initialized.")

        effort = pad_sequence(effort, self.force_vqvae_window)
        force_normed = self._normalize_force_for_vqvae(effort)
        with torch.no_grad():
            self.force_vqvae.eval()
            force_codes = self.force_vqvae.encode(force_normed).to(device=self.force_code_embedder.weight.device)
        force_token = self.force_code_embedder(force_codes)[:, None, :]
        bsize = effort.shape[0]
        embs.append(force_token)
        pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=force_token.device))
        att_masks.append(True)
        return embs, pad_masks, att_masks

    def _process_effort_tokens(self, effort: torch.Tensor | None, mode: str):
        embs = []
        pad_masks = []
        att_masks = []

        if self.effort_type in {"none", "no", "state"}:
            return embs, pad_masks, att_masks

        prefix_effort_types = {"llm", "llm_his_c", "llm_his_t"}
        suffix_effort_types = {"expert", "expert_his_c", "expert_his_t"}
        if mode == "prefix" and self.effort_type not in prefix_effort_types:
            return embs, pad_masks, att_masks
        if mode == "suffix" and self.effort_type not in suffix_effort_types:
            return embs, pad_masks, att_masks
        if effort is None:
            return embs, pad_masks, att_masks
        if self.effort_tokenizer == "force_vqvae":
            return self._process_force_code_tokens(effort, mode)

        bsize = effort.shape[0]
        device = effort.device
        ar_mask_value = mode == "suffix"
        if self.effort_type in {"llm", "expert"}:
            effort_token = self._project_effort(effort[:, -1, :])[:, None, :]
            embs.append(effort_token)
            pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
            att_masks.append(ar_mask_value)
        elif self.effort_type in {"llm_his_c", "expert_his_c"}:
            effort_token = self._project_effort(effort.reshape(bsize, -1))[:, None, :]
            embs.append(effort_token)
            pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
            att_masks.append(ar_mask_value)
        elif self.effort_type in {"llm_his_t", "expert_his_t"}:
            for i in range(effort.shape[1]):
                effort_token = self._project_effort(effort[:, i, :])[:, None, :]
                embs.append(effort_token)
                pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
                att_masks.append(ar_mask_value)

        return embs, pad_masks, att_masks

    def _process_tactile_tokens(self, tactile_data: list[Tensor] | None):
        embs = []
        pad_masks = []
        att_masks = []
        if not self.config.use_tactile:
            return embs, pad_masks, att_masks
        if tactile_data is None or len(tactile_data) == 0:
            raise ValueError(
                "`use_tactile=True` requires tactile distributed_force tensors for force refinement."
            )
        if self.tactile_encoder is None or self.tactile_proj is None:
            raise RuntimeError("Tactile encoder is not initialized.")

        for tactile in tactile_data:
            tactile_tokens = self.tactile_encoder(tactile)
            tactile_tokens = self.tactile_proj(tactile_tokens)
            bsize, n_tokens = tactile_tokens.shape[:2]
            embs.append(tactile_tokens)
            pad_masks.append(torch.ones(bsize, n_tokens, dtype=torch.bool, device=tactile_tokens.device))
            att_masks.extend([True] * n_tokens)
        return embs, pad_masks, att_masks

    def _process_force_prediction_effort_tokens(self, effort: torch.Tensor | None):
        """Embed raw continuous force for the prediction expert, independent of VQ-VAE force codes."""
        embs = []
        pad_masks = []
        att_masks = []
        if self.force_prediction_effort_proj_in is None or effort is None:
            return embs, pad_masks, att_masks

        bsize = effort.shape[0]
        device = effort.device
        effort_type = self.config.force_prediction_effort_type
        if effort_type == "expert":
            effort_token = self._project_force_prediction_effort(effort[:, -1, :])[:, None, :]
            embs.append(effort_token)
            pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
            att_masks.append(True)
        elif effort_type == "expert_his_c":
            effort = pad_sequence(effort, self.config.effort_history_steps)
            effort_token = self._project_force_prediction_effort(effort.reshape(bsize, -1))[:, None, :]
            embs.append(effort_token)
            pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
            att_masks.append(True)
        elif effort_type == "expert_his_t":
            effort = pad_sequence(effort, self.config.effort_history_steps)
            for i in range(effort.shape[1]):
                effort_token = self._project_force_prediction_effort(effort[:, i, :])[:, None, :]
                embs.append(effort_token)
                pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
                att_masks.append(True)

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep, effort: torch.Tensor = None):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        effort_embs, effort_pad_masks, effort_att_masks = self._process_effort_tokens(effort, mode="suffix")
        embs.extend(effort_embs)
        pad_masks.extend(effort_pad_masks)
        att_masks.extend(effort_att_masks)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    @staticmethod
    def _pack_tacforce_tactile(tactile_data: list[Tensor]) -> Tensor:
        """Convert two [B,T,400,3] Tac3D histories to TacForce [B,T,36,20,6]."""
        if len(tactile_data) != 2:
            raise ValueError(f"TacForce-WM requires left and right tactile tensors, got {len(tactile_data)}.")
        packed_hands = []
        for tactile in tactile_data:
            if tactile.ndim != 4 or tuple(tactile.shape[-2:]) != (400, 3):
                raise ValueError(f"TacForce tactile input must be [B,T,400,3], got {tactile.shape}.")
            bsize, steps = tactile.shape[:2]
            hand = tactile.reshape(bsize * steps, 20, 20, 3).permute(0, 3, 1, 2)
            hand = F.interpolate(hand.float(), size=(35, 20), mode="bilinear", align_corners=False)
            hand = hand.permute(0, 2, 3, 1).reshape(bsize, steps, 35, 20, 3)
            packed_hands.append(hand)
        tactile = torch.cat(packed_hands, dim=-1)
        return torch.cat([tactile, tactile[:, :, -1:]], dim=2)

    def encode_tacforce_refine_tokens(
        self, tactile_data: list[Tensor] | None, force_history: Tensor | None
    ) -> Tensor | None:
        if not self.config.tacforce_wm_enabled:
            if not self.config.use_tactile:
                return None
            legacy_embs, _, _ = self._process_tactile_tokens(tactile_data)
            return torch.cat(legacy_embs, dim=1)
        if tactile_data is None or force_history is None:
            raise ValueError("TacForce-WM refinement requires tactile_data and raw force_history.")
        if force_history.ndim != 3 or force_history.shape[1] != self.config.tacforce_wm_history_steps:
            raise ValueError(f"TacForce force history must be [B,16,6], got {force_history.shape}.")
        tactile = self._pack_tacforce_tactile(tactile_data)
        obs = {
            "tactile": tactile,
            "force_4x": torch.repeat_interleave(force_history.float(), self.config.tacforce_wm_force_upsample, dim=1),
        }
        with torch.no_grad():
            latent = self.tacforce_dynamics(obs, compute_predict=True)
        current = self.tacforce_current_proj(latent["tactile_latent_curr"])
        future = self.tacforce_future_proj(latent["tactile_latent_future"])
        return self.tacforce_cross_attention(current, future)

    def embed_tactile_refine_suffix(self, tactile_tokens: Tensor | None, x_t: Tensor, timestep: Tensor):
        """Embed precomputed TacForce cross-attention tokens plus x_t/time tokens."""
        if not self.config.use_tactile:
            return self.embed_suffix(x_t, timestep, effort=None)
        if tactile_tokens is None:
            raise ValueError("Tactile refinement requires precomputed TacForce cross-attention tokens.")

        embs = []
        pad_masks = []
        att_masks = []

        bsize, n_tactile_tokens = tactile_tokens.shape[:2]
        embs.append(tactile_tokens)
        pad_masks.append(torch.ones(bsize, n_tactile_tokens, dtype=torch.bool, device=tactile_tokens.device))
        att_masks.extend([True] * n_tactile_tokens)

        action_emb = self.action_in_proj(x_t)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        embs.append(action_time_emb)
        action_time_dim = action_time_emb.shape[1]
        pad_masks.append(torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device))
        att_masks += [1] * self.config.chunk_size

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def embed_force_prediction_suffix(self, noisy_actions, timestep, effort: torch.Tensor = None):
        """Embed raw force history plus action/time tokens for the force prediction expert."""
        embs = []
        pad_masks = []
        att_masks = []

        effort_embs, effort_pad_masks, effort_att_masks = self._process_force_prediction_effort_tokens(effort)
        embs.extend(effort_embs)
        pad_masks.extend(effort_pad_masks)
        att_masks.extend(effort_att_masks)

        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        action_time_emb = action_time_emb.detach()

        embs.append(action_time_emb)
        action_time_dim = action_time_emb.shape[1]
        pad_masks.append(torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device))
        att_masks += [1] * self.config.chunk_size

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def _detach_cache(self, cache):
        if cache is None:
            return None
        if isinstance(cache, torch.Tensor):
            return cache.detach()
        if isinstance(cache, dict):
            return {key: self._detach_cache(value) for key, value in cache.items()}
        if isinstance(cache, list):
            return [self._detach_cache(value) for value in cache]
        if isinstance(cache, tuple):
            return tuple(self._detach_cache(value) for value in cache)
        return cache

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        effort=None,
        noise=None,
        time=None,
        future_effort=None,
        tactile_tokens=None,
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state, effort=effort
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time, effort=None)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        if not self.config.force_refine_enabled:
            return losses

        force_refine_out = self.forward_force_refine(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            actions,
            effort=effort,
            tactile_tokens=tactile_tokens,
            noise=noise,
            return_force_pred=True,
        )
        force_refine_losses, force_pred = force_refine_out
        force_prediction_loss = None
        if self.force_pred_head is not None and force_pred is not None and future_effort is not None:
            if not hasattr(self, "_debug_force_pred_printed"):
                with torch.no_grad():
                    debug_future = future_effort.detach()
                    debug_pred = force_pred.detach()
                    if debug_future.ndim == 3:
                        print("future_effort seq mean:", debug_future.mean(dim=(0, 1)).cpu())
                        print("future_effort seq min:", debug_future.amin(dim=(0, 1)).cpu())
                        print("future_effort seq max:", debug_future.amax(dim=(0, 1)).cpu())
                        debug_future_pooled = debug_future.mean(dim=1)
                    else:
                        debug_future_pooled = debug_future

                    print("future_effort pooled mean:", debug_future_pooled.mean(dim=0).cpu())
                    print("future_effort pooled min:", debug_future_pooled.amin(dim=0).cpu())
                    print("future_effort pooled max:", debug_future_pooled.amax(dim=0).cpu())
                    print("force_pred mean:", debug_pred.mean(dim=0).cpu())
                    print("force_pred min:", debug_pred.amin(dim=0).cpu())
                    print("force_pred max:", debug_pred.amax(dim=0).cpu())
                self._debug_force_pred_printed = True

            if future_effort.ndim == 3:
                future_effort = future_effort.mean(dim=1)
            future_effort = future_effort.to(device=force_pred.device, dtype=force_pred.dtype)
            force_prediction_loss = F.smooth_l1_loss(force_pred, future_effort, reduction="mean")
        return losses, force_refine_losses, force_prediction_loss

    def forward_force_refine(
        self,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        actions,
        effort=None,
        tactile_tokens=None,
        noise=None,
        return_force_pred: bool = False,
    ) -> Tensor:
        """Train the force-refine head on the lower flow segment used by fast force updates."""
        if self.force_refine_out_proj is None:
            raise RuntimeError("`forward_force_refine` requires `force_refine_enabled=True`.")
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        tau_split = 1.0 - self.config.force_refine_split_step / self.config.num_steps
        time = self.sample_time(actions.shape[0], actions.device) * tau_split
        time = torch.clamp(time, min=0.001)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        if self.config.force_shared_attention_enabled:
            action_context_cache, action_context_pad_masks = self.build_action_context_cache(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                x_t,
                time,
            )
            force_suffix_out = self.forward_force_from_action_cache(
                action_context_pad_masks,
                action_context_cache,
                x_t,
                time,
                tactile_tokens=tactile_tokens,
            )
            refine_hidden = force_suffix_out[:, -self.config.chunk_size :]
            force_v_t = self.force_refine_out_proj(refine_hidden)
            force_refine_losses = F.mse_loss(u_t, force_v_t, reduction="none")
            if self.force_prediction_expert is not None:
                force_pred = self.predict_force_from_action_cache(
                    action_context_pad_masks,
                    action_context_cache,
                    x_t,
                    time,
                    effort=effort,
                )
            else:
                force_pred = self.predict_force_from_suffix(force_suffix_out, detach_hidden=True)
            if return_force_pred:
                return force_refine_losses, force_pred
            return force_refine_losses

        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_tactile_refine_suffix(
            tactile_tokens, x_t, time
        )
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, force_suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
            expert_model=self.force_expert,
        )
        force_suffix_out = force_suffix_out.to(dtype=torch.float32)
        refine_hidden = force_suffix_out[:, -self.config.chunk_size :]
        force_v_t = self.force_refine_out_proj(refine_hidden)
        force_refine_losses = F.mse_loss(u_t, force_v_t, reduction="none")
        if self.force_prediction_expert is not None:
            force_pred = self.predict_force_direct(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                x_t,
                time,
                effort=effort,
            )
        else:
            force_pred = self.predict_force_from_suffix(force_suffix_out, detach_hidden=True)
        if return_force_pred:
            return force_refine_losses, force_pred
        return force_refine_losses

    def pool_force_hidden(self, force_suffix_out: Tensor) -> Tensor:
        n_force_tokens = force_suffix_out.shape[1] - self.config.chunk_size
        if n_force_tokens > 0:
            return force_suffix_out[:, :n_force_tokens].mean(dim=1)
        return force_suffix_out[:, -self.config.chunk_size :].mean(dim=1)

    def predict_force_from_suffix(self, force_suffix_out: Tensor, detach_hidden: bool = False) -> Tensor | None:
        if self.force_pred_head is None:
            return None
        force_hidden = self.pool_force_hidden(force_suffix_out)
        if detach_hidden:
            force_hidden = force_hidden.detach()
        force_pred = self.force_pred_head(force_hidden)
        if self.config.force_prediction_use_tanh:
            force_pred = self.config.force_prediction_scale * torch.tanh(force_pred)
        self._last_predicted_force = force_pred.detach()
        return force_pred

    def predict_force_from_action_cache(
        self,
        action_context_pad_masks,
        action_context_cache,
        x_t,
        timestep,
        effort=None,
    ) -> Tensor | None:
        if self.force_pred_head is None:
            return None
        force_suffix_out = self.forward_force_prediction_from_action_cache(
            action_context_pad_masks,
            self._detach_cache(action_context_cache),
            x_t.detach(),
            timestep.detach() if isinstance(timestep, torch.Tensor) else timestep,
            effort=effort.detach() if isinstance(effort, torch.Tensor) else effort,
        )
        return self.predict_force_from_suffix(force_suffix_out)

    def predict_force_direct(
        self,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        x_t,
        timestep,
        effort=None,
    ) -> Tensor | None:
        if self.force_pred_head is None:
            return None
        if self.force_prediction_expert is None:
            raise RuntimeError("`predict_force_direct` requires `force_prediction_expert_enabled=True`.")
        force_suffix_out = self.forward_force_prediction_direct(
            prefix_embs.detach(),
            prefix_pad_masks,
            prefix_att_masks,
            x_t.detach(),
            timestep.detach() if isinstance(timestep, torch.Tensor) else timestep,
            effort=effort.detach() if isinstance(effort, torch.Tensor) else effort,
        )
        return self.predict_force_from_suffix(force_suffix_out)

    def build_action_context_cache(
        self,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        x_t,
        timestep,
    ) -> tuple[dict, Tensor]:
        """Cache slow-stage action context in SmolVLA's native cache format.

        Self-attention layers are refreshed to [prefix | action]. Cross-attention
        layers keep prefix KV, matching the original SmolVLA expert design.
        """
        action_suffix_embs, action_suffix_pad_masks, action_suffix_att_masks = self.embed_suffix(
            x_t, timestep, effort=None
        )
        action_context_pad_masks = torch.cat([prefix_pad_masks, action_suffix_pad_masks], dim=1)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, action_context_cache = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )

        action_len = action_suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, action_len, prefix_len)
        action_att_2d_masks = make_att_2d_masks(action_suffix_pad_masks, action_suffix_att_masks)
        action_step_att_2d_masks = torch.cat([prefix_pad_2d_masks, action_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        action_position_ids = prefix_offsets + torch.cumsum(action_suffix_pad_masks, dim=1) - 1

        _, action_context_cache = self.vlm_with_expert.forward(
            attention_mask=action_step_att_2d_masks,
            position_ids=action_position_ids,
            past_key_values=action_context_cache,
            inputs_embeds=[None, action_suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
            update_kv_cache=True,
        )
        return action_context_cache, action_context_pad_masks

    def forward_force_from_action_cache(
        self,
        action_context_pad_masks,
        action_context_cache,
        x_t,
        timestep,
        tactile_tokens=None,
    ) -> Tensor:
        """Run force expert on fresh tactile tokens while attending cached [latent | action] KV."""
        if self.force_expert is None:
            raise RuntimeError("Force-cache refinement requires `force_expert_enabled=True`.")

        force_suffix_embs, force_suffix_pad_masks, force_suffix_att_masks = self.embed_tactile_refine_suffix(
            tactile_tokens, x_t, timestep
        )

        force_len = force_suffix_pad_masks.shape[1]
        batch_size = action_context_pad_masks.shape[0]
        context_len = action_context_pad_masks.shape[1]
        context_pad_2d_masks = action_context_pad_masks[:, None, :].expand(
            batch_size, force_len, context_len
        )
        force_att_2d_masks = make_att_2d_masks(force_suffix_pad_masks, force_suffix_att_masks)
        att_2d_masks = torch.cat([context_pad_2d_masks, force_att_2d_masks], dim=2)

        context_offsets = torch.sum(action_context_pad_masks, dim=-1)[:, None]
        position_ids = context_offsets + torch.cumsum(force_suffix_pad_masks, dim=1) - 1

        (_, force_suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=action_context_cache,
            inputs_embeds=[None, force_suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
            expert_model=self.force_expert,
        )
        return force_suffix_out.to(dtype=torch.float32)

    def forward_force_prediction_from_action_cache(
        self,
        action_context_pad_masks,
        action_context_cache,
        x_t,
        timestep,
        effort=None,
    ) -> Tensor:
        """Predict future force with an independent expert over cached [latent | action] context."""
        if self.force_prediction_expert is None:
            raise RuntimeError(
                "`forward_force_prediction_from_action_cache` requires "
                "`force_prediction_expert_enabled=True`."
            )

        force_suffix_embs, force_suffix_pad_masks, force_suffix_att_masks = self.embed_force_prediction_suffix(
            x_t, timestep, effort=effort
        )

        force_len = force_suffix_pad_masks.shape[1]
        batch_size = action_context_pad_masks.shape[0]
        context_len = action_context_pad_masks.shape[1]
        context_pad_2d_masks = action_context_pad_masks[:, None, :].expand(
            batch_size, force_len, context_len
        )
        force_att_2d_masks = make_att_2d_masks(force_suffix_pad_masks, force_suffix_att_masks)
        att_2d_masks = torch.cat([context_pad_2d_masks, force_att_2d_masks], dim=2)

        context_offsets = torch.sum(action_context_pad_masks, dim=-1)[:, None]
        position_ids = context_offsets + torch.cumsum(force_suffix_pad_masks, dim=1) - 1

        (_, force_suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=action_context_cache,
            inputs_embeds=[None, force_suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
            expert_model=self.force_prediction_expert,
        )
        return force_suffix_out.to(dtype=torch.float32)

    def forward_force_prediction_direct(
        self,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        x_t,
        timestep,
        effort=None,
    ) -> Tensor:
        """Predict future force with an independent expert without cached action context."""
        if self.force_prediction_expert is None:
            raise RuntimeError(
                "`forward_force_prediction_direct` requires `force_prediction_expert_enabled=True`."
            )

        force_suffix_embs, force_suffix_pad_masks, force_suffix_att_masks = self.embed_force_prediction_suffix(
            x_t, timestep, effort=effort
        )
        pad_masks = torch.cat([prefix_pad_masks, force_suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, force_suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, force_suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, force_suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
            expert_model=self.force_prediction_expert,
        )
        return force_suffix_out.to(dtype=torch.float32)

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, effort=None, noise=None) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state, effort=effort
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
                effort=None,
            )
            # Euler step
            x_t += dt * v_t
            time += dt
        return x_t

    def sample_actions_for_force_refine(
        self, images, img_masks, lang_tokens, lang_masks, state, effort=None, tactile_tokens=None, noise=None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Run the slow split stage and immediately refine once with the current force.

        This mirrors T-Rex's slow_and_fast path: the slow stage caches the prefix KV and a partially
        denoised chunk; the fast stage can be repeated later with fresh force readings.
        """
        bsize = state.shape[0]
        device = state.device
        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state, effort=effort
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        for _ in range(self.config.force_refine_split_step):
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(prefix_pad_masks, past_key_values, x_t, expanded_time, effort=None)
            x_t += dt * v_t
            time += dt

        if self.config.force_shared_attention_enabled:
            past_key_values, cache_pad_masks = self.build_action_context_cache(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                x_t,
                time.expand(bsize),
            )
        else:
            cache_pad_masks = prefix_pad_masks

        refine_state = {
            "prefix_pad_masks": cache_pad_masks,
            "past_key_values": past_key_values,
            "x_split": x_t.detach(),
            "tau_split": time.detach(),
        }
        if self.force_prediction_expert is not None and not self.config.force_shared_attention_enabled:
            refine_state["prefix_embs"] = prefix_embs.detach()
            refine_state["prefix_att_masks"] = prefix_att_masks
        actions = self.refine_actions_from_force(refine_state, effort=effort, tactile_tokens=tactile_tokens)
        return actions, refine_state

    def refine_actions_from_force(
        self,
        refine_state: dict[str, Tensor],
        effort=None,
        tactile_tokens=None,
    ) -> Tensor:
        """Continue the lower flow segment from cached x_split using a fresh tactile condition."""
        prefix_pad_masks = refine_state["prefix_pad_masks"]
        past_key_values = refine_state["past_key_values"]
        x_t = refine_state["x_split"].clone()
        time = refine_state["tau_split"].clone()

        bsize = x_t.shape[0]
        device = x_t.device
        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=torch.float32, device=device)
        remaining_steps = self.config.num_steps - self.config.force_refine_split_step
        if self.force_prediction_expert is not None:
            if self.config.force_shared_attention_enabled:
                self.predict_force_from_action_cache(
                    prefix_pad_masks,
                    past_key_values,
                    x_t,
                    time.expand(bsize),
                    effort=effort,
                )
            else:
                self.predict_force_direct(
                    refine_state["prefix_embs"],
                    prefix_pad_masks,
                    refine_state["prefix_att_masks"],
                    x_t,
                    time.expand(bsize),
                    effort=effort,
                )
        for _ in range(remaining_steps):
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
                effort,
                tactile_tokens=tactile_tokens,
                force_refine=True,
            )
            x_t += dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        effort=None,
        tactile_tokens=None,
        force_refine: bool = False,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        if force_refine and self.force_refine_out_proj is None:
            raise RuntimeError("Force refinement requires `force_refine_enabled=True`.")
        if force_refine and self.config.force_shared_attention_enabled:
            force_suffix_out = self.forward_force_from_action_cache(
                prefix_pad_masks,
                past_key_values,
                x_t,
                timestep,
                tactile_tokens=tactile_tokens,
            )
            if self.force_prediction_expert is None:
                self.predict_force_from_suffix(force_suffix_out)
            suffix_out = force_suffix_out[:, -self.config.chunk_size :]
            return self.force_refine_out_proj(suffix_out)

        if force_refine:
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_tactile_refine_suffix(
                tactile_tokens, x_t, timestep
            )
        else:
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep, effort=effort)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            expert_model=self.force_expert if force_refine else None,
        )
        suffix_out = outputs_embeds[1].to(dtype=torch.float32)
        if force_refine and self.force_prediction_expert is None:
            self.predict_force_from_suffix(suffix_out)
        suffix_out = suffix_out[:, -self.config.chunk_size :]

        out_proj = self.force_refine_out_proj if force_refine else self.action_out_proj
        v_t = out_proj(suffix_out)
        return v_t
