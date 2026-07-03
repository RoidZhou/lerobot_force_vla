#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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
π0: A Vision-Language-Action Flow Model for General Robot Control

[Paper](https://www.physicalintelligence.company/download/pi0.pdf)
[Jax code](https://github.com/Physical-Intelligence/openpi)

Designed by Physical Intelligence. Ported from Jax by Hugging Face.

Install pi0 extra dependencies:
```bash
pip install -e ".[pi0]"
```

Example of finetuning the pi0 pretrained model (`pi0_base` in `openpi`):
```bash
python lerobot/scripts/train.py \
--policy.path=lerobot/pi0 \
--dataset.repo_id=danaaubakirova/koch_test
```

Example of finetuning the pi0 neural network with PaliGemma and expert Gemma
pretrained with VLM default parameters before pi0 finetuning:
```bash
python lerobot/scripts/train.py \
--policy.type=pi0 \
--dataset.repo_id=danaaubakirova/koch_test
```

Example of using the pi0 pretrained model outside LeRobot training framework:
```python
policy = Pi0Policy.from_pretrained("lerobot/pi0")
```

"""

import math
import copy
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoTokenizer

from lerobot.common.constants import ACTION, OBS_EFFORT, OBS_STATE
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0.paligemma_with_expert import (
    PaliGemmaWithExpertConfig,
    PaliGemmaWithExpertModel,
)
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import populate_queues
from lerobot.common.utils.utils import get_safe_dtype
from lerobot.force_vqvae.models import ForceVQVAE, ForceVQVAEConfig


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
    # reverses this transformation to be consistent with pi0 which is pretrained in
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
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
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


class PI0Policy(PreTrainedPolicy):
    """Wrapper class around PI0FlowMatching model to train and run inference within LeRobot."""

    config_class = PI0Config
    name = "pi0"

    def __init__(
        self,
        config: PI0Config,
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

        self.language_tokenizer = AutoTokenizer.from_pretrained(
            getattr(self.config, "vlm_model_name", "google/paligemma-3b-pt-224")
        )
        self.model = PI0FlowMatching(config)

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque([], maxlen=self.config.n_action_steps),
        }
        self._action_queue = self._queues[ACTION]
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

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

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
            lang_tokens, lang_masks = self.prepare_language(batch)

            if self.config.force_refine_enabled:
                actions, self._force_refine_state = self.model.sample_actions_for_force_refine(
                    images, img_masks, lang_tokens, lang_masks, state, effort=effort, noise=noise
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
        refined_actions = self.model.refine_actions_from_force(self._force_refine_state, effort=effort)

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

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict[str, Tensor]]:
        """Do a full training forward pass to compute the loss"""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        raw_effort = self._raw_effort_for_tokenizer(batch)
        batch = self.normalize_inputs(batch)
        self._restore_raw_effort_for_tokenizer(batch, raw_effort)
        batch = self.normalize_targets(batch)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        effort = self.prepare_effort(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        loss_dict = {}
        model_losses = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions, effort, noise, time
        )
        if isinstance(model_losses, tuple):
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
        loss_dict["l2_loss"] = action_loss.item()
        loss_dict["loss"] = loss.item()

        return loss, loss_dict

    def prepare_images(self, batch):
        """Apply Pi0 preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
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
            img = batch[key]

            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
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

        # PaliGemma prompt has to end with a new line
        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding="max_length",
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
        return pad_sequence(effort, history_steps)

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions


class PI0FlowMatching(nn.Module):
    """
    π0: A Vision-Language-Action Flow Model for General Robot Control

    [Paper](https://www.physicalintelligence.company/download/pi0.pdf)
    [Jax code](https://github.com/Physical-Intelligence/openpi)

    Designed by Physical Intelligence. Ported from Jax by Hugging Face.
    ┌──────────────────────────────┐
    │               actions        │
    │               ▲              │
    │              ┌┴─────┐        │
    │  kv cache    │Gemma │        │
    │  ┌──────────►│Expert│        │
    │  │           │      │        │
    │ ┌┴────────┐  │x 10  │        │
    │ │         │  └▲──▲──┘        │
    │ │PaliGemma│   │  │           │
    │ │         │   │  robot state │
    │ │         │   noise          │
    │ └▲──▲─────┘                  │
    │  │  │                        │
    │  │  image(s)                 │
    │  language tokens             │
    └──────────────────────────────┘
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.effort_type = self.config.effort_type
        self.effort_tokenizer = self.config.effort_tokenizer

        paligemma_with_export_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_implementation=self.config.attention_implementation,
        )
        self.paligemma_with_expert = PaliGemmaWithExpertModel(paligemma_with_export_config)

        # Projections are float32
        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)
        if self.config.force_expert_enabled:
            self.force_expert = copy.deepcopy(self.paligemma_with_expert.gemma_expert)
        else:
            self.force_expert = None
        if self.config.force_refine_enabled:
            self.force_refine_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)
        else:
            self.force_refine_out_proj = None
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
            self.force_code_embedder = nn.Embedding(force_vqvae_cfg.codebook_size, self.config.proj_width)
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
            prefix_hidden_size = self.paligemma_with_expert.paligemma.config.text_config.hidden_size
            self.effort_proj_in = nn.Linear(effort_in_dim, 2 * prefix_hidden_size)
            self.effort_proj_out = nn.Linear(2 * prefix_hidden_size, prefix_hidden_size)
        elif self.effort_tokenizer == "raw" and self.effort_type in {"expert", "expert_his_c", "expert_his_t"}:
            effort_in_dim = self.config.effort_dim * self.config.effort_history_steps
            if self.effort_type == "expert_his_t":
                effort_in_dim = self.config.effort_dim
            self.effort_proj_in = nn.Linear(effort_in_dim, 2 * self.config.proj_width)
            self.effort_proj_out = nn.Linear(2 * self.config.proj_width, self.config.proj_width)
        else:
            self.effort_proj_in = None
            self.effort_proj_out = None

        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)

        self.set_requires_grad()

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

    def _load_force_vqvae(self) -> tuple[ForceVQVAEConfig, dict]:
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
        self, images, img_masks, lang_tokens, lang_masks, effort: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        # TODO: avoid list in python and torch.cat ; prefer pre-allocation with torch.empty
        embs = []
        pad_masks = []
        att_masks = []

        # TODO: remove for loop
        for (
            img,
            img_mask,
        ) in zip(images, img_masks, strict=False):
            img_emb = self.paligemma_with_expert.embed_image(img)
            img_emb = img_emb.to(dtype=torch.bfloat16)

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)

        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        effort_embs, effort_pad_masks, effort_att_masks = self._process_effort_tokens(effort, mode="prefix")
        embs.extend(effort_embs)
        pad_masks.extend(effort_pad_masks)
        att_masks.extend(effort_att_masks)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def _project_effort(self, effort: torch.Tensor) -> torch.Tensor:
        effort_hidden = self.effort_proj_in(effort)
        effort_hidden = F.silu(effort_hidden)
        return self.effort_proj_out(effort_hidden)

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
            return embs, pad_masks, att_masks
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

    def embed_suffix(self, state, noisy_actions, timestep, effort: torch.Tensor = None):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed state
        state_emb = self.state_proj(state)
        state_emb = state_emb.to(dtype=torch.bfloat16)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        dtype = state_emb.dtype
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1]

        effort_embs, effort_pad_masks, effort_att_masks = self._process_effort_tokens(effort, mode="suffix")
        embs.extend(effort_embs)
        pad_masks.extend(effort_pad_masks)
        att_masks.extend(effort_att_masks)

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=dtype)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)

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
        att_masks += [1] + ([0] * (self.config.n_action_steps - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, effort=None, noise=None, time=None
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
            images, img_masks, lang_tokens, lang_masks, effort=effort
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, time, effort=None)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        losses = F.mse_loss(u_t, v_t, reduction="none")
        if not self.config.force_refine_enabled:
            return losses

        force_refine_losses = self.forward_force_refine(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            state,
            actions,
            effort=effort,
            noise=noise,
        )
        return losses, force_refine_losses

    def forward_force_refine(
        self,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        state,
        actions,
        effort=None,
        noise=None,
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

        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, time, effort=effort)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
            expert_model=self.force_expert,
        )
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        force_v_t = self.force_refine_out_proj(suffix_out)
        return F.mse_loss(u_t, force_v_t, reduction="none")

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, effort=None, noise=None) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, effort=effort
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        _, past_key_values = self.paligemma_with_expert.forward(
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
                state,
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
        self, images, img_masks, lang_tokens, lang_masks, state, effort=None, noise=None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Run the slow split stage and immediately refine once with the current force."""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, effort=effort
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_key_values = self.paligemma_with_expert.forward(
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
            v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time, effort=None)
            x_t += dt * v_t
            time += dt

        refine_state = {
            "state": state,
            "prefix_pad_masks": prefix_pad_masks,
            "past_key_values": past_key_values,
            "x_split": x_t.detach(),
            "tau_split": time.detach(),
        }
        actions = self.refine_actions_from_force(refine_state, effort=effort)
        return actions, refine_state

    def refine_actions_from_force(self, refine_state: dict[str, Tensor], effort=None) -> Tensor:
        """Continue the lower flow segment from cached x_split using a fresh force condition."""
        state = refine_state["state"]
        prefix_pad_masks = refine_state["prefix_pad_masks"]
        past_key_values = refine_state["past_key_values"]
        x_t = refine_state["x_split"].clone()
        time = refine_state["tau_split"].clone()

        bsize = x_t.shape[0]
        device = x_t.device
        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=torch.float32, device=device)
        remaining_steps = self.config.num_steps - self.config.force_refine_split_step
        for _ in range(remaining_steps):
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
                effort,
                force_refine=True,
            )
            x_t += dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        effort=None,
        force_refine: bool = False,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        if force_refine and self.force_refine_out_proj is None:
            raise RuntimeError("Force refinement requires `force_refine_enabled=True`.")
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, x_t, timestep, effort=effort
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            expert_model=self.force_expert if force_refine else None,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        out_proj = self.force_refine_out_proj if force_refine else self.action_out_proj
        v_t = out_proj(suffix_out)
        return v_t
