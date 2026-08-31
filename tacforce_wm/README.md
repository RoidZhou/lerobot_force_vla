# TacForce world-model training on the bolt/nut LeRobot dataset

This directory preserves the `dynamics_main` model, loss, optimizer, scheduler,
checkpointing, single-GPU and DDP training code from `/home/zhou/vla/TacForeSight`.
Only the input pipeline is adapted to the local LeRobot v2.1 parquet dataset:

`/home/zhou/autolab/GUFIC_mujoco-main/ur5_rg2_real_smolvla_dataset_tactile_boltnut`

## Data mapping

- tactile target: left/right `displacement`, each `[400, 3]`
- force condition: `observation.force_torque`, `[6]`
- episodes: deterministic 80/20 train/validation split (`split_seed: 42`)
- windows never cross episode boundaries
- each `20x20` tactile grid is row-interpolated to `35x20`; the last row is
  repeated to retain the original tokenizer's exact `36x20` input contract
- synchronous force samples are repeated four times to retain the original
  condition encoder's exact 4x temporal contract
- `preload_to_ram: true` reads all selected episode tactile/force/state columns
  once at dataset construction; tactile spatial conversion is also performed
  once, so shuffled training windows use RAM slicing without Parquet re-decoding

All mappings and hyperparameters are in `config/dynamics_train.yaml`. To use the
distributed tactile force field instead, change both tactile keys from
`displacement` to `distributed_force`.

## Train

From this directory:

```bash
cd /home/zhou/anaconda3/envs/lerobot/lib/python3.10/site-packages/lerobot/tacforce_wm
/home/zhou/anaconda3/envs/lerobot/bin/python train.py dynamics --config config/dynamics_train.yaml
```

Checkpoints and TensorBoard logs are written to
`outputs/tactile_boltnut_dynamics/`.

For multi-GPU training:

```bash
torchrun --standalone --nproc_per_node=2 train.py dynamics \
  --config config/dynamics_train.yaml --ddp
```

Evaluate the best checkpoint:

```bash
/home/zhou/anaconda3/envs/lerobot/bin/python train.py dynamics \
  --config config/dynamics_train.yaml --test
```

## Train the wrist-camera policy

`config/config.yaml` is configured for the same LeRobot dataset and uses only
`observation.wrist_image`. It predicts the dataset's 7D absolute action chunks
from wrist RGB, force/state history, current tactile latent and world-model
predicted tactile latent.

First finish dynamics training so its `best_model.pt` exists, then run:

```bash
/home/zhou/anaconda3/envs/lerobot/bin/python train.py policy \
  --config config/config.yaml
```

Policy outputs are written to `outputs/wrist_policy_run/`.

## Precompute the Policy cache

Policy training can cache all frozen computation: the local DINOv2 wrist-image
backbone feature and the frozen TacForce model's current/future tactile latents.
The cache is tied to the exact DINO and Dynamics checkpoint SHA256 values; if
either checkpoint changes, training asks you to rebuild it instead of silently
using stale features.

Use `distributed_force` for both Dynamics and Policy, then set the cloud paths:

```yaml
data:
  root_dir: /root/autodl-tmp/Tactile-VLA/ur5_rg2_real_smolvla_dataset_tactile_boltnut
  latent_cache_root_dir: /root/autodl-tmp/Tactile-VLA/tacforce_policy_cache
  preload_to_ram: true
  tactile_left_key: observation.tactile_left.distributed_force
  tactile_right_key: observation.tactile_right.distributed_force

model:
  backend:
    config_path: /absolute/path/to/dynamics/resolved_config.yaml
    ckpt_path: /absolute/path/to/dynamics/checkpoints/best_model.pt
  policy:
    freeze_image_encoder: true
    dino_checkpoint_path: /absolute/path/to/DINOv2/model.safetensors
```

Generate both splits before starting Policy training. On a 96 GB RTX PRO 6000,
start with batch 128 and increase to 256 if GPU memory permits:

```bash
cd /absolute/path/to/tacforce_wm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python tools/precompute_policy_latents.py \
  --config config/config.yaml \
  --split all \
  --batch-size 128 \
  --chunk-size 64 \
  --amp
```

This creates:

```text
/root/autodl-tmp/Tactile-VLA/tacforce_policy_cache/
├── train/policy_latent_cache.zarr/
└── val/policy_latent_cache.zarr/
```

Then train normally:

```bash
python train.py policy --config config/config.yaml
```

With `preload_to_ram: true`, numeric observations/actions and all three cache
arrays are loaded once. Policy batches then contain only `force`, `state`,
`action`, `image_backbone_feat`, `tactile_latent_curr`, and
`tactile_latent_future`; raw PNG and tactile columns are not read during
training. Rebuild an existing cache explicitly with `--overwrite`.
