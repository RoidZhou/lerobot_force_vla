# Force VQ-VAE

Discrete tokenizer for high-rate end-effector force/torque windows.

It mirrors the T-Rex `tactile_vqvae` idea, but targets a single Tac3D-style
`force_torque` stream:

```text
force_torque history [window, 6]
-> temporal encoder
-> EMA vector quantizer/codebook
-> discrete force code id
-> decoder reconstructs normalized force_torque
```

Train:

```bash
python -m lerobot.force_vqvae.train \
  --repo_id ur5_rg2_real_smolvla_dataset_force_boltnut_speed_627 \
  --data_root "/media/zhou/Elements SE/YBZHOU/ur5_rg2_real_smolvla_dataset_force_boltnut_speedup_total" \
  --force_key force_torque \
  --force_dim 6 \
  --output_dir /tmp/force_vqvae \
  --window 16 \
  --stride 4 \
  --codebook_size 256 \
  --epochs 30 \
  --batch_size 256
```

Smoke test:

```bash
python -m lerobot.force_vqvae.train \
  --repo_id ur5_rg2_real_smolvla_dataset_force_boltnut_speed_627 \
  --data_root "/path/to/lerobot_dataset" \
  --force_key force_torque \
  --output_dir /tmp/force_vqvae \
  --smoke_test 1 \
  --batch_size 8 \
  --num_workers 0
```

Checkpoint format:

```text
latest.pt
  config       ForceVQVAEConfig as dict
  stats        ForceStats min/max/mask
  model_state  ForceVQVAE weights and EMA codebook
  optim_state
  step
  epoch
```

Encode codes after pretraining:

```bash
python -m lerobot.force_vqvae.encode_codes \
  --repo_id ur5_rg2_real_smolvla_dataset_force_boltnut_speed_627 \
  --data_root "/path/to/lerobot_dataset" \
  --force_key force_torque \
  --vqvae_ckpt /tmp/force_vqvae/<run_name>/latest.pt \
  --output /tmp/force_codes.pt \
  --stride 1
```

The output contains `codes`, `episode_index`, and `frame_index`.
