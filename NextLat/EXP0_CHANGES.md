# Changes to upstream JaydenTeoh/NextLat for Experiment 0

This directory is a clone of [JaydenTeoh/NextLat](https://github.com/JaydenTeoh/NextLat)
(MIT, NeurIPS 2025). The upstream tree is **not** committed to this repo (it is gitignored;
re-clone it to run). Only the files we added or modified for the NextLat-for-safety
Experiment 0 are tracked here, listed below.

## Added
- `data/gridworld.py` — gridworld environment generator, tokenizer, `GridworldDataModule`,
  and `leakage_audit`. The generator + audit import with only numpy+torch (no lightning).
- `config/gridworld/gpt_gridworld.yaml`, `config/gridworld/nextlat_gridworld.yaml` —
  matched training configs (8-layer/256-dim; NextLat uses the paper's Manhattan loss weights).

## Modified (small, backward-compatible)
- `train.py` — registered `"gridworld": GridworldDataModule` in the `DATAMODULES` dict
  (import + one dict entry).
- `models/model_gpt.py`, `models/model_nextlat.py` — added a `return_all_layers: bool = False`
  argument to `Transformer.forward` / `NextLatTransformer.forward` that returns each
  post-block hidden state plus the final-norm state (for per-layer probing). Off by default,
  so all existing behaviour is unchanged.

## Not committed
- The rest of upstream NextLat (gitignored).
- `output/`, `wandb/` (training artifacts; on Modal Volume `nextlat-exp0`).

See `../docs/experiments/exp0.md` and `../experiments/` for the experiment itself.
