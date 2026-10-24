# CS5090 Project 1

Trains small GPT-style models (nanoGPT-based) on Python code from `codeparrot/codeparrot-clean`, using a custom BPE tokenizer.
Three models are trained: `base_model`, `ablation_model` (tokenizer with no learned merges, byte-level only) and `mega_model` (larger).
All commands are run from the repo root.

## Setup

```bash
pip install -r requirements.txt
```

Training needs a CUDA GPU (or a TPU via `torch_xla`).

## Order to run

### 1. Build the tokenizer corpus

```bash
python data.py
```

Streams `codeparrot-clean`, keeps files with apache-2.0, mit or bsd-3-clause licenses, and writes 10,000 of them to `data/tokenizer_text.txt`.

### 2. Train the tokenizers

```bash
python tokenizer.py
python ablation_tokenizing.py
```

- `tokenizer.py` trains an 8192-token BPE tokenizer on the corpus and saves it to `output/base_model/tokenizer/` and `output/mega_model/tokenizer/`.
- `ablation_tokenizing.py` builds a byte-level-only tokenizer (no merges) and saves it to `output/ablation_model/tokenizer/`.

### 3. Train the models

```bash
python train.py --config configs/training_configs/base_config.py
python train.py --config configs/training_configs/ablation_config.py
python train.py --config configs/training_configs/mega_config.py
```

Each config sets the model size, hyperparameters, tokenizer path, and output paths. Data is streamed from `codeparrot-clean` and tokenized on the fly.
Checkpoints go to `<out_dir>/` (`ckpt.pt`, `latest.pt`) and loss logs to `output/<model>/logs/`.
Training is resumable: rerun the same command, or pass `--resume_from <path to latest.pt>`.

### 4. Plot the loss curves

```bash
python visualize.py --input output/base_model/logs/base_logs.csv --output output/base_model/viz/base_viz.jpeg --title "Base Model"
```

Repeat with the ablation and mega logs (`ablation_logs.csv`, `mega_logs.csv`).

### 5. Generate outputs for evaluation

```bash
python gen_pipe.py --model_dir output/base_model --gen_config configs/generation_configs/phase2_test.csv --output_path output/base_model/generations/phase2_generations.csv
python gen_pipe.py --model_dir output/base_model --gen_config configs/generation_configs/phase3_test.csv --output_path output/base_model/generations/phase3_generations.csv
```

Run for each of `base_model`, `ablation_model` and `mega_model`. The generation configs are CSVs of prompts (`phase2_test.csv`, `phase3_test.csv`).
`--model_dir` must contain `python-coder/ckpt.pt` and `tokenizer/bpe_tokenizer.json`.

### 6. (Optional) Sample interactively

```bash
python sample.py --out_dir=output/base_model/python-coder
```

Generates text from a checkpoint. Overrides use the `--key=value` form (e.g. `--device=cpu --dtype=float32` if no GPU is available).

## Layout

- `model.py` — GPT model definition.
- `train.py` — training loop.
- `configs/training_configs/` — per-model training configs.
- `configs/generation_configs/` — evaluation prompt CSVs.
- `output/<model>/` — tokenizer, checkpoints, logs, plots and generations for each model.
- `ai_log.txt` — log of AI assistance.
