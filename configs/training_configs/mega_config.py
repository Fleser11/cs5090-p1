from dataclasses import dataclass

@dataclass
class TrainingConfig:
    out_dir: str = 'output/mega_model/python-coder/'
    dataset_name: str = "codeparrot/codeparrot-clean"
    tokenizer_path: str = "output/mega_model/tokenizer/bpe_tokenizer.json"
    log_dir: str = "output/mega_model/logs/mega_logs.csv"
    eval_iters: int = 200
    eval_interval: int = 200
    log_interval: int = 10
    gradient_accumulation_steps: int = 8  # batch_size * this * block_size = 65,536 tokens per iteration
    batch_size: int = 8  # smaller micro-batch so it fits a 16GB Kaggle GPU
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    learning_rate: float = 5e-4
    min_lr: float = 5e-5  # learning_rate / 10
    max_iters: int = 30000
    lr_decay_iters: int = 30000  # should match max_iters
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_iters: int = 500
    vocab_size: int = 17824
    rope: bool = False  # True: rotary position encodings instead of learned position embeddings
    resume_from: str = ""  # latest.pt from an earlier session (or pass --resume_from); out_dir/latest.pt is tried first
    save_interval: int = 250  # iterations between resumable checkpoints (latest.pt, ~1.2GB)
    max_hours: float = 11.0  # stop and save before Kaggle's 12h session limit
    dtype: str = 'auto'  # 'auto' = bfloat16 on Ampere+, float16 (with GradScaler) on T4/P100
    compile: bool = True  # switched off automatically on GPUs older than sm70 (P100)
    seed: int = 1337
