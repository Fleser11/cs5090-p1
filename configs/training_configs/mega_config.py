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
    gradient_accumulation_steps: int = 4  # batch_size * this * block_size = 65,536 tokens per iteration
    batch_size: int = 16  # smaller micro-batch so it fits a 16GB Kaggle GPU
    block_size: int = 512
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    learning_rate: float = 5e-4
    min_lr: float = 5e-5  # learning_rate / 10
    max_iters: int = 60000
    lr_decay_iters: int = 60000  # should match max_iters
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_iters: int = 500
    vocab_size: int = 8192