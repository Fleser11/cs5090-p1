from dataclasses import dataclass

@dataclass
class TrainingConfig:
    out_dir: str = 'output/sft_model/python-coder/'
    init_from: str = 'output/base_model/python-coder/ckpt.pt'  # pretrained checkpoint to fine-tune
    dataset_name: str = "flytech/python-codes-25k"
    code_column: str = "output"  # column holding the code ("content" for codeparrot)
    tokenizer_path: str = "output/base_model/tokenizer/bpe_tokenizer.json"
    log_dir: str = "output/sft_model/logs/sft_logs.csv"
    max_rows: int = 50000  # rows streamed from the dataset
    max_seq_len: int = 256  # units longer than this many tokens are skipped
    cuts_per_unit: int = 3  # prefix/completion splits made from each unit
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_iters: int = 5000
    eval_interval: int = 100
    eval_iters: int = 50
