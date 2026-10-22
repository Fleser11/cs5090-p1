from dataclasses import dataclass

@dataclass
class TrainingConfig:
    out_dir: str = 'output/sft_codeparrot_model/python-coder/'
    init_from: str = 'output/base_model/python-coder/ckpt.pt'  # pretrained checkpoint to fine-tune
    dataset_name: str = "codeparrot/codeparrot-clean"  # same corpus as pretraining
    val_dataset_name: str = "codeparrot/codeparrot-clean-valid"  # held-out files for validation
    code_column: str = "content"
    tokenizer_path: str = "output/base_model/tokenizer/bpe_tokenizer.json"
    log_dir: str = "output/sft_codeparrot_model/logs/sft_codeparrot_logs.csv"
    max_rows: int = 20000  # training files streamed from the dataset
    val_rows: int = 1000  # validation files streamed from the valid split
    max_seq_len: int = 384  # units longer than this many tokens are skipped
    cuts_per_unit: int = 1  # prefix/completion splits made from each unit
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_iters: int = 2000
    eval_interval: int = 100
    eval_iters: int = 50
