#File for taking the input training data and learning a BPE tokenizer, then tokenizing the data for training.

import pandas as pd
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from tokenizers.trainers import BpeTrainer
import os

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = BpeTrainer(
    vocab_size=8192,
    min_frequency=4,
    special_tokens=["<stop>", "<instruction>","<code>"], 
)
tokenizer.decoder = decoders.ByteLevel()

tokenizer.train(["data/train_plaintext.txt"], trainer)

tokenizer_path = "output/base_model/tokenizer/bpe_tokenizer.json"
os.makedirs("/".join(tokenizer_path.split("/")[:-1]), exist_ok=True)
tokenizer.save(tokenizer_path)
