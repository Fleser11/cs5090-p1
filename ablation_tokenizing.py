#File for taking the input training data and learning a BPE tokenizer, then tokenizing the data for training.

import pandas as pd
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from tokenizers.trainers import BpeTrainer
import os

tokenizer = Tokenizer(models.BPE(vocab={char: id for id, char in enumerate(pre_tokenizers.ByteLevel.alphabet())}, merges=[]))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.add_special_tokens(["<stop>"])
tokenizer.decoder = decoders.ByteLevel()

# 
tokenizer_path = "output/ablation_model/tokenizer/bpe_tokenizer.json"
os.makedirs("/".join(tokenizer_path.split("/")[:-1]), exist_ok=True)
tokenizer.save(tokenizer_path)
