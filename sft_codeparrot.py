# Bare-bones SFT on the pretraining corpus (codeparrot-clean): given a code prefix, learn to finish
# the function/class and emit <stop>. Same as sft.py except for where the units come from:
# top-level functions/classes with a docstring, validated on files from codeparrot-clean-valid.
# Each example is a complete ast-valid unit cut at a random point; the loss covers only the completion.

import argparse
import ast
import importlib
import os
import random
import re
import shutil
from dataclasses import asdict

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

from model import GPT, GPTConfig

def extract_units(code):
    """Top-level functions/classes of `code` that have a docstring."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return []
    defs = [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and ast.get_docstring(n)]
    return [(ast.get_source_segment(code, n) or "") + "\n" for n in defs]


def make_examples(units, tokenizer, stop_id, max_seq_len, cuts_per_unit):
    """Returns (token ids ending in <stop>, cut index) pairs; tokens before the cut are the prefix."""
    examples = []
    for unit in units:
        enc = tokenizer.encode(unit)
        n = len(enc.ids)
        if n < 8 or n + 1 > max_seq_len:
            continue
        # cut indices where the token starts a line (or follows only indentation)
        line_cuts = [k for k in range(4, n)
                     if not unit[unit.rfind("\n", 0, enc.offsets[k][0]) + 1:enc.offsets[k][0]].strip()]
        for _ in range(cuts_per_unit):
            k = random.choice(line_cuts) if line_cuts and random.random() < 0.5 else random.randrange(4, n)
            examples.append((torch.tensor(enc.ids + [stop_id]), k))
    return examples


def main(out_dir, init_from, dataset_name, val_dataset_name, code_column, tokenizer_path, log_dir, max_rows,
         val_rows, max_seq_len, cuts_per_unit, batch_size, learning_rate, max_iters, eval_interval, eval_iters):
    random.seed(1337)
    torch.manual_seed(1337)
    ctx = torch.autocast('cuda', dtype=torch.bfloat16)

    # gen_pipe.py looks for the tokenizer next to python-coder/, so copy it there
    os.makedirs(os.path.join(os.path.dirname(out_dir.rstrip("/")), "tokenizer"), exist_ok=True)
    shutil.copy(tokenizer_path, os.path.join(os.path.dirname(out_dir.rstrip("/")), "tokenizer", "bpe_tokenizer.json"))

    # ---- data: train units from the pretraining corpus, val units from its held-out valid files ----
    tokenizer = Tokenizer.from_file(tokenizer_path)
    stop_id = tokenizer.token_to_id("<stop>")

    def load_units(name, rows):  # {normalized text: unit}, which dedupes within the set
        units = []
        for i, row in enumerate(load_dataset(name, split="train", streaming=True)):
            if i >= rows:
                break
            units += extract_units(row[code_column])
        return {re.sub(r"\s+", " ", u).strip(): u for u in units}

    train_units = load_units(dataset_name, max_rows)
    val_units = [u for k, u in load_units(val_dataset_name, val_rows).items() if k not in train_units]
    splits = {
        'val': make_examples(val_units, tokenizer, stop_id, max_seq_len, cuts_per_unit),
        'train': make_examples(list(train_units.values()), tokenizer, stop_id, max_seq_len, cuts_per_unit),
    }
    print({k: len(v) for k, v in splits.items()}, "examples")

    def get_batch(split):
        x = torch.zeros(batch_size, max_seq_len - 1, dtype=torch.long)
        y = torch.full_like(x, -1)  # -1 = ignored by the loss (padding and prefix)
        for b, (seq, k) in enumerate(random.choices(splits[split], k=batch_size)):
            x[b, :len(seq) - 1] = seq[:-1]
            y[b, k - 1:len(seq) - 1] = seq[k:]
        return x.cuda(), y.cuda()

    # ---- model: start from the pretrained checkpoint ----
    checkpoint = torch.load(init_from, map_location='cpu', weights_only=True)
    model = GPT(GPTConfig(**checkpoint['model_args']))
    # checkpoints saved from a compiled model have an `_orig_mod.` prefix on every key
    model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in checkpoint['model'].items()})
    model.cuda()
    optimizer = model.configure_optimizers(0.1, learning_rate, (0.9, 0.95), 'cuda')

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = []
        for split in ['train', 'val']:
            losses = []
            for _ in range(eval_iters):
                with ctx:
                    losses.append(model(*get_batch(split))[1].item())
            out.append(sum(losses) / len(losses))
        model.train()
        return out

    # ---- train ----
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    log = open(log_dir, "w")
    log.write("iteration,train_loss,val_loss\n")
    best_val = float('inf')
    model.train()
    for it in range(max_iters + 1):
        if it % eval_interval == 0:
            train_loss, val_loss = estimate_loss()
            print(f"step {it}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
            log.write(f"{it},{train_loss},{val_loss}\n")
            log.flush()
            if val_loss < best_val and it > 0:
                best_val = val_loss
                torch.save({'model': model.state_dict(), 'model_args': checkpoint['model_args'],
                            'iter_num': it, 'best_val_loss': best_val}, os.path.join(out_dir, 'ckpt.pt'))

        with ctx:
            _, loss = model(*get_batch('train'))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("-c", "--config", required=True)
    args = argument_parser.parse_args()
    # accept either a module name or a path like configs/training_configs/sft_config.py
    module = importlib.import_module(args.config.removesuffix(".py").replace("/", "."))
    main(**asdict(module.TrainingConfig()))
