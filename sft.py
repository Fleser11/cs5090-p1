#This is a stripped out version of the training file for nanoGPT based on Project 1 specification. 
#Nano gpt can be found here https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare/prepare.py


import argparse
import importlib
import os
import time
import math
from contextlib import nullcontext
from tokenizers import Tokenizer
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset
from model import GPTConfig, GPT
from dataclasses import asdict
import csv



def main( 
    out_dir,
    log_dir: str,
    eval_iters,
    eval_interval,
    log_interval,
    gradient_accumulation_steps, # used to simulate larger batch sizes,
    batch_size,
    block_size,
    n_layer,
    n_head,
    n_embed,
    learning_rate,
    lr_decay_iters,
    max_iters,
    weight_decay,
    beta1,
    beta2,
    grad_clip,
    warmup_iters,
    min_lr,
    vocab_size,
    dataset_name,
    tokenizer_path
    ):

    # -----------------------------------------------------------------------------
    # DDP settings
    backend = 'nccl' # 'nccl', 'gloo', etc.
    # system
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = True # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------

    
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    def get_datasets(dataset_name, tokenizer_path):
        tokenizer = Tokenizer.from_file(tokenizer_path)
        ds_train = load_dataset(dataset_name, split="train", streaming=True)
        # print(next(iter(ds_train)))
        ds_train_tokenized = ds_train.map(
            lambda row: {"tokens": tokenizer.encode(row["content"]+"<stop>").ids},
        )
        ds_train_shuffled = ds_train_tokenized.shuffle(buffer_size = 1000)
        ds_train_iter = iter(ds_train_shuffled)

        ds_val = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)
        ds_val_tokenized = ds_val.map(
            lambda row: {"tokens": tokenizer.encode(row["content"]+"<stop>").ids},
        )
        ds_val_shuffled = ds_val_tokenized.shuffle(buffer_size = 500)
        ds_val_iter = iter(ds_val_shuffled)

        return ds_train_iter, ds_val_iter


    ds_train_iter, ds_val_iter = get_datasets(dataset_name, tokenizer_path)

    train_buffer = []
    val_buffer = []

    def get_batch(split):

        if split == 'train':
            buffer = train_buffer
            iterator = ds_train_iter
        else:
            buffer = val_buffer
            iterator = ds_val_iter

        while len(buffer) < block_size * batch_size + 1:
            # print(f"Type: {type(sample)}, value: {sample}")
            buffer.extend(next(iterator)["tokens"])

        # ix = torch.randint(len(data) - block_size, (batch_size,))
        data = torch.tensor(buffer[:block_size*batch_size + 1], dtype=torch.int64)
        x = data[:block_size*batch_size].view(batch_size, block_size)
        y = data[1:block_size*batch_size + 1].view(batch_size, block_size)

        del buffer[:block_size*batch_size]

        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)

        return x, y

    iter_num = 0
    best_val_loss = 1e9

    # model init
    model_args = dict(n_layer=n_layer, n_head=n_head, 
                      n_embd=n_embed, block_size=block_size,
                      bias=False, vocab_size=vocab_size) # start with model_args from command line
    # init a new model from scratch
    print("Initializing a new model from scratch")
    print(f"Model args: {model_args}")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    # crop down the model block size if desired, using model surgery
    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    checkpoint = None # free up memory

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model) # requires PyTorch 2.0


    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * (it + 1) / (warmup_iters + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        elif it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        elif lr_decay_iters != None:
            decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
            return min_lr + coeff * (learning_rate - min_lr)
        else:
            return learning_rate


    # training loop
    X, Y = get_batch('train') # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model =  model # unwrap DDP container if needed
    running_mfu = -1.0

    os.makedirs("/".join(log_dir.split("/")[:-1]), exist_ok=True)
    f = open(log_dir, "w")
    csv_log_writer = csv.writer(f)

    headers = ["iteration", "train_loss", "val_loss"]
    csv_log_writer.writerow(headers)

    #setting up logging
    while True:

        train_loss = ""
        val_loss = ""

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if (lr_decay_iters != None or warmup_iters != None) else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            train_loss = lossf

        
        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

            train_loss = losses["train"].item()
            val_loss = losses["val"].item()

        f.flush()

        if iter_num % log_interval == 0:
            csv_log_writer.writerow([iter_num, train_loss, val_loss])
            

        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "-c", "--config", required=True
    )
    args = argument_parser.parse_args()
    # accept either a module name (configs.base_config) or a path (configs/base_config.py)
    config_name = args.config.replace("/", ".")
    module = importlib.import_module(config_name)

    main(**asdict(module.TrainingConfig()))