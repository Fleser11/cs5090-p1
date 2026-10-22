#This is a stripped out version of the training file for nanoGPT based on Project 1 specification.
#Nano gpt can be found here https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare/prepare.py

# Resumable (e.g. across Kaggle sessions): out_dir/latest.pt (model, optimizer, scaler, iteration, data position)
# is saved every save_interval iterations and when max_hours is reached. Rerunning the same command picks it up,
# or pass --resume_from <path to a latest.pt from an earlier session>. The data position is a fixed-seed shuffle
# plus the number of rows consumed, so a resumed run skips what it already trained on (and starts a new pass
# with a new shuffle if the dataset is ever exhausted).
#
# Runs on one CUDA GPU or on a TPU (config device='auto' picks CUDA if present, else TPU via torch_xla).
# On a TPU every core runs this function: each core streams its own slice of the data, gradients are averaged
# across cores before the optimizer step, and gradient_accumulation_steps is divided by the number of cores
# so the tokens per iteration stay the same as on a single GPU.


import argparse
import builtins
import importlib
import os
import time
import math
from tokenizers import Tokenizer
import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from model import GPTConfig, GPT
from dataclasses import asdict
import csv


def pick_device(device):
    if device != 'auto':
        return device
    if torch.cuda.is_available():
        return 'cuda'
    try:
        import torch_xla # noqa: F401
        return 'tpu'
    except ImportError:
        raise RuntimeError("no CUDA GPU or torch_xla (TPU) found")


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
    tokenizer_path,
    rope=False, # rotary position encodings instead of learned absolute position embeddings
    resume_from="", # latest.pt from an earlier session (out_dir/latest.pt is tried first)
    save_interval=500, # iterations between resumable checkpoints
    max_hours=1e9, # stop and save after this many hours (Kaggle sessions end at 9h on TPU, 12h on GPU)
    dtype='auto', # 'auto' = bfloat16 on Ampere+, float16 (with GradScaler) on T4/P100; always bfloat16 on TPU
    compile=True, # switched off automatically on GPUs older than sm70 (P100) and on TPU
    seed=1337,
    device='auto', # 'auto', 'cuda' or 'tpu'
    ):

    start_time = time.time()
    # -----------------------------------------------------------------------------
    tpu = pick_device(device) == 'tpu'
    if tpu:
        os.environ.setdefault('XLA_NO_SPECIAL_SCALARS', '1') # a new learning rate each step must not force a recompile
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        device = torch_xla.device() if hasattr(torch_xla, 'device') else xm.xla_device()
        rank, world_size = xr.global_ordinal(), xr.world_size()
        sync = torch_xla.sync if hasattr(torch_xla, 'sync') else xm.mark_step # run the queued XLA graph
        dtype = 'bfloat16' # TPUs are bfloat16 native, so no GradScaler either
        compile = False # torch.compile is for CUDA; XLA already compiles the graph
        gradient_accumulation_steps = max(1, gradient_accumulation_steps // world_size)
        eval_iters = max(1, eval_iters // world_size)
    else:
        device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
        rank, world_size = 0, 1
        sync = lambda: None
        major = torch.cuda.get_device_capability()[0]
        if dtype == 'auto':
            dtype = 'bfloat16' if major >= 8 else 'float16' # T4/P100 have no real bf16
        compile = compile and major >= 7 # torch.compile needs Triton, which needs sm70+ (not P100)
    master_process = rank == 0
    _print = builtins.print
    def print(*args, **kwargs): # only the master process prints
        if master_process:
            _print(*args, **kwargs)
    # -----------------------------------------------------------------------------

    tokens_per_iter = gradient_accumulation_steps * world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,} | {'tpu x%d' % world_size if tpu else 'cuda'} | dtype {dtype} | compile {compile}")

    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    if tpu:
        ctx = torch.autocast('xla', dtype=torch.bfloat16)
    else:
        ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)

    # resume: out_dir/latest.pt wins over resume_from (so a restarted session keeps its own progress)
    model_args = dict(n_layer=n_layer, n_head=n_head,
                      n_embd=n_embed, block_size=block_size,
                      bias=False, vocab_size=vocab_size, rope=rope) # start with model_args from command line
    latest_path = os.path.join(out_dir, 'latest.pt')
    resume_path = next((p for p in (latest_path, resume_from) if p and os.path.exists(p)), None)
    iter_num, best_val_loss, data_state, checkpoint = 0, 1e9, dict(epoch=0, rows=0), None
    if resume_path:
        checkpoint = torch.load(resume_path, map_location='cpu', weights_only=True)
        assert checkpoint['model_args'] == model_args, f"config does not match {resume_path}: {checkpoint['model_args']}"
        iter_num, best_val_loss = checkpoint['iter_num'], checkpoint['best_val_loss']
        # rows are saved as the total over all cores, so a run can resume on different hardware
        data_state = dict(checkpoint['data_state'], rows=checkpoint['data_state']['rows'] // world_size)
        print(f"resuming from {resume_path} at iteration {iter_num} (pass {data_state['epoch']}, {data_state['rows'] * world_size} rows in)")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    assert tokenizer.get_vocab_size() <= vocab_size, "tokenizer is larger than the model's vocab_size"

    def make_iter(name, buffer_size, shuffle_seed, skip):
        # shuffle with a fixed seed, give this core its slice, skip rows already used, then tokenize (skipped rows cost nothing)
        ds = load_dataset(name, split="train", streaming=True).shuffle(seed=shuffle_seed, buffer_size=buffer_size)
        ds = split_dataset_by_node(ds, rank=rank, world_size=world_size).skip(skip)
        return iter(ds.map(lambda row: {"tokens": tokenizer.encode(row["content"]+"<stop>").ids}))

    iterators = {
        'train': make_iter(dataset_name, 1000, seed + data_state['epoch'], data_state['rows']),
        'val': make_iter("codeparrot/codeparrot-clean-valid", 500, seed, 0),
    }
    buffers = {'train': [], 'val': []}

    allowed_licenses = ["apache-2.0", "mit", "bsd-3-clause"]

    def get_batch(split):

        buffer = buffers[split]
        while len(buffer) < block_size * batch_size + 1:
            try:
                row = next(iterators[split])
                while row['license'] not in allowed_licenses:
                    row = next(iterators[split])
            except StopIteration: # ran out of data: start another pass with a different shuffle
                if split == 'train':
                    data_state['epoch'] += 1
                    data_state['rows'] = 0
                    print(f"dataset exhausted, starting pass {data_state['epoch']}")
                    iterators['train'] = make_iter(dataset_name, 1000, seed + data_state['epoch'], 0)
                else:
                    iterators['val'] = make_iter("codeparrot/codeparrot-clean-valid", 500, seed, 0)
                row = next(iterators[split])
            if split == 'train':
                data_state['rows'] += 1
            buffer.extend(row["tokens"])

        data = torch.tensor(buffer[:block_size*batch_size + 1], dtype=torch.int64)
        x = data[:block_size*batch_size].view(batch_size, block_size)
        y = data[1:block_size*batch_size + 1].view(batch_size, block_size)

        del buffer[:block_size*batch_size]

        if tpu:
            return x.to(device), y.to(device)
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)

    # model init
    print("Initializing a new model from scratch" if checkpoint is None else "Initializing model from checkpoint")
    print(f"Model args: {model_args}")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    if checkpoint:
        model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in checkpoint['model'].items()})
    model.to(device)
    if tpu:
        model.transformer.wte.weight = model.lm_head.weight # make sure weight tying survived the move
        if checkpoint is None and world_size > 1:
            xm.broadcast_master_param(model) # every core starts from the master's weights

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler('cuda', enabled=(not tpu and dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), 'xla' if tpu else 'cuda')
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
    checkpoint = None # free up memory

    # compile the model
    raw_model = model # the uncompiled module, so saved state dicts have plain key names
    if compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model) # requires PyTorch 2.0

    def save(obj, path):
        if tpu:
            xm.save(obj, path) # every core calls this: tensors go to the CPU and only the master writes
        else:
            torch.save(obj, path)

    def save_latest():
        # write to a temp file first so a crash mid-save cannot corrupt the previous checkpoint
        save({
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'data_state': dict(data_state, rows=data_state['rows'] * world_size),
        }, latest_path + '.tmp')
        if master_process:
            os.replace(latest_path + '.tmp', latest_path)
        print(f"saved resumable checkpoint at iteration {iter_num}")

    # estimates the validation loss on a fixed set of batches (drawn once, then reused)
    val_batches = []
    @torch.no_grad()
    def estimate_val_loss():
        if not val_batches:
            val_batches.extend(get_batch('val') for _ in range(eval_iters))
        model.eval()
        losses = torch.zeros(eval_iters)
        for k, (X, Y) in enumerate(val_batches):
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        model.train()
        val = losses.mean()
        if tpu: # average over the cores so they all agree on the number
            val = (xm.all_reduce(xm.REDUCE_SUM, val.to(device)) / world_size).cpu()
        return val.item()

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * (it + 1) / (warmup_iters + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        elif it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)


    # training loop
    X, Y = get_batch('train') # fetch the very first batch
    t0 = time.time()

    f = csv_log_writer = None
    new_log = not (resume_path and os.path.exists(log_dir))
    if master_process:
        os.makedirs(os.path.dirname(log_dir) or ".", exist_ok=True)
        f = open(log_dir, "w" if new_log else "a")
        csv_log_writer = csv.writer(f)
        if new_log:
            headers = ["iteration", "train_loss", "val_loss"]
            csv_log_writer.writerow(headers)

    #setting up logging
    while True:

        train_loss = ""
        val_loss = ""

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num)
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
            sync() # TPU: run this micro-step's graph now so the queued graph stays small
        if tpu:
            xm.reduce_gradients(optimizer) # average the gradients over the cores
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
        sync()

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.detach() * gradient_accumulation_steps
            if tpu:
                lossf = xm.all_reduce(xm.REDUCE_SUM, lossf) / world_size
            lossf = lossf.item()
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.0f}ms, {tokens_per_iter/dt:,.0f} tokens/s")
            train_loss = lossf


        # evaluate the loss on the validation set and write checkpoints
        if iter_num % eval_interval == 0:
            val_loss = estimate_val_loss()
            print(f"step {iter_num}: val loss {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    save(checkpoint, os.path.join(out_dir, 'ckpt.pt')) # the best model, as gen_pipe.py and sft.py load
                    checkpoint = None

        if iter_num % log_interval == 0 and master_process:
            csv_log_writer.writerow([iter_num, train_loss, val_loss])
            f.flush()


        iter_num += 1

        # periodic resumable checkpoint
        if iter_num % save_interval == 0:
            save_latest()

        # termination conditions
        out_of_time = time.time() - start_time > max_hours * 3600
        if tpu: # every core must make the same decision or the collectives hang
            out_of_time = bool(xm.all_reduce(xm.REDUCE_MAX, torch.tensor(float(out_of_time), device=device)).item())
        if iter_num > max_iters or out_of_time:
            save_latest()
            print("time budget reached, rerun to resume" if out_of_time and iter_num <= max_iters else "done")
            break


def _tpu_entry(index, config): # torch_xla.launch runs this once per TPU core
    main(**config)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "-c", "--config", required=True
    )
    argument_parser.add_argument(
        "--resume_from", default=None, help="latest.pt from an earlier session (overrides the config)"
    )
    args = argument_parser.parse_args()
    # accept either a module name (configs.base_config) or a path (configs/base_config.py)
    config_name = args.config.removesuffix(".py").replace("/", ".")
    module = importlib.import_module(config_name)

    config = asdict(module.TrainingConfig())
    if args.resume_from:
        config["resume_from"] = args.resume_from
    if pick_device(config.get("device", "auto")) == 'tpu':
        os.environ.setdefault('XLA_NO_SPECIAL_SCALARS', '1')
        import torch_xla
        if hasattr(torch_xla, 'launch'):
            torch_xla.launch(_tpu_entry, args=(config,))
        else:
            import torch_xla.distributed.xla_multiprocessing as xmp
            xmp.spawn(_tpu_entry, args=(config,))
    else:
        main(**config)
