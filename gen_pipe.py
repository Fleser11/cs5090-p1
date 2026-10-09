import os
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT
from tokenizers import Tokenizer
import csv
import argparse

"""
Takes a configuration for generating outputs from the model
for evaluation. Used for project Phase 1 and 2
"""

def main(model_dir, gen_config, output_path):

    max_new_tokens = 50 # number of tokens generated in each sample
    temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 5 # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed = 1234
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    compile = True # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # model
    # init from a model saved in a specific directory
    ckpt_path = f"{model_dir}/python-coder/ckpt.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    tik = Tokenizer.from_file(f"{model_dir}/tokenizer/bpe_tokenizer.json")
    encode = lambda s: tik.encode(s).ids
    decode = lambda l: tik.decode(l, skip_special_tokens=False)

    os.makedirs("/".join(output_path.split("/")[:-1]), exist_ok=True)

    with torch.no_grad():
        with ctx:
            in_f = open(gen_config)
            out_f = open(output_path, "w+")
            dictReader = csv.DictReader(in_f)
            dictWriter = csv.DictWriter(out_f, fieldnames= [fieldname for fieldname in dictReader.fieldnames] + ["output"])
            dictWriter.writeheader()
            for line in dictReader:
                prompt = line["input_prompt"]
                print(prompt)
                ids = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
                output = decode(model.generate(ids, max_new_tokens, temperature=temperature, top_k=top_k)[0].tolist())
                dictWriter.writerow({
                    "test_num": line["test_num"],
                    "input_prompt": line["input_prompt"],
                    "output": output
                })


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--model_dir", required=True
    )
    argparser.add_argument(
        "--gen_config", required=True
    )
    argparser.add_argument(
        "--output_path", required=True
    )

    args = argparser.parse_args()
    main(args.model_dir, args.gen_config, args.output_path)
