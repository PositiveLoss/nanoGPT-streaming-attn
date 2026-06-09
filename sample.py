"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
attention_impl = 'checkpoint' # checkpoint/default, or 'flash', 'manual', 'fast_article', 'triton_article'
article_degree = 2
article_block_size = 128
article_compressor = 'sorted_pair'
article_seed = 0
article_query_chunk_size = 2048
article_low_mode = 'auto'
article_denominator_eps = 1e-12
triton_compress_stride = 2
triton_backward = 'sdpa'
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

def apply_attention_overrides(model_args):
    if attention_impl == 'checkpoint':
        return model_args
    overrides = {
        'attention_impl': attention_impl,
        'article_degree': article_degree,
        'article_block_size': article_block_size,
        'article_compressor': article_compressor,
        'article_seed': article_seed,
        'article_query_chunk_size': article_query_chunk_size,
        'article_low_mode': article_low_mode,
        'article_denominator_eps': article_denominator_eps,
        'triton_compress_stride': triton_compress_stride,
        'triton_backward': triton_backward,
    }
    for key, value in overrides.items():
        if value is not None:
            model_args[key] = value
    return model_args

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
if device_type == 'cuda':
    torch.cuda.reset_peak_memory_stats(device)
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = apply_attention_overrides(dict(checkpoint['model_args']))
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    override_args = apply_attention_overrides(dict(dropout=0.0))
    model = GPT.from_pretrained(init_from, override_args)

model.eval()
model.to(device)
print(f"Using attention_impl={model.config.attention_impl}")
if model.config.attention_impl == 'fast_article' and model.config.article_low_mode == 'auto':
    print("WARNING: using article_low_mode='stream' for fast_article sampling to avoid large CUDA prefix tensors")
    model.config.article_low_mode = 'stream'
    for block in model.transformer.h:
        block.attn.article_low_mode = 'stream'
if model.config.attention_impl in {'fast_article', 'triton_article'} and compile:
    print(f"WARNING: disabling torch.compile for {model.config.attention_impl} attention during sampling")
    compile = False
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')

if device_type == 'cuda':
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    print(
        "GPU memory: "
        f"allocated={allocated:.2f} MiB, "
        f"reserved={reserved:.2f} MiB, "
        f"peak_allocated={peak_allocated:.2f} MiB, "
        f"peak_reserved={peak_reserved:.2f} MiB"
    )
