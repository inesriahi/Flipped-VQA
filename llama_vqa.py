import torch
import json
from llama import ModelArgs, Tokenizer, Transformer
from pathlib import Path

def LLaMA_VQA(args, **kwargs):
    # Load model parameters from a JSON file
    with open(f'{args.llama_model_path}{args.model}/params.json', "r") as f:
        params = json.loads(f.read())
    # Initialize the tokenizer
    tokenizer = Tokenizer(model_path=f'{args.llama_model_path}/tokenizer.model')
    print(f"Using model: {args.model}")
    
    # Locate and sort the checkpoint files for the model
    checkpoints = (Path(args.llama_model_path) / args.model).glob("*.pth")
    checkpoints = sorted(checkpoints)

    # Load the model checkpoints
    loaded = []
    for x in checkpoints:
        print("loading from", x)
        loaded.append(torch.load(x, map_location="cpu"))
    
    # Combine the checkpoints into a single state dictionary
    if len(loaded) == 1:
        full_state_dict = loaded[0]
    else:
        full_state_dict = {}
        split_dims = {}
        
        # Helper function to add weights to the state dictionary with consideration for split dimensions
        def add_weight_with_split_dim(name, dim):
            if dim < 0:  # Broadcast without split
                full_state_dict[name] = loaded[0][name].clone()
            else: # Concatenate weights along the specified dimension
                full_state_dict[name] = torch.cat([x[name] for x in loaded], dim=dim)
            for x in loaded:
                del x[name]
            split_dims[name] = dim
        
        # Add weights for embeddings and normalization layers
        add_weight_with_split_dim("tok_embeddings.weight", 1)
        add_weight_with_split_dim("norm.weight", -1)
        add_weight_with_split_dim("output.weight", 0)
        
        # Add weights for each layer in the model
        for i in range(params["n_layers"]):
            print("gathering layer %d of %d" % (i, params["n_layers"]))
            layer_prefix = f"layers.{i}."
            bcast_names = ["attention_norm.weight", "ffn_norm.weight"]
            column_parallel_names = ["attention.wq.weight", "attention.wk.weight", "attention.wv.weight", "feed_forward.w1.weight", "feed_forward.w3.weight"]
            row_parallel_names = ["attention.wo.weight", "feed_forward.w2.weight"]
            for key in bcast_names:
                add_weight_with_split_dim(layer_prefix + key, -1)
            for key in column_parallel_names:
                add_weight_with_split_dim(layer_prefix + key, 0)
            for key in row_parallel_names:
                add_weight_with_split_dim(layer_prefix + key, 1)
    
    # Set up model arguments and initialize the LLaMA_VQA model
    model_args: ModelArgs = ModelArgs(max_seq_len=args.max_seq_len, max_batch_size=32, adapter_len=args.adapter_len, adapter_layer=args.adapter_layer, **params)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model_llama_vqa = Transformer(model_args, args)
    torch.set_default_tensor_type(torch.FloatTensor)

    # Load the combined state dictionary into the model, handling missing or unexpected keys
    missing_keys, unexpected_keys = model_llama_vqa.load_state_dict(full_state_dict, strict=False)

    # Set training requirements and data types for specific parameters in the model
    for name, param in model_llama_vqa.named_parameters():
        if ('gate' in name) or ('adapter' in name) or ('temporal_emb' in name) or ('visual_proj' in name):
            param.requires_grad = True
            param.data = param.data.float()
        else:
            param.requires_grad = False

    return model_llama_vqa