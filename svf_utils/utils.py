import os

import torch
import torch.nn as nn

from typing import Dict
import re

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.6f}"
    )


def set_clip_trainable_parameters(args, model: nn.Module) -> None:
    if(args.include_train_layers == "all"):
        include_layers = [0,1,2,3,4,5,6,7,8,9,10,11]
    elif(args.include_train_layers == "top"):
        include_layers = [8,9,10,11]
    elif(args.include_train_layers == "bottom"):
        include_layers = [0,1,2,3]
    elif(args.include_train_layers == "middle"):
        include_layers = [4,5,6,7]
    elif(args.include_train_layers == "top_middle"):
        include_layers = [4,5,6,7,8,9,10,11]
    elif(args.include_train_layers == "bottom_middle"):
        include_layers = [0,1,2,3,4,5,6,7]
    elif(args.include_train_layers == "top_bottom"):
        include_layers = [0,1,2,3,8,9,10,11]
    for name, param in model.named_parameters():
        param.requires_grad_(False)
        
        vector = "vector_S"
        if vector in name:
            layer_num = int(re.findall(r'\d+', name)[0])
            if(layer_num in include_layers):
                if("visual" in name):
                    if(args.train_image_encoder):
                        if("mlp" in name and args.train_mlp):
                            param.requires_grad_(True)
                        if("attn.q_proj" in name and args.train_q):
                            param.requires_grad_(True)
                        if("attn.k_proj" in name and args.train_k):
                            param.requires_grad_(True)
                        if("attn.v_proj" in name and args.train_v):
                            param.requires_grad_(True)
                        if("attn.proj" in name and args.train_o_proj):
                            param.requires_grad_(True)
                else:
                    if(args.train_text_encoder):
                        if("mlp" in name and args.train_mlp):
                            param.requires_grad_(True)
                        if("attn.q_proj" in name and args.train_q):
                            param.requires_grad_(True)
                        if("attn.k_proj" in name and args.train_k):
                            param.requires_grad_(True)
                        if("attn.v_proj" in name and args.train_v):
                            param.requires_grad_(True)
                        if("attn.proj" in name and args.train_o_proj):
                            param.requires_grad_(True)

def set_biomedclip_trainable_parameters(args, model: nn.Module) -> None:
    if(args.include_train_layers == "all"):
        include_layers = [0,1,2,3,4,5,6,7,8,9,10,11]
    elif(args.include_train_layers == "top"):
        include_layers = [8,9,10,11]
    elif(args.include_train_layers == "bottom"):
        include_layers = [0,1,2,3]
    elif(args.include_train_layers == "middle"):
        include_layers = [4,5,6,7]
    elif(args.include_train_layers == "top_middle"):
        include_layers = [4,5,6,7,8,9,10,11]
    elif(args.include_train_layers == "bottom_middle"):
        include_layers = [0,1,2,3,4,5,6,7]
    elif(args.include_train_layers == "top_bottom"):
        include_layers = [0,1,2,3,8,9,10,11]
    for name, param in model.named_parameters():
        param.requires_grad_(False)

        vector = "vector_S"
        if vector in name:
            layer_num = int(re.findall(r'\d+', name)[0]) if len(re.findall(r'\d+', name)) != 0 else None
            if(layer_num is None):
                continue
            if(layer_num in include_layers):
                if("visual" in name and args.train_image_encoder):
                    if("mlp" in name and args.train_mlp):
                        param.requires_grad_(True)
                    if("attn.q_proj" in name and args.train_q):
                        param.requires_grad_(True)
                    if("attn.k_proj" in name and args.train_k):
                        param.requires_grad_(True)
                    if("attn.v_proj" in name and args.train_v):
                        param.requires_grad_(True)
                    if("attn.proj" in name and args.train_o_proj):
                        param.requires_grad_(True)
                    
                if("text" in name and args.train_text_encoder):
                    if("attention.self.query" in name and args.train_q):
                        param.requires_grad_(True)
                    if("attention.self.key" in name and args.train_k):
                        param.requires_grad_(True)
                    if("attention.self.value" in name and args.train_v):
                        param.requires_grad_(True)
                    if("attention.output" in name and args.train_o_proj):
                        param.requires_grad_(True)
                    if("intermediate.dense" in name and args.train_mlp):
                        param.requires_grad_(True)
                    if("output.dense" in name and args.train_mlp):
                        param.requires_grad_(True)


def save_model(args, model):
   
    metadata = {
        'image_encoder': args.train_image_encoder,
        'text_encoder': args.train_text_encoder,  
    }

    weights = model.state_dict()

    save_data = {
        'weights': weights,
        'metadata': metadata
    }

    # to manage names like ViT-B/16
    backbone = args.backbone.replace('/', '').replace('-', '').lower()
    save_dir = f'{args.save_path}/{args.dataset}/{backbone}/{args.shots}shots/seed{args.seed}'
    os.makedirs(save_dir, exist_ok=True)

    save_path = f'{save_dir}/{args.filename}.pt'
    torch.save(save_data, save_path)
    print(f'Model weights saved to {save_path}')


def load_model(args, model):
    # to manage names like ViT-B/16
    backbone = args.backbone.replace('/', '').replace('-', '').lower()

    if(len(args.save_path.split("/")) >= 1): # For Cross-dataset and Domain Generalization experiments
        load_path = f'{args.save_path}/{backbone}/{args.shots}shots/seed{args.seed}/base_model.pt'
    else:
        load_path = f'{args.save_path}/{args.dataset}/{backbone}/{args.shots}shots/seed{args.seed}/base_model.pt'

    if not os.path.exists(load_path):
        raise FileNotFoundError(f'File {load_path} does not exist.')

    loaded_data = torch.load(load_path)

    metadata = loaded_data['metadata']
    weights = loaded_data['weights']
    model.load_state_dict(weights)

    print(f'Model weights loaded from {load_path}')
