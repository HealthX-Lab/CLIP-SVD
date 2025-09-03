
import random
import argparse  
import numpy as np 
import torch

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_arguments():

    parser = argparse.ArgumentParser()
    # Dataset arguments
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--dataset', type=str, default='dtd')
    parser.add_argument('--shots', default=1, type=int)
    # Model arguments
    parser.add_argument('--backbone', default='ViT-B/16', type=str)
    # Training arguments
    parser.add_argument('--lr', default=2e-3, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--use_scaler', default=False, type=bool)
    parser.add_argument('--tasks', default=3, type=int)
    parser.add_argument('--min_term', default=1, type=int)
    parser.add_argument('--train_epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--include_train_layers', default="top_bottom", choices = ["top","bottom","middle", "top_bottom", "top_middle", "bottom_middle"])
    parser.add_argument('--train_image_encoder', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--train_text_encoder', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--train_mlp', default=True, type=lambda x: (str(x).lower() == 'true'))

    parser.add_argument('--train_q', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--train_k', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--train_v', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--train_proj', default=True, type=lambda x: (str(x).lower() == 'true')) 
    parser.add_argument('--train_patch_embed', default=False, type=lambda x: (str(x).lower() == 'true')) 
    parser.add_argument('--model', default = "CLIP", choices = ["CLIP", "BiomedCLIP"])
    parser.add_argument('--save_path', default=None, help='path to save the lora modules after training, not saved if None')
    parser.add_argument('--output_dir', default="outputs_ablations", help='path to save the lora modules after training, not saved if None')
    parser.add_argument('--filename', default='lora_weights', help='file name to save the lora weights (.pt extension will be added)')
    
    parser.add_argument('--eval_only', default=False, action='store_true', help='only evaluate the model (save_path should not be None)')
    args = parser.parse_args()

    return args