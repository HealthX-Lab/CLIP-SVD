import torch
import torch.nn.functional as F
import torch.nn as nn

from utils import *

from svf_utils.utils import set_clip_trainable_parameters, set_biomedclip_trainable_parameters, \
                            print_trainable_parameters, save_model, load_model
from svf_utils.svf_torch_final import resolver
from svf_utils.PlainHeadAttention import PlainMultiHeadAttention
from svf_utils.BiomedCLIPHeadAttention import BiomedCLIPMultiHeadAttention
from open_clip import create_model_from_pretrained, get_tokenizer
import copy

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--load_path",
    type=str,
    default="outputs_tsne/dtd/vitb16/16shots/seed2/base_model.pt",
    help="Path to the model checkpoint"
)
args = parser.parse_args()

def compute_singular_values(attn_weights):
    """
    Compute and normalize singular values for each attention head in a given layer.
    attn_weights: (num_heads, dim, dim) tensor
    Returns a tensor of normalized singular values of shape (num_heads, min(dim, dim)).
    """
    num_heads = attn_weights.shape[0]
    singular_values = []
    for head in range(num_heads):
        _, S, _ = torch.svd(attn_weights[head])
        
        S_norm = S / torch.norm(S, p=2) 
        singular_values.append(S_norm)
    return torch.stack(singular_values) 

def compare_singular_values(before, after):
    """
    Compare normalized singular values before and after fine-tuning using L2 norm.
    Returns a list of tuples sorted by absolute change magnitude [(layer, head, signed_L2_norm_change)].
    """
    sv_changes = []
    for layer in before.keys():

        diff = after[layer] - before[layer]  
        l2_norm_change = torch.norm(diff, p=2, dim=1)
    
        sign = torch.sign(diff.mean(dim=1))
        signed_l2_change = l2_norm_change * sign
        
        for head in range(len(signed_l2_change)):
            sv_changes.append((layer, head, signed_l2_change[head].item()))
    
    sv_changes.sort(key=lambda x: abs(x[2]))
    return sv_changes

before_finetune_V = {}
after_finetune_V = {}
before_finetune_O = {}
after_finetune_O = {}

clip_model, preprocess = clip.load("ViT-B/16")

for i,module in enumerate(clip_model.visual.transformer.resblocks):
    new_module = PlainMultiHeadAttention()
    new_module.set_parameters(module.attn)
    module.attn = new_module

for i,module in enumerate(clip_model.transformer.resblocks):
    new_module = PlainMultiHeadAttention(embed_dim=512, num_heads=8)
    new_module.set_parameters(module.attn)
    module.attn = new_module

clip_model = resolver(clip_model.cuda().float())
clip_model = clip_model.cuda() 

model_state_before = clip_model.state_dict()

loaded_data = torch.load(args.load_path)

metadata = loaded_data['metadata']
weights = loaded_data['weights']
model_state_after = weights

num_layers = 12
num_heads = 12
head_dim = 64

for layer in range(num_layers):
    v_attn_weights_before = model_state_before[f"visual.transformer.resblocks.{layer}.attn.v_proj.vector_S"].squeeze()
    v_attn_weights_after = model_state_after[f"visual.transformer.resblocks.{layer}.attn.v_proj.vector_S"].squeeze()
    output_weights_before = model_state_before[f"visual.transformer.resblocks.{layer}.attn.proj.vector_S"].squeeze()
    output_weights_after = model_state_after[f"visual.transformer.resblocks.{layer}.attn.proj.vector_S"].squeeze()
    
    V_before = v_attn_weights_before.view(num_heads, head_dim)
    V_after = v_attn_weights_after.view(num_heads, head_dim)

    O_before = output_weights_before.view(num_heads, head_dim)
    O_after = output_weights_after.view(num_heads, head_dim)
    
    before_finetune_V[layer] = V_before
    after_finetune_V[layer] = V_after
    before_finetune_O[layer] = O_before
    after_finetune_O[layer] = O_after

singular_value_changes_V = compare_singular_values(before_finetune_V, after_finetune_V)
singular_value_changes_O = compare_singular_values(before_finetune_O, after_finetune_O)

positive_changes_V = {}
negative_changes_V = {}
positive_changes_O = {}
negative_changes_O = {}

for layer, head, change in singular_value_changes_V:
    if change >= 0:
        positive_changes_V[(layer, head)] = positive_changes_V.get((layer, head), 0) + change
    else:
        negative_changes_V[(layer, head)] = negative_changes_V.get((layer, head), 0) + change

for layer, head, change in singular_value_changes_O:
    if change >= 0:
        positive_changes_O[(layer, head)] = positive_changes_O.get((layer, head), 0) + change
    else:
        negative_changes_O[(layer, head)] = negative_changes_O.get((layer, head), 0) + change

sorted_singular_changes_V = sorted(singular_value_changes_V, key=lambda x: abs(x[2]))
sorted_singular_changes_O = sorted(singular_value_changes_O, key=lambda x: abs(x[2]))

print("\nTop Changes in V Projection (Sorted by Absolute Change):")
for layer, head, change in sorted_singular_changes_V:
    print(f"Layer {layer}, Head {head}, Change: {abs(change):.6f}")

print("\nTop Changes in O Projection (Sorted by Absolute Change):")
for layer, head, change in sorted_singular_changes_O:
    print(f"Layer {layer}, Head {head}, Change: {abs(change):.6f}")

combined_absolute_changes = {}
for layer, head, change in singular_value_changes_V:
    combined_absolute_changes[(layer, head)] = abs(change)

for layer, head, change in singular_value_changes_O:
    if (layer, head) in combined_absolute_changes:
        combined_absolute_changes[(layer, head)] += abs(change)
    else:
        combined_absolute_changes[(layer, head)] = abs(change)

sorted_combined_changes = sorted(combined_absolute_changes.items(), key=lambda x: x[1])

head_roles = {}
attention_head_roles = {
    "Layer 8": {
        0: "Foundational Shapes & Silhouettes",
        1: "Motion & Directional Cues",
        2: "Contrast & Light-Dark Regions",
        3: "Geometric Patterns & Grids",
        4: "Object Boundaries & Contours",
        5: "Repeated Structures & Textures",
        6: "Refined Textural Details of Everyday Objects",
        7: "Positional Layout & Spatial Balance",
        8: "Edge Detection & Detail Refinement",
        9: "Primitive Forms & Visual Anchors",
        10: "Coarse-to-Fine Visual Transitions",
        11: "Structural Symmetry & Alignment"
    },
    "Layer 9": {
        0: "Scene Composition & Perspective",
        1: "Depth Perception & Visual Planes",
        2: "Interaction Between Objects",
        3: "Material Surfaces & Reflection",
        4: "Natural Landscapes & Textures",
        5: "Architectural Layouts & Forms",
        6: "Facial Features & Gaze Cues",
        7: "Semantic Grouping of Elements",
        8: "Soft Lighting & Shadow Play",
        9: "Organic Flow & Movement",
        10: "Conceptual Boundaries & Themes",
        11: "Temporal Sequences & Visual Narratives"
    },
    "Layer 10": {
        0: "Aerial Landscapes & Environments",
        1: "Human Experiences & Objects in Action",
        2: "Cultural & Textural Scenes",
        3: "Colorful Festivities & Patterns",
        4: "Cozy Indoors & Historical Locations",
        5: "Action, Emotion & Faces",
        6: "Organic Forms & Flow",
        7: "Geographic & Cultural Diversity",
        8: "Color & Design Aesthetics",
        9: "Sketches, Illustrations & Concepts",
        10: "Mood, Atmosphere & Highlights",
        11: "Human-Centered Realism & Objects"
    },
    "Layer 11": {
        0: "Semantic Layout & Spatial Context",
        1: "Human Relationships & Portraiture",
        2: "Numbers, Symbols & Temporal Cues",
        3: "Lifestyle & Tranquil Activities",
        4: "Visual Symbols & Cultural Motifs",
        5: "Rare Words & Abstract Concepts",
        6: "Global Geographic Locations",
        7: "Color Themes & Tonal Palettes",
        8: "Weather & Natural Forces",
        9: "Everyday Objects & Refined Details",
        10: "Natural Flora, Landscapes & Tranquility",
        11: "Living Creatures & Natural Forms"
    }
}

for layer_name, heads in attention_head_roles.items():
    layer_idx = int(layer_name.split()[-1])
    for head_idx, role in heads.items():
        head_roles[(layer_idx, head_idx)] = role


print("\n=== Top Singular Value Changes (Summed |V| + |O|, Sorted by Total Change) ===")
for (layer, head), total_change in sorted_combined_changes:
    if 8 <= layer <= 11:
        role = head_roles.get((layer, head), "Unknown")
        print(f"Layer {layer}, Head {head}, Total Change: {total_change:.6f}, Role: {role}")