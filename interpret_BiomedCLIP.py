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
        # Compute difference with normalized singular values
        diff = after[layer] - before[layer]  
        l2_norm_change = torch.norm(diff, p=2, dim=1)  # L2 norm per head
        
        # Retain the sign based on the mean difference
        sign = torch.sign(diff.mean(dim=1))
        signed_l2_change = l2_norm_change * sign
        
        for head in range(len(signed_l2_change)):
            sv_changes.append((layer, head, signed_l2_change[head].item()))
    
    # Sort by absolute L2 norm change magnitude in descending order
    sv_changes.sort(key=lambda x: abs(x[2]))
    return sv_changes
    
# Dictionaries to store singular values before and after fine-tuning
before_finetune_V = {}
after_finetune_V = {}
before_finetune_O = {}
after_finetune_O = {}

clip_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
clip_model = clip_model.cuda()
logit_scale = clip_model.logit_scale.exp().detach()
for i,module in enumerate(clip_model.visual.trunk.blocks):
    new_module = BiomedCLIPMultiHeadAttention()
    new_module.set_parameters(module.attn)
    module.attn = new_module

clip_model = resolver(clip_model.cuda().float())
clip_model = clip_model.cuda() 

model_state_before = clip_model.state_dict()

load_path = "outputs_tsne/eurosat/vitb16/16shots/seed2/base_model.pt"

loaded_data = torch.load(args.load_path)

metadata = loaded_data['metadata']
weights = loaded_data['weights']
model_state_after = weights

num_layers = 12
num_heads = 12
head_dim = 64

for layer in range(num_layers):
    v_attn_weights_before = model_state_before[f"visual.trunk.blocks.{layer}.attn.v_proj.vector_S"].squeeze()
    v_attn_weights_after = model_state_after[f"visual.trunk.blocks.{layer}.attn.v_proj.vector_S"].squeeze()
    output_weights_before = model_state_before[f"visual.trunk.blocks.{layer}.attn.proj.vector_S"].squeeze()
    output_weights_after = model_state_after[f"visual.trunk.blocks.{layer}.attn.proj.vector_S"].squeeze()
    
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
        0: "Focal Markers & Shape Cues",
        1: "Peripheral Edges & Enhancement Rings",
        2: "Lobe Collapse & Serpentine Patterns",
        3: "Radiologic Artifacts & Diffuse Shapes",
        4: "Flow Voids & Vascular Boundaries",
        5: "Textural Shifts & Patchy Regions",
        6: "Contour Irregularity & Internal Spread",
        7: "Symmetry Deviation & Dense Regions",
        8: "Clustered Signals & Midline Shifts",
        9: "Scattered Highlights & Artifactual Spots",
        10: "Diffuse Zones & Overlapping Shapes",
        11: "Dense Regions & Local Signal Buildup"
    },
    "Layer 9": {
        0: "Symmetry Shifts & Linear Features",
        1: "Ring-Like Structures & Localized Spread",
        2: "Irregular Densities & Textured Borders",
        3: "Small Patterned Anomalies",
        4: "Defined Edges & Subpleural Zones",
        5: "Streaks, Texture, & Soft Borders",
        6: "Sharp Edges & Tissue Mismatch",
        7: "Thickened Zones & Star-Like Forms",
        8: "Homogeneous Textures & Clustered Lines",
        9: "Contour Disruptions & Shape Markers",
        10: "Circumscribed Lesions & Flow Paths",
        11: "Signal Artifacts & Internal Septations"
    },
    "Layer 10": {
        0: "Low-Contrast Regions & Collapse Patterns",
        1: "Localized Irregularities & Shadowing",
        2: "Contrast Rings & Interface Blur",
        3: "Multi-Compartment Spread & Fine Edges",
        4: "Fused Regions & Curved Forms",
        5: "Sharp Transitions & Highlighted Foci",
        6: "Circular Opacities & Clean Boundaries",
        7: "Textured Rings & Septated Centers",
        8: "Artifact-Like Patterns & Symmetry Shift",
        9: "Smooth Outlines & Diffuse Flow",
        10: "Star Patterns & Linear Disruptions",
        11: "Texture Aggregates & Uniform Zones"
    },
    "Layer 11": {
        0: "Signal Voids & Shifts",
        1: "Blurred Regions & Artifact Shapes",
        2: "Peripheral Contrast & Mosaic Textures",
        3: "Cross-Lobe Flow & Density Buildup",
        4: "Sharp-Edged Shapes & Contrast Pockets",
        5: "Ringed Zones & Pattern Complexity",
        6: "Rounded Opacities & Midline Distortions",
        7: "Compact Signal Shifts & Grid-Like Forms",
        8: "Converging Edges & Cluster Markers",
        9: "Scattered Intensity & Symmetry Cues",
        10: "Rimmed Signals & Flow Distributions",
        11: "Sharp Vascular Lines & Compact Zones"
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