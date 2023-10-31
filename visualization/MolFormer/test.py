from bertviz import head_view
from get_attention_map_full import get_full_attention

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms

import torch

#%%
sequences = ["CC1(C)C(C)(O)C1(C)O", "CC(O)C(C)(O)C(N)=O", "CC(C)C(C)(C)O"] # gdb_62509, gdb_58097, gdb_1105

# CHANGE THIS SEQ_IDX To 0,1,2 TO ANALYSIS FOR RESPECTIVE SEQUENCE
seq_idx = 0
sequence = sequences[seq_idx]

attentions, tokens = get_full_attention(sequence)

print(attentions)