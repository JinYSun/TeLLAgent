# -*- coding: utf-8 -*-
import pandas as pd 
 
import math
from tqdm import tqdm
import argparse
from .model import GPT, GPTConfig
import torch
import numpy as np 
import re
import json
from rdkit.Chem import RDConfig
from torch.nn import functional as F
import selfies as sf
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
from rdkit import Chem
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def sample(model, x, steps, temperature=1.0, sample=False, top_k=None, prop = None, scaffold = None):
     """
     take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
     the sequence, feeding the predictions back into the model each time. Clearly the sampling
     has quadratic complexity unlike an RNN that is only linear, and has a finite context window
     of block_size, unlike an RNN that has an infinite context window.
     """
     block_size = model.get_block_size()   
     model.eval()

     for k in range(steps):
         x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
         logits, _, _ = model(x_cond, prop = prop, scaffold = scaffold)   # for liggpt
         # logits, _, _ = model(x_cond)   # for char_rnn
         # pluck the logits at the final step and scale by temperature
         logits = logits[:, -1, :] / temperature
         # optionally crop probabilities to only the top k options
         if top_k is not None:
             logits = top_k_logits(logits, top_k)
         # apply softmax to convert to probabilities
         probs = F.softmax(logits, dim=-1)
         # sample from the distribution or take the most likely
         if sample:
             ix = torch.multinomial(probs, num_samples=1)
         else:
             _, ix = torch.topk(probs, k=1, dim=-1)
         # append to the sequence and continue
         x = torch.cat((x, ix), dim=1)

     return x
def get_selfie_and_smiles_encodings_for_dataset(smiles):
            """
            Returns encoding, alphabet and length of largest molecule in SMILES and
            SELFIES, given a file containing SMILES molecules.
        
            input:
                csv file with molecules. Column's name must be 'smiles'.
            output:
                - selfies encoding
                - selfies alphabet
                - longest selfies string
                - smiles encoding (equivalent to file content)
                - smiles alphabet (character based)
                - longest smiles string
            """
       
            smiles_list = np.asanyarray(smiles)
        
            smiles_alphabet = list(set("".join(smiles_list)))
            smiles_alphabet.append(" ")  # for padding
        
            largest_smiles_len = len(max(smiles_list, key=len))
        
            print("--> Translating SMILES to SELFIES...")
            selfies_list = list(map(sf.encoder, smiles_list))
        
            all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)
            all_selfies_symbols.add("[nop]")
            selfies_alphabet = list(all_selfies_symbols)
        
            largest_selfies_len = max(sf.len_selfies(s) for s in selfies_list)
        
            print("Finished translating SMILES to SELFIES.")
        
            return selfies_list, selfies_alphabet, largest_selfies_len, \
                   smiles_list, smiles_alphabet, largest_smiles_len
                   
def generation(value):
 
   
    scaffold = False
    lstm = False
    data_name = 'ppcenos'
    batch_size = 5
    gen_size = 20
    vocab_size = 29
    block_size = 196
    n_layer = 8
    n_head = 8
    n_embd = 256
    lstm_layers = 2
    props = ['pce']
    csv_name = 'ppcenos'
    
 
    context = "[C]"
    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    
   
    if 'moses' in data_name and scaffold:
        scaffold_max_len = 48
    elif 'guacamol' in data_name:
        scaffold_max_len = 107
    else:
        scaffold_max_len = 181
    
 
    stoi = json.load(open('tool/comget/' + f'{data_name}.json', 'r'))
    itos = {i: ch for ch, i in stoi.items()}
    
    print(len(itos))
     
    num_props = len(props)
    mconf = GPTConfig(vocab_size, block_size, num_props=num_props,
                      n_layer=n_layer, n_head=n_head, n_embd=n_embd, 
                      scaffold=scaffold, scaffold_maxlen=scaffold_max_len,
                      lstm=lstm, lstm_layers=lstm_layers)
    model = GPT(mconf)
    
    # 加载模型权重
    model_weight = f'{csv_name}.pt'
    model.load_state_dict(torch.load('tool/comget/' + model_weight))
    model.to('cuda')
    print('Model loaded')
    
     
    gen_iter = math.ceil(gen_size / batch_size)
    
    
    prop2value = {'pce': [float(value)]}  # 需要定义value的值，这里保持原逻辑但需要修正
    prop_condition = None
    if len(props) > 0:
        prop_condition = prop2value['_'.join(props)]
        
    scaf_condition = None
    
    all_dfs = []
    all_metrics = []
    count = 0
    
    if prop_condition is not None and scaf_condition is None:
        for c in prop_condition:
            molecules = []
            selfies = []
            count += 1
            for i in tqdm(range(gen_iter)):
                x = torch.tensor([stoi[s] for s in regex.findall(context)], 
                                dtype=torch.long)[None, ...].repeat(batch_size, 1).to('cuda')
                p = None
                if len(props) == 1:
                    p = torch.tensor([c]).repeat(batch_size, 1).to('cuda')
                else:
                    p = torch.tensor([c]).repeat(batch_size, 1).unsqueeze(1).to('cuda')
                sca = None
                y = sample(model, x, 300, temperature=1.0, sample=True, top_k=10, prop=p, scaffold=sca)
                for gen_mol in y:
                    completion = ''.join([itos[int(i)] for i in gen_mol])
                    completion = completion.replace('<', '')
                    selfies.append(completion)
            file = pd.DataFrame(selfies)
    
            for ind, i in enumerate(file[0]):
                smi = sf.decoder(eval(repr(i)))
                mol = get_mol(smi)
                if mol:
                    molecules.append(mol)
                else:
                    print(ind)
                    print(i)
        
            print("Valid molecules % = {}".format(len(molecules)))
        
            mol_dict = []
            for i in molecules:
                mol_dict.append({'molecule': i, 'smiles': Chem.MolToSmiles(i)})
    
            results = pd.DataFrame(mol_dict)
            all_dfs.append(results)
    
    results = pd.concat(all_dfs)
    return results