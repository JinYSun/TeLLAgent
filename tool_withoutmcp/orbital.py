# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 09:14:55 2024

@author: BM109X32G-10GPU-02
"""


from sklearn.metrics import confusion_matrix
 
import numpy as np
from rdkit.Chem import AllChem
from sklearn.datasets import make_blobs
import json
import numpy as np
import math
 
from scipy import sparse
from sklearn.metrics import median_absolute_error,r2_score, mean_absolute_error,mean_squared_error
from langchain.tools import BaseTool
import pandas as pd
 
from rdkit import Chem
import pickle
from sklearn.ensemble import GradientBoostingRegressor
 

def split_string(string):
    
    result = []
   
    for char in string:
      
        result.append(char)
 
    return result
 
def main(sm):
   
 
        inchis = list([sm])
        rts = list([0])
        
        smiles, targets,features = [], [],[]
        for i, inc in enumerate(inchis):
            mol = Chem.MolFromSmiles(inc)
            if mol is None:
                continue
            else:
                smi =AllChem. GetMorganFingerprintAsBitVect(mol,1024)
                smi = smi.ToBitString()
                a = split_string(smi)
                a = np.array(a)
                #smi = Chem.MolToSmiles(mol)
                features.append(a)
                targets.append(rts[i])
                
       

        features = np.asarray(features)
        targets = np.asarray(targets)
        X_test=features
        Y_test=targets
 
        model = GradientBoostingRegressor()   
        load_homo = pickle.load(open(r"tool/orbital/homo.dat", 'rb'))
        load_lumo = pickle.load(open(r"tool/orbital/lumo.dat",'rb'))
     #   model = load_model('C:/Users/sunjinyu/Desktop/FingerID Reference/drug-likeness/CNN/single_model.h5')
        Y_homo= load_homo.predict(X_test)
        Y_lumo = load_lumo.predict(X_test)
        homo =  float(Y_homo)
        lumo =  float(Y_lumo)
        return homo, lumo

class homolumo_predictor(BaseTool):
    name: str = "homolumo_predictor"
    description: str = (
        "Input SMILES , returns the HOMO/LUMO  (Highest Occupied Molecular Orbital (HOMO) \
        and  Lowest Unoccupied Molecular Orbital)."
    )
    def __init__(self):
        super().__init__()
    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        Y_homo, Y_lumo = main( str(smiles) ) 
        return f"The HOMO is predicted to be {'{:.2f}'.format(Y_homo)} eV , the LUMO is predicted to be {'{:.2f}'.format(Y_lumo)}  eV" 
 
    async def _arun(self, smiles: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()
          