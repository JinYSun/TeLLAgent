# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:38:59 2023

@author: BM109X32G-10GPU-02
"""


 
import numpy as np
from rdkit.Chem import AllChem
 
 
 
import pickle
 
from rdkit import Chem

from sklearn.ensemble import RandomForestRegressor
 

def split_string(string):
    result = []

    for char in string:

        result.append(char)

    return result 
def main(sm):
        
        inchis = list([sm])
        rts = list([0])
        
        smiles, targets,features = [], [],[]
        for i, inc in enumerate((inchis)):
            mol = Chem.MolFromSmiles(inc)
            if mol is None:
                continue
            else:
                smi =AllChem. GetMorganFingerprintAsBitVect(mol,3,2048)
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
        n_features=10
        
        model = RandomForestRegressor(n_estimators=500)

        load_model = pickle.load(open(r"tool\deepacceptor\deepacceptor.pkl","rb"))
        Y_predict = load_model.predict(X_test)
 
        return Y_predict