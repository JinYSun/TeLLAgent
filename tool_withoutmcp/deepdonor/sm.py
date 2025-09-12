# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:38:59 2023

@author: BM109X32G-10GPU-02
"""



import numpy as np
from rdkit.Chem import AllChem
 
 
import pickle
 
from tqdm import tqdm

 
from rdkit import Chem

from sklearn.ensemble import RandomForestRegressor
 
def split_dataset(dataset, ratio):
    """Shuffle and split a dataset."""
   # np.random.seed(111)  # fix the seed for shuffle.
    #np.random.shuffle(dataset)
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]

def split_string(string):
    
    result = []
   
    for char in string:
      
        result.append(char)
 
    return result
 
def main(sm):
 
        inchis = list([sm])
        rts = list([0])
        
        smiles, targets,features = [], [],[]
        for i, inc in enumerate(tqdm(inchis)):
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
        
        model = RandomForestRegressor(n_estimators=100)
        load_model = pickle.load(open(r"tool/deepdonor/sm.pkl", 'rb'))

     #   model = load_model('C:/Users/sunjinyu/Desktop/FingerID Reference/drug-likeness/CNN/single_model.h5')
        Y_predict = load_model.predict((X_test))
         #Y_predict = model.predict(X_test) 
        x = list(Y_test)
        y = list(Y_predict)
       
        return Y_predict
      