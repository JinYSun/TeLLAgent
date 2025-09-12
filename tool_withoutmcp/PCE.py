# -*- coding: utf-8 -*-

"""
Created on Wed Sep 11 10:27:20 2024

@author: BM109X32G-10GPU-02
"""
import pandas as pd
from langchain.tools import BaseTool
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from .deepacceptor import RF
from .deepdonor import sm, pm
from .dap import run, screen

class acceptor_predictor(BaseTool):
    name:str  = "acceptor_predictor"
    description:str  = (
        "Input acceptor SMILES , returns the score of the acceptor."
    )
    
    def __init__(self):
        super().__init__()
    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        smiles = Chem.MolToSmiles(mol) 
        pce = RF.main( str(smiles) ) 
        return f'The power conversion efficiency (PCE) is predicted to be {pce} (predicted by DeepAcceptor)  '  
 
    async def _arun(self, smiles: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()
        
class donor_predictor(BaseTool):
    name:str  = "donor_predictor"
    description:str  = (
        "Input donor SMILES , returns the score of the donor."
    )
    
    def __init__(self):
        super().__init__()
    def _run(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
     
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        sdpce = sm.main( str(smiles) ) 
        pdpce =  pm.main( str(smiles) ) 
        return f'The power conversion efficiency (PCE) of the given molecule is predicted to be {sdpce} as a small molecule donor , and {pdpce} as a polymer donor(predicted by DeepDonor)  '  
  
    async def _arun(self, smiles: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()
        
 

class dap_predictor(BaseTool):
    name:str  = "dap_predictor"
    description :str = (
        "Input SMILES of D/A pairs in order of acceptor and then donor(separated by '.') , returns the performance of the D/A pairs ."
    )


    def __init__(self):
        super().__init__()

    def _run(self, smiles_pair: str) -> str:
        smi_list = smiles_pair.split(".")
        if len(smi_list) != 2:
 
            return "Input error, please input two smiles strings separated by '.'"  
            
        else:
            smiles1, smiles2 = smi_list

         
        pce = run.smiles_aas_test( str(smiles1 ), str(smiles2) )
         
        return pce

    async def _arun(self, smiles_pair: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()
        
    

class dap_screen(BaseTool):
    name:str  = "dap_screen"
    description :str = (
        "Input dataset path containing D/A pairs, returns the files of prediction results."
    )

    def __init__(self):
        super().__init__()

    def _run(self, file_path: str) -> str:
        smi_list = screen.smiles_aas_test(file_path)
        smi_list = pd.DataFrame(smi_list)
        smi_list.to_csv('screen_results.csv')
        pd.set_option('display.max_rows', None) 
        pd.set_option('display.max_columns', None) 
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None) 
        Three = smi_list.nlargest(2,'predict')
        Three = Three.reset_index()
        if len(set(Three['acceptor'])) == 1:
            return f"""The screening results is available at screen_results.csv, the potential donors are {Three['donor'][0]} and {Three['donor'][1]}"""
        elif len(set(Three['donor'])) == 1:
            return f"""The screening results is available at screen_results.csv, the potential acceptors are {Three['acceptor'][0] } and {Three['acceptor'][1]}"""
        else:
            return f"""The screening results is available at screen_results.csv, the potential D/A pairs are {list (Three.iloc[0,[-3,-2,-1]])} and {list (Three.iloc[1,[-3,-2,-1]])}"""
        
        

    async def _arun(self, smiles_pair: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()
        
 
from .comget import generator  
 
class molgen(BaseTool):
    name: str = "donorgen"
    description: str = (

        "Useful to generate polymer donor molecules with required PCE value. "
        "Input the values of PCE , return the SMILES"
    )
 
    
    def __init__(self
                 ):
        super().__init__(  )
 

    def _run(self, value: float|int ) -> str:
        try:
            results = generator.generation(value)
            for i in results['smiles']:
                pdpce =  pm.main( str(i) ) 
                if abs(pdpce-float(value))<1.0:
                    return f"The SMILES of generated donor is {i}, its predicted PCE is {pdpce}."
                    break
                
            
        
        except Exception as e: 
            return str(e)
        

    async def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("this tool does not support async")
