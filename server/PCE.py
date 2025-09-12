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
from deepacceptor import RF
from deepdonor import sm, pm
from dap import run, screen
from mcp.server.fastmcp import FastMCP 

 
mcp =FastMCP("pce") 
 
@mcp.tool(
    name="acceptor_predictor",          
    description= (  "Input acceptor SMILES , returns the score(PCE) of the acceptor.") 
)
def acceptor_predictor(  smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES string"
    smiles = Chem.MolToSmiles(mol) 
    pce =RF.main( str(smiles) ) 
    return f'The power conversion efficiency (PCE) is predicted to be {pce} (predicted by DeepAcceptor)  '  
 
@mcp.tool(
    name="donor_predictor",          
    description=   "Input DONOR SMILES, returns the score (PCE) of the donor."
) 
def donor_predictor(  smiles:str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES string"
 
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES string"
    sdpce = sm.main( str(smiles) ) 
    pdpce =  pm.main( str(smiles) ) 
    return f'The power conversion efficiency (PCE) of the given molecule is predicted to be {sdpce} as a small molecule donor , and {pdpce} as a polymer donor(predicted by DeepDonor)  '  
  
 
@mcp.tool(
    name="dap_predictor",        
    description=   """Input SMILES in order of acceptor and then donor , 
    returns the performance (PCE)of the D/A pairs .
    Do not get the order wrong"""
)
def dap_predictor(acceptor: str,donor: str) -> float:
   
    pce = run.smiles_aas_test( str(acceptor ), str(donor) )
     
    return pce
        
    
@mcp.tool(
    name="dap_screen",           
    description=  "Input dataset path containing D/A pairs, returns the files of prediction results."
)
def dap_screen(file_path: str) -> str:
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
    
        
 
 
from comget import generator  
@mcp.tool(
    name="donorgen",           # Custom tool name for the LLM
    description=  (
        "Useful to generate  donor molecules with required PCE value. "
        "Input the value, return the SMILES"
    ))
def donorgen( value :float | int) -> str:
    try:
        results = generator.generation(value)
        for i in results['smiles']:
            pdpce =  pm.main( str(i) ) 
            if abs(pdpce-float(value))<1.0:
                return f"The SMILES of generated donor is {i}, its predicted PCE is {pdpce}."
                break
 
    except Exception as e: 
        return str(e)
    

if __name__ =="__main__":
    mcp.run(transport="stdio")       
 

