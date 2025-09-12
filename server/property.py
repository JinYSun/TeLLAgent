# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 21:42:51 2024

@author: BM109X32G-10GPU-02
"""
import asyncio
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
 
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
import os
from rdkit.Chem import RDConfig
from rdkit.ML.Descriptors import MoleculeDescriptors
from mcp.server.fastmcp import FastMCP 
from rdkit.Contrib.SA_Score import sascorer
from dotenv import load_dotenv
load_dotenv()

def tanimoto(s1, s2):
        """Calculate the Tanimoto similarity of two SMILES strings."""
        try:
            mol1 = Chem.MolFromSmiles(s1)
            mol2 = Chem.MolFromSmiles(s2)
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except (TypeError, ValueError, AttributeError):
            return "Error: Not a valid SMILES string"
mcp =FastMCP("property") 

@mcp.tool(
    name="MolSimilarity",           # Custom tool name for the LLM
    description="Input two molecule SMILES , returns Tanimoto similarity.", # Custom description
)
async  def MolSimilarity(  smiles1: str, smiles2: str,) -> str:
 

    similarity = tanimoto(smiles1, smiles2)

    if isinstance(similarity, str):
        return similarity

    if similarity == 1:
        return "Error: Input Molecules Are Identical"
    else:
       
        message = f"The Tanimoto similarity between {smiles1} and {smiles2} is {round(similarity, 4)}"
    return message

 
@mcp.tool(
    name="SMILES2Weight",           # Custom tool name for the LLM
    description="Input SMILES, returns molecular weight.", # Custom description
)
async def SMILES2Weight( smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES string"
    mol_weight = rdMolDescriptors.CalcExactMolWt(mol)
    return mol_weight

 
@mcp.tool(
    name="SMILES2LogP",           # Custom tool name for the LLM
    description= "Input SMILES, returns molecular LogP." # Custom description
)
async def SMILES2LogP( smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES string"
    MolLogP = Descriptors.MolLogP(mol)
    return MolLogP

 
@mcp.tool(
    name="SMILES2SAScore",           # Custom tool name for the LLM
    description= "Input SMILES, returns synthetic accessibility score to evaluate the difficulty of molecular synthesis." # Custom description
)        
async def SMILES2SAScore( smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES string"
    SAScore = sascorer.calculateScore(mol)
    return f"This SAScore of the molecule is {SAScore}."
@mcp.tool(
    name="SMILES2Properties",           # Custom tool name for the LLM
    description= "Input SMILES, returns basic physical and chemical properties."
) 
async def SMILES2Properties( smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES string"
    SAScore = sascorer.calculateScore(mol)
    des_list = ['MolWt','NOCount', 'NumHAcceptors', 'NumHDonors', 'MolLogP', 'NumRotatableBonds','RingCount','NumAromaticRings','TPSA']
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)
    results = calculator.CalcDescriptors(mol)
    return f"SAScore: {'{:.2f}'.format(SAScore)}; molecular weight: {'{:.2f}'.format(results[0])}; number of Nitrogens and Oxygens: {results[1]}; number of Hydrogen Bond Acceptors: {results[2]}; number of Hydrogen Bond Donors:{results[3]}; LogP:{'{:.2f}'.format(results[4])}; number of Rotatable Bonds: {results[5]}; Ring count: {results[6]}; number of aromatic rings: {results[7]}; TPSA: {'{:.2f}'.format(results[8])}."
 
    
def _is_fg_in_mol( mol, fg):
        fgmol = Chem.MolFromSmarts(fg)
        mol = Chem.MolFromSmiles(mol.strip())
        return len(Chem.Mol.GetSubstructMatches(mol, fgmol, uniquify=True)) > 0
    
@mcp.tool(
    name="FuncGroups",           # Custom tool name for the LLM
    description= "Input SMILES, return list of functional groups and substructures in the molecule."
)     
async def FuncGroups(smiles: str) -> str:
    """
    Input a molecule SMILES or name.
    Returns a list of functional groups identified by their common name (in natural language).
    """
    dict_fgs = {
        "furan": "o1cccc1",
        "aldehydes": " [CX3H1](=O)[#6]",
        "esters": " [#6][CX3](=O)[OX2H0][#6]",
        "ketones": " [#6][CX3](=O)[#6]",
        "amides": " C(=O)-N",
        "thiol groups": " [SH]",
        "alcohol groups": " [OH]",
        "methylamide": "*-[N;D2]-[C;D3](=O)-[C;D1;H3]",
        "carboxylic acids": "*-C(=O)[O;D1]",
        "carbonyl methylester": "*-C(=O)[O;D2]-[C;D1;H3]",
        "terminal aldehyde": "*-C(=O)-[C;D1]",
        "amide": "*-C(=O)-[N;D1]",
        "carbonyl methyl": "*-C(=O)-[C;D1;H3]",
        "isocyanate": "*-[N;D2]=[C;D2]=[O;D1]",
        "isothiocyanate": "*-[N;D2]=[C;D2]=[S;D1]",
        "nitro": "*-[N;D3](=[O;D1])[O;D1]",
        "nitroso": "*-[N;R0]=[O;D1]",
        "oximes": "*=[N;R0]-[O;D1]",
        "Imines": "*-[N;R0]=[C;D1;H2]",
        "terminal azo": "*-[N;D2]=[N;D2]-[C;D1;H3]",
        "hydrazines": "*-[N;D2]=[N;D1]",
        "diazo": "*-[N;D2]#[N;D1]",
        "cyano": "*-[C;D2]#[N;D1]",
        "primary sulfonamide": "*-[S;D4](=[O;D1])(=[O;D1])-[N;D1]",
        "methyl sulfonamide": "*-[N;D2]-[S;D4](=[O;D1])(=[O;D1])-[C;D1;H3]",
        "sulfonic acid": "*-[S;D4](=O)(=O)-[O;D1]",
        "methyl ester sulfonyl": "*-[S;D4](=O)(=O)-[O;D2]-[C;D1;H3]",
        "methyl sulfonyl": "*-[S;D4](=O)(=O)-[C;D1;H3]",
        "sulfonyl chloride": "*-[S;D4](=O)(=O)-[Cl]",
        "methyl sulfinyl": "*-[S;D3](=O)-[C;D1]",
        "methyl thio": "*-[S;D2]-[C;D1;H3]",
        "thiols": "*-[S;D1]",
        "thio carbonyls": "*=[S;D1]",
        "halogens": "*-[#9,#17,#35,#53]",
        "t-butyl": "*-[C;D4]([C;D1])([C;D1])-[C;D1]",
        "tri fluoromethyl": "*-[C;D4](F)(F)F",
        "acetylenes": "*-[C;D2]#[C;D1;H]",
        "cyclopropyl": "*-[C;D3]1-[C;D2]-[C;D2]1",
        "ethoxy": "*-[O;D2]-[C;D2]-[C;D1;H3]",
        "methoxy": "*-[O;D2]-[C;D1;H3]",
        "side-chain hydroxyls": "*-[O;D1]",
        "ketones": "*=[O;D1]",
        "primary amines": "*-[N;D1]",
        "nitriles": "*#[N;D1]",
        "Amide": "[NX3][CX3](=[OX1])[#6]",
        "Amino acid side chains": "[CH3X4]",
        "Any carbon attached to any halogen": "[#6][F,Cl,Br,I]",
        "Aromatic 5-Ring O with Lone Pair": "[oX2r5]",
        "Azole": "[$([nr5]:[nr5,or5,sr5]),$([nr5]:[cr5]:[nr5,or5,sr5])]",
        "Carbonyl group": "[$([CX3]=[OX1]),$([CX3+]-[OX1-])]",
        "Carbonyl with Carbon": "[CX3](=[OX1])C",
        "Carbonyl with Nitrogen": "[OX1]=CN",
        "Carbonyl with Oxygen": "[CX3](=[OX1])O",
        "Carboxylic acid or conjugate base": "[CX3](=O)[OX1H0-,OX2H1]",
        "Carboxylic acid": "[CX3](=O)[OX2H1]",
        "Dicarboxdiimide": "[CX3](=[OX1])[NX3H0]([NX3H0]([CX3](=[OX1]))[CX3](=[OX1]))[CX3](=[OX1])",
        "Enamine or Aniline Nitrogen": "[NX3][$(C=C),$(cc)]",
        "Enamine": "[NX3][CX3]=[CX3]",
        "Enol or Phenol": "[OX2H][$(C=C),$(cc)]",
        "Ester Also hits anhydrides": "[#6][CX3](=O)[OX2H0][#6]",
        "Ethenyl carbon": "[$([CX2]#C)]",
        "Ether": "[OD2]([#6])[#6]",
        "Halogen": "[F,Cl,Br,I]",
        "Hydrazine H2NNH2": "[NX3][NX3]",
        "Hydrazone C=NNH2": "[NX3][NX2]=[*]",
        "Hydroxyl (includes alcohol, phenol)": "[OX2H]",
        "Hydroxyl in Alcohol": "[#6][OX2H]",
        "Hydroxyl in Carboxylic Acid": "[OX2H][CX3]=[OX1]",
        "Hydroxyl_acidic": "[$([OH]-*=[!#6])]",
        "Isoleucine side chain": "[CHX4]([CH3X4])[CH2X4][CH3X4]",
        "Ketone": "[#6][CX3](=O)[#6]",
        "Leucine side chain": "[CH2X4][CHX4]([CH3X4])[CH3X4]",
        "Lysine side chain": "[CH2X4][CH2X4][CH2X4][CH2X4][NX4+,NX3+0]",
        "Mono-sulfide": "[#16X2H0][!#16]",
        "Multiple non-fused benzene rings": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1.[cR1]1[cR1][cR1][cR1][cR1][cR1]1",
        "N in 5-sided aromatic ring": "[nX2r5]",
        "Nitrile": "[NX1]#[CX2]",
        "Nitro group": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
        "Nitroso-group": "[NX2]=[OX1]",
        "Peroxide groups": "[OX2,OX1-][OX2,OX1-]",
        "Phenol": "[OX2H][cX3]:[c]",
        "Phenylalanine side chain": "[CH2X4][cX3]1[cX3H][cX3H][cX3H][cX3H][cX3H]1",
        "Primary amine, not amide": "[NX3;H2;!$(NC=[!#6]);!$(NC#[!#6])][#6]",
        "S in aromatic 5-ring with lone pair": "[sX2r5]",
        "Schiff base Substituted or un-substituted imine": "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]",
        "Serine side chain": "[CH2X4][OX2H]",
        "Spiro-ring center": "[X4;R2;r4,r5,r6](@[r4,r5,r6])(@[r4,r5,r6])(@[r4,r5,r6])@[r4,r5,r6]",
        "Substituted dicarboximide": "[CX3](=[OX1])[NX3H0]([#6])[CX3](=[OX1])",
        "Substituted imine": "[CX3;$([C]([#6])[#6]),$([CH][#6])]=[NX2][#6]",
        "Sulfide": "[#16X2H0]",
        "Sulfone. Low specificity.": "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",
        "Sulfoxide High specificity": "[$([#16X3](=[OX1])([#6])[#6]),$([#16X3+]([OX1-])([#6])[#6])]",
        "Sulfur with at-least one hydrogen": "[#16!H0]",
        "Thio analog of carbonyl": "[#6X3](=[SX1])([!N])[!N]",
        "Thioamide": "[NX3][CX3]=[SX1]",
        "Three_halides groups": "[F,Cl,Br,I].[F,Cl,Br,I].[F,Cl,Br,I]",
        "Two Nitro groups": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8].[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
        "Two primary or secondary amines": "[NX3;H2,H1;!$(NC=O)].[NX3;H2,H1;!$(NC=O)]",
        "Two Sulfides": "[#16X2H0][!#16].[#16X2H0][!#16]",
        "Unfused benzene ring": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1",
        "Valine side chain": "[CHX4]([CH3X4])[CH3X4]",
        "Vinylic Carbon": "[$([CX3]=[CX3])]"}
    try:
        fgs_in_molec = [
            name
            for name, fg in dict_fgs.items()
            if _is_fg_in_mol(smiles, fg)
        ]
        if len(fgs_in_molec) > 1:
            return f"This molecule contains {', '.join(fgs_in_molec[:-1])}, and {fgs_in_molec[-1]}."
        else:
            return f"This molecule contains {fgs_in_molec[0]}."
    except:
        return "Wrong argument. Please input a valid molecular SMILES."

 
        
if __name__ =="__main__":
    mcp.run(transport="stdio")