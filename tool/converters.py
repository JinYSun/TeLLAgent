from langchain.tools import BaseTool

from tool.chemspace import ChemSpace
import pandas as pd

from utils import (
    is_multiple_smiles,
    is_smiles,
    pubchem_query2smiles,
    query2cas,
    smiles2name,
)


class Query2CAS(BaseTool):
    name:str  = "Mol2CAS"
    description:str  = "Input molecule (name or SMILES), returns CAS number."
    url_cid: str = None
    url_data: str = None
    

    def __init__(
        self,
    ):
        super().__init__()
        self.url_cid = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{}/{}/cids/JSON"
        )
        self.url_data = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/JSON"
        )

    def _run(self, query: str) -> str:
        try:
            # if query is smiles
            smiles = None
            if is_smiles(query):
                smiles = query
            try:
                cas = query2cas(query, self.url_cid, self.url_data)
            except ValueError as e:
                return str(e)
            if smiles is None:
                try:
                    smiles = pubchem_query2smiles(cas, None)
                except ValueError as e:
                    return str(e)
 
            return cas
        except ValueError:
            return "CAS number not found"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class Query2SMILES(BaseTool):
    name:str  = "CAS2SMILES"
    description :str = "Input a   CAS number, returns SMILES."
    url: str = None
    chemspace_api_key: str = None
     
    def __init__(self, chemspace_api_key: str = None):
        super().__init__()
        self.chemspace_api_key = chemspace_api_key
        self.url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/{}"

    def _run(self, query: str) -> str:
        """This function queries the given molecule name and returns a SMILES string from the record"""
        """Useful to get the SMILES string of one molecule by searching the name of a molecule. Only query with one specific name."""
        if is_smiles(query) and is_multiple_smiles(query):
            return "Multiple SMILES strings detected, input one molecule at a time."
        try:
            smi = pubchem_query2smiles(query, self.url)
        except Exception as e:
            if self.chemspace_api_key:
                try:
                    chemspace = ChemSpace(self.chemspace_api_key)
                    smi = chemspace.convert_mol_rep(query, "smiles")
                    smi = smi.split(":")[1]
                except Exception:
                    return str(e)
            else:
                try:
 
                    smi = chemspace.convert_mol_rep(query, "smiles")
                    smi = smi.split(":")[1]
                except Exception:
                    return str(e)

      
        return smi

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()

class Mol2SMILES(BaseTool):
    name:str  = "Mol2SMILES"
    description :str = "Input a molecular name , returns SMILES."
  
    def __init__(self, chemspace_api_key: str = None):
        super().__init__()
 
    def _run(self, query: str) -> str:
        """This function queries the given molecule name and returns a SMILES string from the record"""
        """Useful to get the SMILES string of one molecule by searching the name of a molecule. Only query with one specific name."""
        if is_smiles(query) and is_multiple_smiles(query):
            return "Multiple SMILES strings detected, input one molecule at a time."
        try:
            smi = pubchem_query2smiles(query  )
            return smi
        except Exception as e:
            try:
               csv_data = pd.read_csv('tool/dataset.csv',encoding='ISO-8859-1')
               relevant_rows = csv_data[csv_data['Name']==(query)]
               if not relevant_rows.empty:
                   # Get the most relevant answer (assuming we return the first match)
                   return relevant_rows.iloc[0]['SMILES']
            except:    
                return str(e)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()

class SMILES2Name(BaseTool):
    name:str  = "SMILES2Name"
    description:str  = "Input SMILES, returns molecule name."
    
    

    def __init__(self):
        super().__init__()

    def _run(self, query: str) -> str:
        """Use the tool."""
        try:
            if not is_smiles(query):
                try:
                    query2smiles = Query2SMILES()
                    query = query2smiles.run(query)
                except:
                    raise ValueError("Invalid molecule input, no Pubchem entry")
            name = smiles2name(query)
 
            return name
        except Exception as e:
            return "Error: " + str(e)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()
