 
import os
import pandas as pd
import requests 
import pandas as pd
from mcp.server.fastmcp import FastMCP 
from DECIMER import predict_SMILES
 
def load_api_keys(file_path='api.txt'):
     
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):  
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
    
                    if value == 'None':
                        continue
                    
                    os.environ[key] = value
                    print(f" {key}")
  
load_api_keys("api.txt")
import re

import requests
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from urllib.request import urlopen
from urllib.parse import quote

def is_smiles(text):
    try:
        m = Chem.MolFromSmiles(text, sanitize=False)
        if m is None:
            return False
        return True
    except:
        return False


def is_multiple_smiles(text):
    if is_smiles(text):
        return "." in text
    return False


def split_smiles(text):
    return text.split(".")


def is_cas(text):
    pattern = r"^\d{2,7}-\d{2}-\d$"
    return re.match(pattern, text) is not None


def largest_mol(smiles):
    ss = smiles.split(".")
    ss.sort(key=lambda a: len(a))
    while not is_smiles(ss[-1]):
        rm = ss[-1]
        ss.remove(rm)
    return ss[-1]


def canonical_smiles(smiles):
    try:
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
        return smi
    except Exception:
        return "Invalid SMILES string"


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

def CIRconvert(ids):
   
    url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(ids) + '/smiles'
    ans = urlopen(url).read().decode('utf8')
    return ans
    
    
    
def pubchem_query2smiles(
    query: str,
    url: str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/{}",
) -> str:
    if is_smiles(query):
        if not is_multiple_smiles(query):
            return query
        else:
            raise ValueError(
                "Multiple SMILES strings detected, input one molecule at a time."
            )
    if url is None:
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/{}"
    r = requests.get(url.format(query, "property/IsomericSMILES/JSON"))
    # convert the response to a json object
    data = r.json()
    # return the SMILES string
    try:
        smi = data["PropertyTable"]["Properties"][0]["SMILES"]
    except: 
        try: 
            smi = CIRconvert(query)
    
        except KeyError:
             return "Could not find a molecule matching the text. One possible cause is that the input is incorrect, input one molecule at a time."
    return str(Chem.CanonSmiles(largest_mol(smi)))


def query2cas(query: str, url_cid: str, url_data: str):
    try:
        mode = "name"
        if is_smiles(query):
            if is_multiple_smiles(query):
                raise ValueError(
                    "Multiple SMILES strings detected, input one molecule at a time."
                )
            mode = "smiles"
        url_cid = url_cid.format(mode, query)
        cid = requests.get(url_cid).json()["IdentifierList"]["CID"][0]
        url_data = url_data.format(cid)
        data = requests.get(url_data).json()
    except (requests.exceptions.RequestException, KeyError):
        raise ValueError("Invalid molecule input, no Pubchem entry")

    try:
        for section in data["Record"]["Section"]:
            if section.get("TOCHeading") == "Names and Identifiers":
                for subsection in section["Section"]:
                    if subsection.get("TOCHeading") == "Other Identifiers":
                        for subsubsection in subsection["Section"]:
                            if subsubsection.get("TOCHeading") == "CAS":
                                return subsubsection["Information"][0]["Value"][
                                    "StringWithMarkup"
                                ][0]["String"]
    except KeyError:
        raise ValueError("Invalid molecule input, no Pubchem entry")

    raise ValueError("CAS number not found")


def smiles2name(smi, single_name=True):
    """This function queries the given molecule smiles and returns a name record or iupac"""

    try:
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)
    except Exception:
        raise ValueError("Invalid SMILES string")
    # query the PubChem database
    r = requests.get(
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
        + smi
        + "/synonyms/JSON"
    )
    # convert the response to a json object
    data = r.json()
    # return the SMILES string
    try:
        if single_name:
            index = 0
            names = data["InformationList"]["Information"][0]["Synonym"]
            while is_cas(name := names[index]):
                index += 1
                if index == len(names):
                    raise ValueError("No name found")
        else:
            name = data["InformationList"]["Information"][0]["Synonym"]
    except KeyError:
        raise ValueError("Unknown Molecule")
    return name

class ChemSpace:
    def __init__(self, chemspace_api_key=None):
        self.chemspace_api_key = chemspace_api_key
        self._renew_token()  # Create token

    def _renew_token(self):
        self.chemspace_token = requests.get(
            url="https://api.chem-space.com/auth/token",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.chemspace_api_key}",
            },
        ).json()["access_token"]
            
    def _make_api_request(
        self,
        query,
        request_type,
        count,
        categories,
    ):
        """
        Make a generic request to chem-space API.

        Categories request.
            CSCS: Custom Request: Could be useful for requesting whole synthesis
            CSMB: Make-On-Demand Building Blocks
            CSSB: In-Stock Building Blocks
            CSSS: In-stock Screening Compounds
            CSMS: Make-On-Demand Screening Compounds
        """

        def _do_request():
            data = requests.request(
                "POST",
                url=f"https://api.chem-space.com/v3/search/{request_type}?count={count}&page=1&categories={categories}",
                headers={
                    "Accept": "application/json; version=3.1",
                    "Authorization": f"Bearer {self.chemspace_token}",
                },
                data={"SMILES": f"{query}"},
            ).json()
            return data

        data = _do_request()

        # renew token if token is invalid
        if "message" in data.keys():
            if data["message"] == "Your request was made with invalid credentials.":
                self._renew_token()

        data = _do_request()
        return data

    def _convert_single(self, query, search_type: str):
        """Do query for a single molecule"""
        data = self._make_api_request(query, "exact", 1, "CSCS,CSMB,CSSB")
        if data["count"] > 0:
            return data["items"][0][search_type]
        else:
            return "No data was found for this compound."

    def convert_mol_rep(self, query, search_type: str = "smiles"):
        if ", " in query:
            query_list = query.split(", ")
        else:
            query_list = [query]
        smi = ""
        try:
            for q in query_list:
                smi += f"{query}'s {search_type} is: {str(self._convert_single(q, search_type))}"
                return smi
        except Exception:
            return "The input provided is wrong. Input either a single molecule, or multiple molecules separated by a ', '"

  
mcp =FastMCP("converters") 
@mcp.tool(
    name="Query2CAS",           # Custom tool name for the LLM
    description=  (
        "Input molecule (name or SMILES), returns CAS number."
    ))  
async def Query2CAS( query: str) -> str:
    url_cid =  "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{}/{}/cids/JSON"
    url_data = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/JSON"    
    try:
        # if query is smiles
        smiles = None
        if is_smiles(query):
            smiles = query
        try:
            cas = query2cas(query, url_cid, url_data)
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

@mcp.tool(
    name="CAS2SMILES",           # Custom tool name for the LLM
    description=  (
        "Input a CAS number, returns SMILES."
    )) 
async def CAS2SMILES( query: str) -> str:
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/{}"
    chemspace_api_key = None 
    
    """This function queries the given molecule name and returns a SMILES string from the record"""
    """Useful to get the SMILES string of one molecule by searching the name of a molecule. Only query with one specific name."""
    if is_smiles(query) and is_multiple_smiles(query):
        return "Multiple SMILES strings detected, input one molecule at a time."
    try:
        smi = pubchem_query2smiles(query, url)
    except Exception as e:
        if chemspace_api_key:
            try:
                chemspace = ChemSpace(chemspace_api_key)
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
 
@mcp.tool(
    name="Name2SMILES",           # Custom tool name for the LLM
    description=  (
      "Input a IUPAC or common name , returns SMILES."
      """This function queries the given molecule name and returns a SMILES string from the record
      Only query with one specific name."""
      
    )) 
async def Name2SMILES( query: str) -> str:
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
           query = query.upper()
           relevant_rows = csv_data[csv_data['Name']==(query)]
           if not relevant_rows.empty:
               # Get the most relevant answer (assuming we return the first match)
               return relevant_rows.iloc[0]['SMILES']
        except:    
            return str(e)

@mcp.tool(
    name="SMILES2Name",           # Custom tool name for the LLM
    description=  (
     "Input SMILES, returns molecule name."     
    )) 
async def SMILES2Name( query: str) -> str:
    """Use the tool."""
    try:
        if not is_smiles(query):
            try:
                query2smiles = Name2SMILES()
                query = query2smiles.run(query)
            except:
                raise ValueError("Invalid molecule input, no Pubchem entry")
        name = smiles2name(query)
 
        return name
    except Exception as e:
        try:
           csv_data = pd.read_csv('tool/dataset.csv',encoding='ISO-8859-1')
           
           relevant_rows = csv_data[csv_data['SMILES']==(query)]
           if not relevant_rows.empty:
               # Get the most relevant answer (assuming we return the first match)
               return relevant_rows.iloc[0]['Name']
        except:    
            return str(e)
        
@mcp.tool(
   name  = "graphconverter",
   description = (
       "Input graph path , returns SMILES."
       "It was used to convert graph to SMILES"
   )  
     )  
async def graphconverter( image_path: str) -> str:       
    try:
        image_path = str(image_path)
        SMILES = predict_SMILES(image_path, hand_drawn=True)
    except:
            return 'Please recheck the graph path'
    return SMILES
 
        
        
if __name__ =="__main__":
    mcp.run(transport="stdio")       
 
 