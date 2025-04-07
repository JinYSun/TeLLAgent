import os

from langchain import agents
from langchain.base_language import BaseLanguageModel

from tool import *


def make_tools(llm: BaseLanguageModel, api_keys: dict = {}, verbose=True,  image_path = r"...", file_path = r"..."):
    serp_api_key = api_keys.get("SERP_API_KEY") or os.getenv("SERP_API_KEY")
    image_path = image_path 
    file_path = file_path
    openai_api_key = api_keys.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    chemspace_api_key = api_keys.get("CHEMSPACE_API_KEY") or os.getenv(
        "CHEMSPACE_API_KEY"
    )
    semantic_scholar_api_key = api_keys.get("SEMANTIC_SCHOLAR_API_KEY") or os.getenv(
        "SEMANTIC_SCHOLAR_API_KEY"
    )
    serp_api_key = '3795acda6a74ea15033d34b54eac82982b26f559147d9cf04aca4bfca91c3e9d'
    all_tools = agents.load_tools(
        [
            #"python_repl",
            # "ddg-search",
            "wikipedia",
            # "human"
        ]
    )

    all_tools += [
       # browseruse(),
        
        rag(),
        codewriter(),
        graphconverter(),
        Query2SMILES(chemspace_api_key),
        Mol2SMILES(chemspace_api_key) ,
        Query2CAS(),
        SMILES2Name(),
        SMILES2SAScore(),
        SMILES2LogP(),
        SMILES2Properties(),
        MolSimilarity(),
        SMILES2Weight(),
        FuncGroups(),
        donor_predictor(),
        acceptor_predictor(),
        homolumo_predictor(),
        dap_screen(),
        
        molgen(),
        dap_predictor(),
  
    ]
    if semantic_scholar_api_key:
        all_tools += [       LiteratureSearch(
                  llm=llm,
                  openai_api_key=openai_api_key,
                  semantic_scholar_api_key=semantic_scholar_api_key        ),
    ]
        
    if serp_api_key:
        all_tools += [WebSearch(serp_api_key)
    ]
    if image_path is not None:
            all_tools += [Imageanalysis(image_path),
        ]   
    if file_path is not None:
            all_tools += [pdfreader(file_path),
        ]               
 
    return all_tools
