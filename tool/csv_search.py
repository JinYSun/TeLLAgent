# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 16:18:29 2024

@author: BM109X32G-10GPU-02
"""

from langchain.tools import BaseTool
import pandas as pd

class search_csv(BaseTool):
    name = "csvsearch"
    description = (
        "input name, return the SMILES of materials "
        "convert name to SMILES."
    )
    llm: BaseLanguageModel = None
    openai_api_key: str = None 
    semantic_scholar_api_key: str = None

    def __init__(self):
        super().__init__()
    def _run(self, smiles: str) -> str:
        csv_data = pd.read_csv('dataset.csv',encoding='ISO-8859-1')
        relevant_rows = csv_data[csv_data['Name']==(query)]
        
        if not relevant_rows.empty:
            # Get the most relevant answer (assuming we return the first match)
            return relevant_rows.iloc[0]['SMILES']
        return None
    async def _arun(self, smiles: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()
        
