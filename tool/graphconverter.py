# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:38:35 2024

@author: BM109X32G-10GPU-02
"""

from DECIMER import predict_SMILES
from langchain.tools import BaseTool

class graphconverter(BaseTool):
    name: str = "graphconverter"
    description: str = (
        "Input graph path , returns SMILES."
        "It was used to convert graph to SMILES"
    )
    def __init__(self):
        super().__init__()
    def _run(self, image_path: str) -> str:
         
        try:
            image_path = str(image_path)
            SMILES = predict_SMILES(image_path)
        except:
                return 'Please recheck the graph path'
        return SMILES
 
    async def _arun(self, smiles: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()
        
        
 