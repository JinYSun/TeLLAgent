# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 15:35:19 2024

@author: BM109X32G-10GPU-02
"""
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.base_language import BaseLanguageModel
import base64
from io import BytesIO
from PIL import Image
from mcp.server.fastmcp import FastMCP 
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
 
mcp =FastMCP("Imageanalysis") 
def convert_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

@mcp.tool(
    name="Imageanalysis",           # Custom tool name for the LLM
    description=  (
        "Useful to answer questions according to the image, figure, diagram or graph. "
        "Useful to analysis the information in the image, figure, diagram or graph. "
        "Input query and path about image/figure/graph/diagram, return the response"
    ))        
async def Imageanalysis( query, path ) -> str:
    base_url=os.getenv("OPENAI_API_BASE")
 
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o-2024-11-20",openai_api_key= openai_api_key,
         base_url=os.getenv("OPENAI_API_BASE"))
    try:
        pil_image = Image.open(path)
        rgb_im = pil_image.convert('RGB')
        image_b64 = convert_to_base64(pil_image)
        message = HumanMessage(
            content=[
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {"url":f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],)
        response = llm.invoke([message])
        return response.content
    
    except Exception as e: 
        return str(e)
 
from DECIMER import predict_SMILES
 
 
@mcp.tool(
    name="graphconverter",           # Custom tool name for the LLM
    description=  (
       "Input molecule graph path , returns SMILES."
       "It was used to convert graph to SMILES"
    ))  
async def graphconverter( image_path: str) -> str:
     
    try:
        image_path = str(image_path)
        SMILES = predict_SMILES(image_path, hand_drawn=True)
    except:
            return 'Please recheck the graph path'
    return SMILES
 
 
            
if __name__ =="__main__":
    mcp.run(transport="stdio")       
 

        