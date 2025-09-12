# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 15:35:19 2024

@author: BM109X32G-10GPU-02
"""
import os
 
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.base_language import BaseLanguageModel
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
 
mcp =FastMCP("codewriter") 
@mcp.tool(
    name="codewriter",           # Custom tool name for the LLM
    description=  (
    "Useful to answer questions that require writing codes "
    "return the usage and instruction of codes"
    )) 
async def codewriter( query) -> str:
    base_url=os.getenv("OPENAI_API_BASE")
 
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="deepseek-v3.1",openai_api_key= openai_api_key,
         base_url=base_url)
    messages = [
        SystemMessage(content="You are an expert at writing code, write the corresponding code based on the inputs"),
        HumanMessage(content=query),
    ]
    
    response =  llm.invoke(messages)
    return response
 
if __name__ =="__main__":
    mcp.run(transport="stdio")       
 
 