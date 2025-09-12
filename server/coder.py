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
os.environ["OPENAI_API_BASE"] ="https://www.dmxapi.com/v1"
os.environ["OPENAI_API_KEY"] = 'sk-itPrztYm9F6XZZpsBMJB9O7Vq0pYUABVVBSoThuBxEGTnDik'
os.environ["SERP_API_KEY"] = '3795acda6a74ea15033d34b54eac82982b26f559147d9cf04aca4bfca91c3e9d'
os.environ["SEMANTIC_SCHOLAR_API_KEY"] = 'ih2U0GIUZn9RMGy8GYgTz0C0ZmIaG4R4ujHZi7d3'
  
 
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
 
 