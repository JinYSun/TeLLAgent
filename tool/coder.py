# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 15:35:19 2024

@author: BM109X32G-10GPU-02
"""

from langchain_community.embeddings import OllamaEmbeddings
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.base_language import BaseLanguageModel


class codewriter(BaseTool):
    name:str = "codewriter"
    description:str = (
        "Useful to answer questions that require writing codes "
        "return the usage and instruction of codes"
    )
    openai_api_key: str = None
    llm: BaseLanguageModel = None
    def __init__(self,llm, openai_api_key):
        super().__init__()
        
        self.openai_api_key = openai_api_key
        self.llm =ChatOpenAI(model="gpt-4o-2024-11-20",openai_api_key=self.openai_api_key,
             base_url="https://www.dmxapi.com/v1")
    def _run(self, query) -> str:
        messages = [
            SystemMessage(content="You are an expert at writing code, write the corresponding code based on the inputs"),
            HumanMessage(content=query),
        ]
        llm = self.llm 
        response =  llm.invoke(messages)
        return response

    async def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("this tool does not support async")
        

 
