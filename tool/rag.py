# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 20:31:22 2025

@author: BM109X32G-10GPU-02
"""


from langchain.tools import BaseTool

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    )

from langchain import PromptTemplate
from langchain import HuggingFacePipeline

from langchain.base_language import BaseLanguageModel
from langchain.chains import LLMChain, SimpleSequentialChain, RetrievalQA, ConversationalRetrievalChain

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from torch import cuda, bfloat16
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
from langchain_openai import OpenAIEmbeddings 

 

 
template = """

You are an expert chemist and your task is to respond to the question or
solve the problem to the best of your ability.You can only respond with a single "Final Answer" format.
You need to list the key points  and explain them in detail and accurately
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}
Answer: 

"""


class rag(BaseTool):
    name: str = "rag"
    description: str= (
        "Useful to answer questions that require technical "
        
        "Provide specialized knowledge information for solving Q&A questions"
        "Input query , return the response"
         
    )
    openai_api_key: str = None
    llm: BaseLanguageModel = None
  
    
    def __init__(self,  openai_api_key):
        super().__init__(  )
         
        # api keys
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(model="deepseek-v3.1-nothinking",api_key=self.openai_api_key,
             base_url="https://www.dmxapi.com/v1")
        
    def _run(self, query ) -> str:
        embeddings = OpenAIEmbeddings(api_key=self.openai_api_key,
              base_url="https://www.dmxapi.com/v1")
        
        vectorstore=FAISS.load_local(r"tool/rag", embeddings,allow_dangerous_deserialization =True)  
        prompt = PromptTemplate(template=template, input_variables=[ "question"])
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=False,
            
           chain_type_kwargs={"prompt": prompt},
        )
        chat_history = []

         
        result = qa_chain.invoke(query)
        return result['result']
        
 
    async def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("this tool does not support async")
        
 




