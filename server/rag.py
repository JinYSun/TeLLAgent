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
from mcp.server.fastmcp import FastMCP 
from langchain.chains import LLMChain, SimpleSequentialChain, RetrievalQA, ConversationalRetrievalChain
import os
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from torch import cuda, bfloat16
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
from langchain_openai import OpenAIEmbeddings 
os.environ["OPENAI_API_BASE"] ="https://www.dmxapi.com/v1"
os.environ["OPENAI_API_KEY"] = 'sk-itPrztYm9F6XZZpsBMJB9O7Vq0pYUABVVBSoThuBxEGTnDik'
os.environ["SERP_API_KEY"] = '3795acda6a74ea15033d34b54eac82982b26f559147d9cf04aca4bfca91c3e9d'
os.environ["SEMANTIC_SCHOLAR_API_KEY"] = 'ih2U0GIUZn9RMGy8GYgTz0C0ZmIaG4R4ujHZi7d3'
 
 
mcp =FastMCP("rag") 
 
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
@mcp.tool(
    name="rag",           # Custom tool name for the LLM
    description= ( "Useful to answer questions that require technical " 
     "Provide specialized knowledge information for solving Q&A questions"
     "Input query , return the response") # Custom description
)
async def rag(  query ) -> str:
    base_url=os.getenv("OPENAI_API_BASE")
 
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="deepseek-v3.1",openai_api_key= openai_api_key,
         base_url=os.getenv("OPENAI_API_BASE"))
    embeddings = OpenAIEmbeddings(api_key= openai_api_key,
          base_url=base_url)
    
    vectorstore=FAISS.load_local(r"rag", embeddings,allow_dangerous_deserialization =True)  
    prompt = PromptTemplate(template=template, input_variables=[ "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm= llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=False,
        
       chain_type_kwargs={"prompt": prompt},
    )
    chat_history = []

     
    result = qa_chain.invoke(query)
    return result['result']
      
        
if __name__ =="__main__":
    mcp.run(transport="stdio")       
 



