# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 22:20:13 2024

@author: BM109X32G-10GPU-02
"""
from langchain.chains import   RetrievalQA 
from mcp.server.fastmcp import FastMCP 
from langchain import PromptTemplate 
import os 
 
  
from langchain.text_splitter import CharacterTextSplitter
 
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
os.environ["OPENAI_API_BASE"] ="https://www.dmxapi.com/v1"
os.environ["OPENAI_API_KEY"] = 'sk-itPrztYm9F6XZZpsBMJB9O7Vq0pYUABVVBSoThuBxEGTnDik'
os.environ["SERP_API_KEY"] = '3795acda6a74ea15033d34b54eac82982b26f559147d9cf04aca4bfca91c3e9d'
os.environ["SEMANTIC_SCHOLAR_API_KEY"] = 'ih2U0GIUZn9RMGy8GYgTz0C0ZmIaG4R4ujHZi7d3'
 
template = """

        You are an expert chemist and your task is to respond to the question or
        solve the problem to the best of your ability. You need to answer in as much detail as possible.
        You can only respond with a single "Final Answer" format.
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        <context>
        {context}
        </context>

        Question: {question}
        Answer: 

        """
mcp =FastMCP("pdfreader") 

@mcp.tool(
    name="pdfreader",           # Custom tool name for the LLM
    description=(

        "Used to read papers, summarize papers, Q&A based on papers, literature or publication"
        "Input query and file path , return the response"
    )
)
async def pdfreader(  query , path ) -> str:
         
        openai_api_key = os.getenv("OPENAI_API_KEY")
        llm = ChatOpenAI(model="deepseek-v3.1",openai_api_key= openai_api_key,
             base_url=os.getenv("OPENAI_API_BASE"))       
        loader = PyPDFLoader( path)  
        documents = loader.load()  
        
        text_splitter = CharacterTextSplitter(chunk_size=6000, chunk_overlap=1000)
        docs = text_splitter.split_documents(documents) 
        embeddings =  OpenAIEmbeddings(model="text-embedding-3-large",api_key= openai_api_key,
             base_url=os.getenv("OPENAI_API_BASE"))

       
        vectorstore = FAISS.from_documents(docs, embeddings)
        prompt = PromptTemplate(template=template, input_variables=[ "question"])
        qa_chain = RetrievalQA.from_chain_type(
            llm=  llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
           chain_type_kwargs={"prompt": prompt},
        )
         
        result = qa_chain.invoke(query)
        return result['result']
        
if __name__ =="__main__":
      mcp.run(transport="stdio")       
   
 