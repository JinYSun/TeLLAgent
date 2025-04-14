# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 22:20:13 2024

@author: BM109X32G-10GPU-02
"""
from langchain.chains import LLMChain, SimpleSequentialChain, RetrievalQA, ConversationalRetrievalChain

from langchain import PromptTemplate 
 
from langchain.tools import BaseTool
 
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.base_language import BaseLanguageModel
from langchain.text_splitter import CharacterTextSplitter
 
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

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
        
class pdfreader(BaseTool):
    name: str = "pdfreader"
    description: str = (

        "Used to read papers, summarize papers, Q&A based on papers, literature or publication"
        "Input query , return the response"
    )

    llm: BaseLanguageModel = None
    path : str = None 
    return_direct: bool = True
    openai_api_key: str = None
    
    def __init__(self, path , openai_api_key):
        super().__init__(  )
        
        self.path = path
        # api keys
        self.openai_api_key = openai_api_key
        self.llm =  ChatOpenAI(model="gpt-4o-2024-11-20",api_key=self.openai_api_key,
             base_url="https://www.dmxapi.com/v1")
    def _run(self, query ) -> str:
       
        loader = PyPDFLoader(self.path)  
        documents = loader.load()  
        
        text_splitter = CharacterTextSplitter(chunk_size=6000, chunk_overlap=1000)
        docs = text_splitter.split_documents(documents) 
        embeddings =  OpenAIEmbeddings(model="text-embedding-3-large",api_key=self.openai_api_key,
             base_url="https://www.dmxapi.com/v1")

       
        vectorstore = FAISS.from_documents(docs, embeddings)
        prompt = PromptTemplate(template=template, input_variables=[ "question"])
        qa_chain = RetrievalQA.from_chain_type(
            llm= self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
           chain_type_kwargs={"prompt": prompt},
        )
         
        result = qa_chain.invoke(query)
        return result['result']
        
 
    async def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("this tool does not support async")
        
 