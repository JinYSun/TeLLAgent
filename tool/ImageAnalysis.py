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
import base64
from io import BytesIO
from PIL import Image

 

def convert_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


class Imageanalysis(BaseTool):
    name: str = "Imageanalysis"
    description: str = (
        "Useful to answer questions according to the image, figure, diagram or graph. "
        "Useful to analysis the information in the image, figure, diagram or graph. "
        "Input query about image/figure/graph/diagram, return the response"
    )
   
    llm: BaseLanguageModel = None
    path : str = None 
    openai_api_key: str = None
    def __init__(self, path, openai_api_key):
        super().__init__(  )
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(model="gpt-4o-2024-11-20",api_key=self.openai_api_key,
             base_url="https://www.dmxapi.com/v1")
        self.path = path
        # api keys
        
    def _run(self, query ) -> str:
        try:
            pil_image = Image.open(self.path)
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
            response = self.llm.invoke([message])
            return response.content
        
        except Exception as e: 
            return str(e)
        

    async def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("this tool does not support async")
        
 