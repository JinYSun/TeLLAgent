from typing import Optional

import langchain
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain import chains
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import ValidationError
from langchain.agents import AgentExecutor
from langchain.agents.mrkl.base import ZeroShotAgent
from prompts import FORMAT_INSTRUCTIONS, QUESTION_PROMPT, QUESTION_PROMPT1, SUFFIX
from tools import make_tools
 
from rmrkl import ChatZeroShotAgent, RetryAgentExecutor
from langchain_ollama import OllamaLLM
import base64
from io import BytesIO
from PIL import Image
 
from langchain_openai import ChatOpenAI , OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI


def convert_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def _make_llm(model, temp, api_key, streaming: bool = False):
    if model.startswith("claude")  :
        llm = OpenAI(
       temperature=temp,
       model_name=model,
max_tokens = 5000,
       openai_api_key=api_key,
       base_url="https://www.dmxapi.com/v1"
   )
    elif model.startswith("gpt") or model.startswith("deepseek"):
        llm = ChatOpenAI(model=model,
            temperature = 0.1,
            
            timeout=1000,
            
            callbacks=[StreamingStdOutCallbackHandler()],
            openai_api_key=api_key,base_url="https://www.dmxapi.com/v1"
            )
    elif model.startswith("llama") :
            llm = OllamaLLM(model=model,
                temperature = 0.1,    
                )
    else:
        raise ValueError(f"Invalid model name: {model}")
    return llm


class TeLLAgent:
    def __init__(
        self,
        tools=None,
        model1: str = "deepseek-ai/DeepSeek-R1",
        model2: str = "gpt-4o-2024-11-20",
        tools_model="gpt-4o-2024-11-20",
        temp=0.1,
        max_iterations=50,
        verbose=True,
        streaming: bool = True,
        openai_api_key= None,
        api_keys: str = {},
        file_path: str= r"...",
        image_path: str = r"..."
    ):
        """Initialize agent."""
        self.file_path = file_path
        self.image_path = image_path
        load_dotenv()
        try:
            self.llm1 = _make_llm(model1, temp, openai_api_key, streaming)
            self.llm2 = _make_llm(model2, temp, openai_api_key, streaming)
        except ValidationError:
            raise ValueError("Invalid OpenAI API key")

        if tools is None:
            api_keys["OPENAI_API_KEY"] = 'sk-itPrztYm9F6XZZpsBMJB9O7Vq0pYUABVVBSoThuBxEGTnDik'
            tools_llm = _make_llm(tools_model, temp, openai_api_key, streaming)
            tools = make_tools(tools_llm, api_keys=api_keys, verbose=verbose, image_path = image_path, file_path = file_path)
 
        # Initialize agent
        self.agent_executor1 = RetryAgentExecutor.from_agent_and_tools(
            tools=tools,
            agent=ChatZeroShotAgent.from_llm_and_tools(
                self.llm1,
                tools,
                suffix=SUFFIX,
                format_instructions=FORMAT_INSTRUCTIONS,
                question_prompt=QUESTION_PROMPT1, return_intermediate_steps=True ,handle_parsing_errors=True 
            ),
            verbose=True,
            max_iterations=1 , return_intermediate_steps=True, handle_parsing_errors=True 
        )
        self.agent_executor2 = RetryAgentExecutor.from_agent_and_tools(
            tools=tools,
            agent=ChatZeroShotAgent.from_llm_and_tools(
                self.llm2,
                tools,
                suffix=SUFFIX,
                format_instructions=FORMAT_INSTRUCTIONS,
                question_prompt=QUESTION_PROMPT, handle_parsing_errors=True 
            ),
            verbose=True,
            max_iterations=max_iterations, handle_parsing_errors=True  
        )
 

    def run(self, prompt):
        prompt = prompt + ' ' + str(self.file_path) + ' ' + str(self.image_path)
        outputs = self.agent_executor1.invoke( {"input": prompt})
        if outputs["intermediate_steps"] ==[]:
            prompt = str(' ' +outputs["input"]+ ' ' + outputs["output"].split('Action:')[0] )
            outputs = self.agent_executor2.invoke( {"input":prompt })            
        else: 
            prompt = str(' ' + outputs["input"] + ' ' + outputs["intermediate_steps"][0][0].log.split('Action:')[0])
            outputs = self.agent_executor2.invoke( {"input": prompt})
        return outputs['output'] 
    
if __name__ == '__main__':
        agent = TeLLAgent( temp=0.1, streaming=  True,
                           openai_api_key =r'sk-itPrztYm9F6XZZpsBMJB9O7Vq0pYUABVVBSoThuBxEGTnDik', 
                           image_path= r"..." ,file_path = r"..." )
        A = agent.run(r"""who are you""")
        