from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio
from dotenv import load_dotenv
load_dotenv()
import os
from langchain.tools import BaseTool
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
async def main(task, openai_api_key):
    agent = Agent(
        task=task,
        llm=ChatOpenAI(model="deepseek-v3.1-nothinking",api_key=openai_api_key,  base_url=os.getenv("OPENAI_API_BASE") ))
    result = await agent.run()
    return result

class browseruse(BaseTool):
    name: str = "browseruse"
    description: str = ("Calling the browser to search for information in specific website"
                        "input query, return the searching results")
    openai_api_key: str = None
    def __init__(
        self,openai_api_key
    ):
        super().__init__()
        self.openai_api_key = openai_api_key
    def _run(self, task: str) -> str:
         result = asyncio.run(main(task, self.openai_api_key)) 
         return result
    
    async def _arun(self, smiles: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()
        
        
 