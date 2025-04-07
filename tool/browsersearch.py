from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio
from dotenv import load_dotenv
load_dotenv()
import os
from langchain.tools import BaseTool
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
async def main(task):
    agent = Agent(
        task=task,
        llm=ChatOpenAI(model="gpt-4o-2024-11-20",api_key='sk-itPrztYm9F6XZZpsBMJB9O7Vq0pYUABVVBSoThuBxEGTnDik',
             base_url="https://www.dmxapi.com/v1"),
    )
    result = await agent.run()
    return result

class browseruse(BaseTool):
    name: str = "browseruse"
    description: str = ("Calling the browser to search for information in specific website"
                        "input query, return the searching results")

    def __init__(
        self,
    ):
        super().__init__()

    def _run(self, task: str) -> str:
         result = asyncio.run(main(task)) 
         return result
    
    async def _arun(self, smiles: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()
        
        
 