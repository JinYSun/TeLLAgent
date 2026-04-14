import os
import re

import langchain
from paperqa import Docs, Settings
import asyncio
import paperqa
import paperscraper
from langchain_community.utilities import SerpAPIWrapper
from langchain.base_language import BaseLanguageModel
from langchain.tools import BaseTool
from langchain_openai import OpenAIEmbeddings
from pypdf.errors import PdfReadError
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import nest_asyncio
from langchain_openai import ChatOpenAI
nest_asyncio.apply() 
def is_smiles(text):
    try:
        m = Chem.MolFromSmiles(text, sanitize=False)
        if m is None:
            return False
        return True
    except:
        return False



def is_multiple_smiles(text):
    if is_smiles(text):
        return "." in text
    return False


def split_smiles(text):
    return text.split(".")

def paper_scraper(search: str, pdir: str = "query", semantic_scholar_api_key: str = None) -> dict:
    try:
        return paperscraper.search_papers(
            search,
            pdir=pdir,
            semantic_scholar_api_key=semantic_scholar_api_key,
        )
    except KeyError:
        return {}


def paper_search(llm, query, semantic_scholar_api_key=None):
    prompt = langchain.prompts.PromptTemplate(
        input_variables=["question"],
        template="""
        I would like to find scholarly papers to answer
        this question: {question}. Your response must be at
        most 10 words long.
        'A search query that would bring up papers that can answer
        this question would be: '""",
    )

    query_chain = langchain.chains.llm.LLMChain(llm=llm, prompt=prompt)
    if not os.path.isdir("./query"):  # todo: move to ckpt
        os.mkdir("query/")
    search = query_chain.invoke(query)
    print("\nSearch:", search)
    papers = paper_scraper(search['text'],   semantic_scholar_api_key=semantic_scholar_api_key)
    return papers


async def scholar2result_llm(llm, query, k=5, max_sources=2, openai_api_key=None, semantic_scholar_api_key=None):
    """Useful to answer questions that require
    technical knowledge. Ask a specific question."""
    papers = paper_search(llm, query, semantic_scholar_api_key=semantic_scholar_api_key)
    if len(papers) == 0:
        return "Not enough papers found"
    docs = Docs()
    settings = Settings()
    settings.llm = llm
    
    not_loaded = 0
    for path, data in papers.items():
        try:
            await docs.aadd(path)
        except (ValueError, FileNotFoundError, PdfReadError):
            not_loaded += 1

    if not_loaded > 0:
        print(f"\nFound {len(papers.items())} papers but couldn't load {not_loaded}.")
    else:
        print(f"\nFound {len(papers.items())} papers and loaded all of them.")

      
    answer =  await docs.aquery(query)
    return answer.answer


class LiteratureSearch(BaseTool):
    name: str = "LiteratureSearch"
    description: str = (
        "Useful to answer questions that require technical "
        "knowledge. Ask a specific question."
    )
    llm: BaseLanguageModel = None
    openai_api_key: str = None 
    semantic_scholar_api_key: str = None


    def __init__(self, llm, openai_api_key, semantic_scholar_api_key):
        super().__init__()
        
        # api keys
        self.openai_api_key = openai_api_key
        self.semantic_scholar_api_key = semantic_scholar_api_key
        self.llm = ChatOpenAI(model="deepseek-v3.1-nothinking",openai_api_key=self.openai_api_key,
             base_url=os.getenv("OPENAI_API_BASE"))
    def _run(self, query) -> str:
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
        return asyncio.run(scholar2result_llm(
            self.llm,
            query,
            openai_api_key=self.openai_api_key,
            semantic_scholar_api_key=self.semantic_scholar_api_key
        ))

    async def _arun(self, query) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("this tool does not support async")

def web_search(keywords, search_engine="google"):
    try:
        return SerpAPIWrapper(
            serpapi_api_key=os.getenv("SERP_API_KEY"), search_engine=search_engine
        ).run(keywords)
    except:
        return "No results, try another search"



class WebSearch(BaseTool):
    name: str = "WebSearch"
    description: str = (
        "Input a specific question, returns an answer from web search. "
        "Give more detailed information and use more general features to formulate your questions."
    )
    serp_api_key: str = None

    def __init__(self, serp_api_key: str = None):
        super().__init__()
        self.serp_api_key = serp_api_key

    def _run(self, query: str) -> str:
       
        if not self.serp_api_key:
            return (
                "No SerpAPI  key found. This tool may not be used without a SerpAPI key."
                )
        return web_search(query)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented")
