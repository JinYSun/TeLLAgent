import os
import re
from mcp.server.fastmcp import FastMCP 
import langchain
from paperqa import Docs, Settings
import asyncio
import paperqa
import paperscraper
from langchain_community.utilities import SerpAPIWrapper
 
 
 
from pypdf.errors import PdfReadError
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import nest_asyncio
 
from typing import Any, Dict, List
import requests
 
 
mcp =FastMCP("search") 
from langchain.chains import LLMChain
nest_asyncio.apply() 
def load_api_keys(file_path='api.txt'):
     
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):  
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
    
                    if value == 'None':
                        continue
                    
                    os.environ[key] = value
                    print(f" {key}")
  
load_api_keys("api.txt")
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
    prompt = langchain.PromptTemplate(
        input_variables=["question"],
        template="""
        I would like to find scholarly papers to answer
        this question: {question}. Your response must be at
        most 10 words long.
        'A search query that would bring up papers that can answer
        this question would be: '""",
    )

    query_chain = LLMChain(llm=llm, prompt=prompt)
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

# @mcp.tool(
#     name="LiteratureSearch",           # Custom tool name for the LLM
#     description= ("Useful to answer questions that require technical "
#     "knowledge. Ask a specific question."), # Custom description
 
# )
# async  def LiteratureSearch(  query) -> str:
#         semantic_scholar_api_key =  os.getenv(  "SEMANTIC_SCHOLAR_API_KEY")
#         openai_api_key = os.getenv("OPENAI_API_KEY")
#         llm = ChatOpenAI(model="deepseek-v3.1",openai_api_key= openai_api_key,
#              base_url=os.getenv("OPENAI_API_BASE"))
  
#         return asyncio.run(scholar2result_llm(
#              llm,
#             query,
#             openai_api_key= openai_api_key,
#             semantic_scholar_api_key= semantic_scholar_api_key
#         ))

  

def web_search(keywords, search_engine="google"):
    try:
        return SerpAPIWrapper(
            serpapi_api_key=os.getenv("SERP_API_KEY"), search_engine=search_engine
        ).run(keywords)
    except:
        return "No results, try another search"

@mcp.tool(
    name="WebSearch",           # Custom tool name for the LLM
    description= ( "Input a specific question, returns an answer from web search. "
     "Give more detailed information and use more general features to formulate your questions.") # Custom description
)
async def WebSearch(  query: str) -> str:
    serp_api_key = os.getenv("SERP_API_KEY")
    if not serp_api_key:
        return (
            "No SerpAPI key found. This tool may not be used without a SerpAPI key."
        )
    return web_search(query)

class WikipediaSearcher:
    """Wikipedia searcher class"""
    
    def __init__(self):
        self.base_url = "https://en.wikipedia.org/w/api.php"
        self.session = requests.Session()
        # Set User-Agent to avoid being blocked
        self.session.headers.update({
            'User-Agent': 'WikipediaSearchTool/1.0 (https://example.com/contact)'
        })
    
    def get_best_match(self, query: str, max_paragraphs: int = 5) -> Dict[str, Any]:
        """Search and get detailed information for the most relevant article"""
        
        # Step 1: Search for the best match
        search_params = {
            "action": "opensearch",
            "search": query,
            "limit": 1,  # Only the most relevant one
            "namespace": 0,
            "format": "json"
        }
        
        try:
            search_response = self.session.get(self.base_url, params=search_params, timeout=10)
            if search_response.status_code != 200:
                return {"found": False, "error": "Search request failed"}
            
            search_data = search_response.json()
            titles = search_data[1] if len(search_data) > 1 else []
            
            if not titles:
                return {"found": False, "error": "No relevant articles found"}
            
            best_title = titles[0]
            
            # Step 2: Get detailed content
            content_params = {
                "action": "query",
                "format": "json",
                "titles": best_title,
                "prop": "extracts|info|pageimages",
                "exintro": False,  # Get full content, not just intro
                "explaintext": True,
                "exsectionformat": "plain",
                "inprop": "url",
                "piprop": "original"  # Get original image
            }
            
            content_response = self.session.get(self.base_url, params=content_params, timeout=15)
            if content_response.status_code != 200:
                return {"found": False, "error": "Content retrieval failed"}
            
            content_data = content_response.json()
            pages = content_data.get("query", {}).get("pages", {})
            
            for page_id, page_info in pages.items():
                if page_id != "-1":
                    extract = page_info.get("extract", "")
                    
                    # Split into paragraphs and limit the count
                    paragraphs = [p.strip() for p in extract.split('\n\n') if p.strip()]
                    if len(paragraphs) > max_paragraphs:
                        paragraphs = paragraphs[:max_paragraphs]
                        truncated = True
                    else:
                        truncated = False
                    
                    return {
                        "found": True,
                        "title": page_info.get("title", ""),
                        "content": '\n\n'.join(paragraphs),
                        "url": page_info.get("fullurl", ""),
                        "word_count": len(extract),
                        "paragraph_count": len(paragraphs),
                        "truncated": truncated,
                        "image": page_info.get("original", {}).get("source") if page_info.get("original") else None
                    }
            
            return {"found": False, "error": "Page content is empty"}
                    
        except Exception as e:
            return {"found": False, "error": f"Error during search: {str(e)}"}
    
    def close(self):
        """Close the session"""
        self.session.close()

# Define search tool
@mcp.tool()
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia and return the full article content of the most relevant entry
    
    Parameters:
    - query: Search keyword
    
    Returns:
    Full article content in plain text for the most relevant Wikipedia entry
    """
    searcher = WikipediaSearcher()
    try:
        result = searcher.get_best_match(query)
        
        if not result.get("found"):
            return f"Error: No Wikipedia entries found for '{query}'. {result.get('error', '')}"
        
        return result["content"]
        
    finally:
        searcher.close() 
        
if __name__ =="__main__":
    mcp.run(transport="stdio")