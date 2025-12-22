import asyncio
import os
import ssl
from typing import Any, Dict, List
import certifi
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from logger import Colors, log_error, log_header, log_info, log_success, log_warning
from langchain_ollama import OllamaEmbeddings

load_dotenv()


""" embedding = OpenAIEmbeddings(
    model='text-embedding-3-small',
    show_progress_bar=True,
    chunk_size=70,
    retry_min_seconds=10
    ) """
embeddings = OllamaEmbeddings(model="nomic-embed-text")

#chromeLocalDB = Chroma(persist_directory='chroma_db', embedding_function=embedding)
vectorStore = PineconeVectorStore(index_name='langchain-doc-index', embedding=embeddings)

tavilyExtract = TavilyExtract()
tavilyMap = TavilyMap(max_depth=1,max_breadth=1,max_pages=1)
tavilyCrawl = TavilyCrawl()

async def main():
    log_header('Doc injection pipeline!')

    log_info('Web crawler started', Colors.PURPLE)

    res = tavilyCrawl.invoke({
        'url':'https://en.wikipedia.org/wiki/Narendra_Modi',
        'max_depth':1,
        'instructions':'content on Narendra Modi'
    })

    all_docs = res['results']
    print(all_docs)
    log_success(f'Finish crawling {len(all_docs)} URLs from the site')

if __name__ == "__main__":
    asyncio.run(main())