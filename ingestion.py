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

vectorstore = Chroma(persist_directory='chroma_db', embedding_function=embeddings)
# vectorStore = PineconeVectorStore(index_name='langchain-collection-768', embedding=embeddings)

tavilyExtract = TavilyExtract()
# tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=500)
tavily_map = TavilyMap(max_depth=1, max_breadth=1)
tavilyCrawl = TavilyCrawl()


""" async def index_doc_async(documents: List[Document], batch_size:int = 50):
    Process documents in batches asynchronously.
    log_header("VECTOR STORAGE PHASE")
    log_info(f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store", Colors.DARKCYAN)
    
    batches = [documents[i : i + batch_size] for i in range(0, len(documents), batch_size)]

    log_info(f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each")

    async def add_batch(batch: List[Document], batch_num: int):
        try:
            async with semaphore:
                await vectorStore.aadd_documents(batch)
                log_success(f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)")
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})")
    else:
        log_warning(f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully")

 """


SEM_LIMIT = 10  # Pinecone-safe: 1–3

async def index_doc_async(documents: List[Document], batch_size: int = 50):
    log_header("VECTOR STORAGE PHASE")

    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
    semaphore = asyncio.Semaphore(SEM_LIMIT)

    async def add_batch(batch: List[Document], batch_num: int):
        async with semaphore:
            try:
                await vectorstore.aadd_documents(batch)
                log_success(f"Batch {batch_num}/{len(batches)} indexed")
                return True
            except Exception as e:
                log_error(f"Batch {batch_num} failed: {e}")
                return False

    results = []
    for i, batch in enumerate(batches):
        results.append(await add_batch(batch, i + 1))

    log_success(f"Indexed {sum(results)}/{len(batches)} batches")


""" async def main():
    log_header('Doc injection pipeline!')
    log_info('Web crawler started', Colors.PURPLE)

    res = tavilyCrawl.invoke({'url':'https://en.wikipedia.org/wiki/Narendra_Modi'})

    all_docs = res['results']
    print("all_docs ==>", all_docs)
    log_success(f'Finish crawling {len(all_docs)} URLs from the site')

    docs_to_split = []
    for item in all_docs:
        doc = Document(
            page_content=item.get('raw_content') or item.get('content') or '',
            metadata={'source': item.get('url'), 'title':'Narendra Modi wiki data bete'}
        )
        docs_to_split.append(doc)

    text_splitter = RecursiveCharacterTextSplitter()
    splitted_doc = text_splitter.split_documents(docs_to_split)

    log_success(f'{len(splitted_doc)} chunks created')

    await index_doc_async(splitted_doc, batch_size=10)
 """

# if __name__ == "__main__":
    # asyncio.run(main())