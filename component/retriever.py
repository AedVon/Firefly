import os
import json
from typing import Dict, List, Any

from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers import TFIDFRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever


def build_documents(file_path: str, split_text: bool = True) -> List[Dict]:
    assert os.path.isfile(file_path) and os.path.splitext(file_path)[-1] == '.json', "Only support *.json document."

    # build text splitter
    splitter_cfg = {
        "chunk_size": 128,
        "chunk_overlap": 0,
        "separators": ["---break---", "\n\n", "。", "\n", " ", ""],
        "keep_separator": False,
        "length_function": lambda elem: len(elem.split()),
    }
    text_splitter = RecursiveCharacterTextSplitter(**splitter_cfg)

    # load document
    with open(file_path, encoding='utf-8') as f:
        doc_list = json.load(f)

    docs = []
    for doc in doc_list:
        knowledge = doc['knowledge']
        source = doc['source']

        for text_idx, text in enumerate(knowledge, 1):
            if split_text:
                chunks = text_splitter.split_text(text['content'])

                for chunk_idx, chunk in enumerate(chunks, 1):
                    if len(text['summary']) > 0:
                        chunk = f"{text['summary']}\n\n部分原文如下：\n...\n{chunk}\n..."

                    metadata = {
                        'source': source,
                        'doc_id': f'{source}_{text_idx:0>4d}',
                        'chunk_id': f'{source}_{text_idx:0>4d}_{chunk_idx:0>4d}',
                        'summary': text['summary'],
                    }
                    docs.append(Document(page_content=chunk, metadata=metadata))
            else:
                metadata = {
                    'source': source,
                    'doc_id': f'{source}_{text_idx:0>4d}',
                    'summary': text['summary'],
                }
                docs.append(Document(page_content=text['content'], metadata=metadata))

    return docs


def build_tfidf_retriever(file_path: str, split_text: bool = True, **kwargs: Any) -> TFIDFRetriever:
    documents = build_documents(file_path, split_text)
    retriever = TFIDFRetriever.from_documents(documents, **kwargs)
    return retriever


def build_faiss_retriever(
    file_path: str,
    save_dir: str = "resources/faiss_vectorstore",
    **kwargs: Any,
) -> VectorStoreRetriever:

    embed_model = HuggingFaceBgeEmbeddings(
        model_name="/nvme_disk1/public/weights/bge-large-zh-v1.5",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True},
        query_instruction="为这个句子生成表示以用于检索相关文章：",
    )
    embed_model.query_instruction = "为这个句子生成表示以用于检索相关文章："


    if os.path.exists(save_dir):
        vectorstore = FAISS.load_local(save_dir, embeddings=embed_model)
    else:
        documents = build_documents(file_path)
        vectorstore = FAISS.from_documents(
            documents, embedding=embed_model
        )

        # save
        vectorstore.save_local(save_dir)

    retriever = vectorstore.as_retriever(**kwargs)
    return retriever
