from haystack.document_stores import ElasticsearchDocumentStore, InMemoryDocumentStore
import os
import pickle
import os

def get_document_store(document_index, use_ada_embeddings = False):
    if use_ada_embeddings:
        embedding_dim = 1538
    else:
        embedding_dim = 768
    document_store = InMemoryDocumentStore(index=document_index, embedding_dim = embedding_dim)
    return document_store

def add_data(filenames, document_store, document_index):
    data = []
    for filename in filenames:
        with open(f"./data/website_data/{filename}", "rb") as fp:
            file = pickle.load(fp)
            data.append(file)
            document_store.write_documents(file, index=document_index)
    return document_store, data