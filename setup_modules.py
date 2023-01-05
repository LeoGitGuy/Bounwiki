from haystack.nodes.retriever import EmbeddingRetriever
from haystack.nodes import TableReader, FARMReader, RouteDocuments, JoinAnswers, OpenAIAnswerGenerator
from haystack import Pipeline

text_reader_types = {
    "minilm": "deepset/minilm-uncased-squad2", 
    "distilroberta": "deepset/tinyroberta-squad2", 
    "electra-base": "deepset/electra-base-squad2", 
    "bert-base": "deepset/bert-base-cased-squad2", 
    "deberta-large": "deepset/deberta-v3-large-squad2", 
    "gpt3": "gpt3"
}
table_reader_types = {
    "tapas": "deepset/tapas-large-nq-hn-reader"
}


def create_retriever(document_store, api_key="", use_ada_embeddings = False):
    if not use_ada_embeddings:
        retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/all-mpnet-base-v2-table")
    else:
        retriever = EmbeddingRetriever(
                    document_store=document_store,
                    batch_size=8,
                    embedding_model="text-embedding-ada-002",
                    api_key=api_key,
                    max_seq_len=8191
                    )
    document_store.update_embeddings(retriever=retriever)
    return document_store, retriever

def create_readers_and_pipeline(retriever, text_reader_type = "deepset/roberta-base-squad2", table_reader_type="deepset/tapas-large-nq-hn-reader", use_table=True, use_text=True, api_key = ""):
    both = (use_table and use_text)
    if (use_text or both) and text_reader_type!="gpt3":
        print("Initializing Text reader..")
        text_reader = FARMReader(text_reader_type)
    if (use_table or both) and text_reader_type!="gpt3":
        print("Initializing table reader..")
        table_reader = TableReader(table_reader_type)
    if text_reader_type == "gpt3":
        print("Correct reader")
        text_reader = OpenAIAnswerGenerator(api_key=api_key, model="text-davinci-003")
    if both:
        route_documents = RouteDocuments()
        join_answers = JoinAnswers()
    
    text_table_qa_pipeline = Pipeline()
    text_table_qa_pipeline.add_node(component=retriever, name="EmbeddingRetriever", inputs=["Query"])
    if use_table and not use_text:
        text_table_qa_pipeline.add_node(component=table_reader, name="TableReader", inputs=["EmbeddingRetriever"])
    if (use_text and not use_table) or text_reader_type=="gpt3":
        print("Correct clause")
        text_table_qa_pipeline.add_node(component=text_reader, name="TextReader", inputs=["EmbeddingRetriever"])
    if both and text_reader_type!="gpt3":
        text_table_qa_pipeline.add_node(component=route_documents, name="RouteDocuments", inputs=["EmbeddingRetriever"])
        text_table_qa_pipeline.add_node(component=text_reader, name="TextReader", inputs=["RouteDocuments.output_1"])
        text_table_qa_pipeline.add_node(component=table_reader, name="TableReader", inputs=["RouteDocuments.output_2"])
        text_table_qa_pipeline.add_node(component=join_answers, name="JoinAnswers", inputs=["TextReader", "TableReader"])

    return text_table_qa_pipeline
