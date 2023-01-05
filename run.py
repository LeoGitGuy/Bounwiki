# imports
import logging
#import torch_scatter
import argparse
import csv
from setup_database import get_document_store, add_data
from setup_modules import create_retriever, create_readers_and_pipeline, text_reader_types, table_reader_types
from eval_helper import create_labels, save_results

def parse_args():
   parser = argparse.ArgumentParser(description="JointXplore")

   parser.add_argument("--context", help='which information should be added as context, subset of [processed_website_tables, processed_website_text, processed_schedule_tables], enter as multiple strings', 
                     nargs='+', default=["processed_website_tables","processed_website_text","processed_schedule_tables"])
   parser.add_argument("--text_reader", help="specify the model to use as text reader", choices=["minilm", "distilroberta", "electra-base", "bert-base", "deberta-large", "gpt3"], default="bert-base")
   parser.add_argument("--api-key", help="if gpt3 choosen as reader, please provide api-key", default="")
   parser.add_argument("--table_reader", help="choose tapas or convert table to text file and treat them as such", choices=["tapas", "text"], default="tapas")
   parser.add_argument("--seperate_evaluation", help="if specified, student generated questions and synthetically generated questions are evaluated seperately", action="store_true")
   parser.add_argument("--top_k", help="number of docs the retriever gets and number of returned answers", type=int, default=3)
   parser.add_argument("--use_ada_embeddings", help="if specified, openai's ada embeddings are used, api key must be provided for this", action="store_true")
   args = parser.parse_args()

   return args
 
def main(*args):
   if args=={}:
      args = parse_args()
      filenames = args.context
      text_reader = args.text_reader
      table_reader = args.table_reader
      seperate_evaluation = args.seperate_evaluation
      api_key = args.api_key
      top_k = args.top_k
      use_ada_embeddings = args.use_ada_embeddings
   else:
      filenames, text_reader, table_reader, seperate_evaluation, api_key, top_k, use_ada_embeddings = args
   print(f"Filenames: {filenames}")
   use_table = False
   use_text = False
   if "processed_schedule_tables" in filenames:
      use_table = True
   if "processed_website_text" or "processed_website_tables" in filenames:
      use_text = True
   
   # configure logger
   logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
   logging.getLogger("haystack").setLevel(logging.WARNING)
   logging.info("Starting..")
   # create document store and add all context documents to it
   document_index = "document"
   document_store = get_document_store(document_index, use_ada_embeddings)
   document_store, data = add_data(filenames, document_store, document_index)
   # create the embeddings and with it the embedding retriever
   document_store, retriever = create_retriever(document_store, api_key, use_ada_embeddings)
   text_reader_type = text_reader_types[text_reader]
   table_reader_type = table_reader_types[table_reader]
   # create the retriever - reader pipeline, depending on which context and reader type was selected
   pipeline = create_readers_and_pipeline(retriever, text_reader_type, table_reader_type, use_table, use_text, api_key)
   # get columns from results file
   with open("./output/results.csv", "r") as f:
      reader = csv.reader(f)
      for header in reader:
         break
   
   # load questions and annotations
   labels_file = "./data/validation_data/processed_qa.json"
   # create labels from questions
   labels = create_labels(labels_file, data, seperate_evaluation)
   # differentiate between student questions and synthetic questions
   label_types = ["all_eval"]
   if seperate_evaluation:
      label_types = ["students", "synthetic"]
   # evaluate student and synthetic questions seperately
   for idx, label in enumerate(labels):
      # predict answers and calculate evaluation metrics
      results = pipeline.eval(label, params={"top_k": top_k}, sas_model_name_or_path="cross-encoder/stsb-roberta-large")
      res_dict = results.calculate_metrics()
      # print summary
      print(res_dict)
      # save results to results file
      save_results(text_reader, table_reader, filenames, label_types, idx, top_k, results, res_dict, header)
   
   # empty database again to clean up
   document_store.delete_index(document_index)


if __name__ == "__main__":
   main()