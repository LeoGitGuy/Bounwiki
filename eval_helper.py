from haystack import Label, MultiLabel, Answer
import json
import re
import pickle
import csv
import copy

def read_labels(labels, tables):
    processed_labels = []
    for table in tables:
        if table.id not in labels:
            continue
        doc_labels = labels[table.id]
        for label in doc_labels:
            label = Label(
                query=label["question"],
                document=table,
                is_correct_answer=True,
                is_correct_document=True,
                answer=Answer(answer=label["answers"][0]["text"]),
                origin="gold-label",
            )
            processed_labels.append(MultiLabel(labels=[label]))
    return processed_labels

def create_labels(labels_file, data, seperate_eval):
  eval_labels = []
  with open(labels_file) as labels_file:
    labels = json.load(labels_file)
  if seperate_eval:
    use_labels = filter_labels(labels)
  else:
    use_labels = [labels]
  for l in use_labels:
    labels = []
    for d in data:
      labels += read_labels(l, d)
    print(f"Number of Labels: {len(labels)}")
    eval_labels.append(labels)
  return eval_labels

def get_processed_squad_labels(squad_labels):
  with open(f'./data/validation_data/{squad_labels}') as fp:
    squad_labels = json.load(fp)
  # Process Squad File by aligning the right document IDs for the course schedules
  processed_squad_labels = {}
  for paragraph in squad_labels["data"]:
    context = paragraph["paragraphs"][0]["context"]
    if context[:43] == "Code\tName\tEcts\tInstructor\tDays\tHours\tRooms\n":
      faculty_abb = re.search(r"[a-z]*", context[43:], re.IGNORECASE).group()
      if faculty_abb in processed_squad_labels:
        processed_squad_labels[faculty_abb].extend(paragraph["paragraphs"][0]["qas"])
      else:
        processed_squad_labels[faculty_abb] = paragraph["paragraphs"][0]["qas"]
    else:
      processed_squad_labels[str(paragraph["paragraphs"][0]["document_id"])] = paragraph["paragraphs"][0]["qas"]

  with open("./data/validation_data/processed_qa.json", "w") as outfile:
    json.dump(processed_squad_labels, outfile)
  #return processed_squad_labels
  
def filter_labels(labels):
  with open("./data/validation_data/questions_new.txt", "r") as fp:
    user_questions = fp.read()

  user_questions = user_questions.split("\n")
  user_questions = [qu.strip() for qu in user_questions]
  user_squad_labels = {}
  synthetic_squad_labels = {}
  for doc, questions in labels.items():
    for q in questions:
      if q["question"].strip() in user_questions:
        if doc in user_squad_labels:
          user_squad_labels[doc].append(q)
        else:
          user_squad_labels[doc] = [q]
      else:
        if doc in synthetic_squad_labels:
          synthetic_squad_labels[doc].append(q)
        else:
          synthetic_squad_labels[doc] = [q]
          
  return [user_squad_labels, synthetic_squad_labels]

def save_results(text_reader, table_reader, filenames, label_types, idx, top_k, results, res_dict, header):
   with open(f"./output/{text_reader}_{table_reader}_{('_').join(filenames)}_{label_types[idx]}_{top_k}", "wb") as fp:
      pickle.dump(results, fp)
   exp_dict = {
      "Text Reader": text_reader,
      "Table Reader": table_reader,
      "Context" : ('_').join(filenames),
      "Label type": label_types[idx],
      "Topk": top_k
   }
   if 'JoinAnswers' in res_dict:
      csv_dict_new = {**res_dict['EmbeddingRetriever'], **res_dict['JoinAnswers'], **exp_dict}
   elif 'TableReader' in res_dict:
      csv_dict_new = {**res_dict['EmbeddingRetriever'], **res_dict['TableReader'], **exp_dict}
   elif 'TextReader' in res_dict:
      csv_dict_new = {**res_dict['EmbeddingRetriever'], **res_dict['TextReader'], **exp_dict}
   if idx == 1:
      csv_dict_all = {}
      # iterating key, val with chain()
      total_num_samples = csv_dict["num_examples_for_eval"] + csv_dict_new["num_examples_for_eval"]
      weight_old = csv_dict["num_examples_for_eval"]
      weight_new = csv_dict_new["num_examples_for_eval"]
      print("Weights for datasets:", weight_old, weight_new)
      print("new")
      for key, val in csv_dict.items():
         if not isinstance(val, str) and key != "Topk":
            if key != "num_examples_for_eval":
               csv_dict_all[key] = ((val*weight_old + csv_dict_new[key]*weight_new)/total_num_samples)
            else:
               csv_dict_all[key] = (val + csv_dict_new[key])
         else:
            csv_dict_all[key] = val
      csv_dict_all["Label type"] = "all_eval"
      with open("./output/results.csv", "a", newline='') as f:
         writer = csv.DictWriter(f, fieldnames=header)
         writer.writerow(csv_dict_all)
   csv_dict = copy.deepcopy(csv_dict_new)
   print(csv_dict)
   
   with open("./output/results.csv", "a", newline='') as f:
      writer = csv.DictWriter(f, fieldnames=header)
      writer.writerow(csv_dict)
