import gradio as gr

from setup_database import get_document_store, add_data
from setup_modules import create_retriever, create_readers_and_pipeline, text_reader_types, table_reader_types

document_index = "document"
document_store = get_document_store(document_index)
filenames = ["processed_website_tables","processed_website_text","processed_schedule_tables"]
document_store, data = add_data(filenames, document_store, document_index)
document_store, retriever = create_retriever(document_store)
text_reader_type = text_reader_types['deberta-large']
table_reader_type = table_reader_types['tapas']
pipeline = create_readers_and_pipeline(retriever, text_reader_type, table_reader_type, True, True)

title = "Welcome to the BounWiki: The Question Answering Enginge for Bogazici Students!"

head = '''
This engine uses information from the Bogazici University Website to answer questions about different areas such as:

 - Semester Dates (e.g. Registration Period, Add/Dropp Period...)
 - Campus buildings and their locations
 - General Uni Information, like Busses from Uni, Taxi-Numbers
 - Schedule Information for all courses
 
It returns the top 
'''


article = '''
# How does this work?

This App uses an "MPNet" sentence-transformer to encode information from the website into an embedding space.
When faced with a query, the semantically most similar document is retrieved.
A language model ("deberta-large" here) extracts the answer to the original question from this document and returns it to the interface 
'''


examples = ["When is the add/dropp period?", "What does it mean if instructor consent is required?", "Where is the english preparatory unit located?"],



label = gr.outputs.Label(num_top_classes=3)

def predict(input):
    prediction = pipeline.run(
        query=input, params={"top_k": 3}
        )
    return {a.answer: float(a.score) for a in prediction["answers"]}

interface = gr.Interface(fn=predict, inputs=gr.Textbox(lines=5, max_lines=6, label="Input Text"), outputs=label, title=title, description=head, article=article, examples=examples)
interface.launch()

