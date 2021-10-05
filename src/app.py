# Import TensorFlow model dependencies (if needed) - https://github.com/tensorflow/tensorflow/issues/38250 
from re import A
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from flask import Flask
import pandas as pd
model_path = "skimlit_gs_model/skimlit_tribrid_model"
import json as json
from spacy.lang.en import English
from sklearn.preprocessing import LabelEncoder
from flask import request
from flask_cors import CORS
from flask import json
import psycopg2


label_encoder = LabelEncoder()
label_encoder.classes_ = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']


# Load downloaded model from Google Storage
loaded_model = tf.keras.models.load_model(model_path)#,
                                          # Note: with TensorFlow 2.5+ if your SavedModel has a keras_metadata.pb file 
                                          # (created when using model.save()), you shouldn't need the custom_objects
                                          # parameter. I'm leaving the code below here in case you do.
                                          # custom_objects={"TextVectorization": TextVectorization, # required for char vectorization
                                          #                 "KerasLayer": hub.KerasLayer}) # required for token embedding

# Make predictions with the loaded model on the validation set
def split_chars(text):
  return " ".join(list(text))
with open("skimlit_example_abstracts.json", "r") as f:
  example_abstracts = json.load(f)
# print(example_abstracts) abstract, source, details
# See what our example abstracts look like
abstracts = pd.DataFrame(example_abstracts)
# print(abstracts)

nlp = English()
sentencizer = nlp.create_pipe("sentencizer") # create sentence splitting pipeline object
nlp.add_pipe('sentencizer') # add sentence splitting pipeline object to sentence parser

def processData(data):

    doc = nlp(data) # create "doc" of parsed sequences, change index for a different abstract
    abstract_lines = [str(sent) for sent in list(doc.sents)] # return detected sentences from doc in string type (not spaCy token type)
# print(f"these are the abstract sentences: {abstract_lines}")

# Get total number of lines
    total_lines_in_sample = len(abstract_lines)

# Go through each line in abstract and create a list of dictionaries containing features for each line
    sample_lines = []
    for i, line in enumerate(abstract_lines):
        sample_dict = {}
        sample_dict["text"] = str(line)
        sample_dict["line_number"] = i
        sample_dict["total_lines"] = total_lines_in_sample - 1
        sample_lines.append(sample_dict)
# print(sample_lines)

# Get all line_number values from sample abstract
    test_abstract_line_numbers = [line["line_number"] for line in sample_lines]
# One-hot encode to same depth as training data, so model accepts right input shape
    test_abstract_line_numbers_one_hot = tf.one_hot(test_abstract_line_numbers, depth=15) 

    test_abstract_total_lines = [line["total_lines"] for line in sample_lines]
# One-hot encode to same depth as training data, so model accepts right input shape
    test_abstract_total_lines_one_hot = tf.one_hot(test_abstract_total_lines, depth=20)

    abstract_chars = [split_chars(sentence) for sentence in abstract_lines]
    test_abstract_pred_probs = loaded_model.predict(x=(test_abstract_line_numbers_one_hot,
                                                   test_abstract_total_lines_one_hot,
                                                   tf.constant(abstract_lines),
                                                   tf.constant(abstract_chars)))

    test_abstract_preds = tf.argmax(test_abstract_pred_probs, axis=1)
    test_abstract_pred_classes = [label_encoder.classes_[i] for i in test_abstract_preds]
    output = []
    for i, line in enumerate(abstract_lines):
        output.append({"class": test_abstract_pred_classes[i], "content": line}) 
        # print(f"{test_abstract_pred_classes[i]}: {line}")
    return output





# Turn prediction class integers into string class names


app = Flask('skimlit')
app.debug = True
CORS(app)
conn = psycopg2.connect("dbname='skimlit' user='academy' host='localhost'")
conn.autocommit = True
cur = conn.cursor()
@app.route("/", methods=['POST'])
def hello_world():
    if request.json:
        abstract = request.get_json()
        actualabstract = abstract["abstract"]
        cur.execute("""insert into abstracts (abstract) values (%(abstract)s) returning id""", {'abstract': actualabstract})
        # conn.commit()
        idOfInserted = cur.fetchone()[0]
        predictions = processData(actualabstract)
        for prediction in predictions:
            cur.execute("""insert into sentences (abstract_id, sentence, sentence_role) values (%(id)s, %(sent)s, %(role)s)""", {'id': idOfInserted, 'sent': prediction["content"], 'role': prediction["class"]})
        data = {"data": processData(actualabstract)}
        return data
    elif request.form:
        return "<p>got some form</p>"
    return "<p>hello world</p>"

# gets all abstracts   
@app.route("/summaries")
def getAllSummaries():
    cur.execute("select * from abstracts")
    data = {"data": cur.fetchall()}
    print(data)
    dataJson = json.jsonify(data)
    return dataJson

# gets an abstract + its sentences by its id
@app.route("/summaries/<id>")
def getSummaryById(id):
    cur.execute("select * from abstracts left join sentences on abstracts.id = sentences.abstract_id where abstracts.id = %(id)s", {"id": id})
    data = {"data": cur.fetchall()}
    print(data)
    dataJson = json.jsonify(data)
    return dataJson