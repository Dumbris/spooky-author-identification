import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # regular expressions
from collections import Counter # for counting
# spacy
import spacy
from spacy.parts_of_speech import ADV,ADJ,VERB
from spacy.symbols import nsubj,ORTH,LEMMA, POS,PERSON
from pathlib import Path

from gensim.models import Word2Vec

DATA_DIR = Path("/media/ssd_data/kagle/spooky-author-identification/input")

AUTHORS = ["EAP", "HPL", "MWS"]

train = pd.read_csv(str(DATA_DIR / Path("train.csv")), encoding = "utf8")



# Useful functions:
# transform lists to texts
def list_to_text(l):
    text = " "
    for s in range(len(l)):
        sent = "".join(l[s]).strip()
        text += sent
    return text
# Clean up the texts
def cleanText(text):
    text = text.strip().replace("?","? ").replace(".",". ")
    text = text.lower()
    return text
# Get the documents with spacy
def to_nlp(text):
    nlp = spacy.load('en')
    cleaned_text = cleanText(text)
    document = nlp(cleaned_text)
    return document

all_texts = []
for author in AUTHORS:
    list_ = list(train.text[train.author == author])
    text_ = list_to_text(list_)
    eap = to_nlp(text_eap)
print(len(eap))

# train model
model = Word2Vec(highly_used_verbs, min_count=1)


# For insertion into TensorFlow let's convert the wv word vectors into a numpy matrix
vocabulary_size = len(model.wv.vocab)
vector_dim = 100 #model size
embedding_matrix = np.zeros((vocabulary_size, vector_dim))
for i in range(vocabulary_size):
    embedding_vector = model.wv[model.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


import tensorflow as tf
from tensorflow.contrib import lookup
from tensorflow.python.platform import gfile
#Using gensim Word2Vec embeddings in TensorFlow
# embedding layer weights are frozen to avoid updating embeddings while training
frozen_embeddings = tf.constant(embedding_matrix)
embedding = tf.Variable(initial_value=frozen_embeddings, trainable=False)
# trainable=False, otherwise the embedding layer would be trained with negative performance impacts.

MAX_DOCUMENT_LENGTH = 500
