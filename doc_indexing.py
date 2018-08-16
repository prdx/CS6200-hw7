import email
import os
import random
import re
import string
import nltk
from bs4 import BeautifulSoup
from utils.constants import Constants
from utils.es import *
from utils.text import *

def build_labels():
    labels = {}
    with open(Constants.LABEL_PATH, "r") as f:
        labels_list = f.read()
        labels_list = labels_list.split("\n")
    for line in labels_list:
        try:
            label, doc_id = line.split()
            doc_id = doc_id.split("/").pop()
            labels[doc_id] = 1 if label == "spam" else 0
        except Exception as e:
            print(e)

    return labels

words = set(nltk.corpus.words.words())
words.update(set(['viagra', 'xanax', 'valium', 'vicodin', 'morphine', 'percocet']))

delete_index()
create_index()
counter = 0
files = [ f for f in os.listdir(Constants.DATA_PATH) if f.startswith('inmail') ]

train_len = int(len(files) * 0.8)
train_set = set(random.sample(files, train_len))
test_set = set(files) - train_set

labels = build_labels()

for f in files:
    with open(Constants.DATA_PATH + f, 'rb') as d:
        data = d.read().decode('utf-8', 'ignore')
        text = ""
        e = email.message_from_string(data)

    if e['subject'] != None:
        text += e['subject'] + " "
    else:
        print("Subject in None: " + f)

    if e.is_multipart():
        for part in e.walk():
            for payload in e.get_payload():
                ctype = part.get_content_type()
                cdispo = str(part.get('Content-Disposition'))

                # skip any text/plain (txt) attachments
                if (ctype == 'text/plain' or ctype == 'text/html') and 'attachment' not in cdispo:
                    text += part.get_payload() + " "  # decode
                    break
    else:
        text += e.get_payload() + " "

    # Clean the HTML tags
    text = BeautifulSoup(text, "lxml").text

    # Remove spaces and tab
    text = text.replace("\n", " ")
    text = text.replace("\t", "")

    # Lower the case
    text = text.lower()

    # If there is HTML comments, clean that as well
    text = re.sub("<!--.+?-->", "", text, flags=re.MULTILINE)

    # Remove non-english words
    text = " ".join(w for w in nltk.wordpunct_tokenize(text) \
             if w.lower() in words or not w.isalpha())

    text = text.strip()

    # Remove punctuations
    for punct in string.punctuation:
        if punct != '_' and punct != '\'' and punct != '-':
            text = text.replace(punct, " ")

    if f in test_set:
        is_test = 1
    else:
        is_test = 0
    is_spam = labels[f]
    store_document(f, text, is_spam, is_test, len(text))
    counter += 1

print(str(counter))

