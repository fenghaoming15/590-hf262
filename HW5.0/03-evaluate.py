#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 11:57:18 2021

@author: haomingfeng
"""
from tensorflow import keras
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import auc
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import nltk

model = keras.models.load_model('novel_classification_model')

model.summary()

dir = "/Users/haomingfeng/590-hf262/HW5.0/Data"

pioneer = ""
f = open(os.path.join(dir,"pioneer.txt"))
pioneer = f.read()
f.close()

pioneer_list = np.array(nltk.tokenize.sent_tokenize(pioneer))
pioneer_list = pioneer_list[:700]

ascanio = ""
f = open(os.path.join(dir,"ascanio.txt"))
ascanio = f.read()
f.close()

ascanio_list = np.array(nltk.tokenize.sent_tokenize(ascanio))
ascanio_list = ascanio_list[:700]

roadtobunker = ""
f = open(os.path.join(dir,"roadtobunker.txt"))
roadtobunker = f.read()
f.close()

roadtobunker_list = np.array(nltk.tokenize.sent_tokenize(roadtobunker))
roadtobunker_list = roadtobunker_list[:700]

texts = []
labels = []


for i in range(700):
    texts.append(pioneer_list[i])
    labels.append(0)
    texts.append(ascanio_list[i])
    labels.append(1)
    texts.append(roadtobunker_list[i])
    labels.append(2)
    

maxlen = 100
training_samples = 1200
validation_samples = 300
max_words = 10000

tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print("Found %s unique tokens. " % len(word_index))

data = pad_sequences(sequences,maxlen = maxlen)

labels = np.asarray(labels)
print('shape of data tensor:' , data.shape)
print('shape of label tensor:', labels.shape)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples:training_samples + validation_samples]
y_val = labels[training_samples:training_samples + validation_samples]
y_train = to_categorical(y_train, 3)
y_val = to_categorical(y_val, 3)

model.evaluate(x_train,y_train)