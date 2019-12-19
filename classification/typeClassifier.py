"""
Possible types:
- Akkoord
- Akte van executeurschap
- Attestatie
- Beraad
- Bestek
- Bevrachtingscontract
- Bewijs aan minderjarigen
- Bijlbrief
- Bodemerij
- Boedelinventaris
- Boedelscheiding
- Borgtocht
- Cessie
- Compagnieschap
- Consent
- Contract
- Conventie (echtscheiding)
- Huur
- Huwelijkse voorwaarden
- Hypotheek
- Insinuatie
- Interrogatie
- Koop
- Kwitantie
- Machtiging
- Non prejuditie
- Obligatie
- Onbekend
- Overig
- Procuratie
- Renunciatie
- Revocatie
- Scheepsverklaring
- Schenking
- Testament
- Transport
- Trouwbelofte
- Uitspraak
- Voogdij
- Wisselprotest
"""

import os
from lxml import etree

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation


def f1(y_true, y_pred):
    """
    Taken from https://www.kaggle.com/guglielmocamporese/macro-f1-score-keras
    """
    y_pred = keras.backend.round(y_pred)
    tp = keras.backend.sum(keras.backend.cast(y_true * y_pred, 'float'),
                           axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = keras.backend.sum(keras.backend.cast((1 - y_true) * y_pred, 'float'),
                           axis=0)
    fn = keras.backend.sum(keras.backend.cast(y_true * (1 - y_pred), 'float'),
                           axis=0)

    p = tp / (tp + fp + keras.backend.epsilon())
    r = tp / (tp + fn + keras.backend.epsilon())

    f1 = 2 * p * r / (p + r + keras.backend.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return keras.backend.mean(f1)


def xml2text(xmlfilepath: str) -> str:

    text = ""
    tree = etree.parse(xmlfilepath)

    lines = tree.xpath(
        '//prima:TextRegion/prima:TextLine/prima:TextEquiv/prima:Unicode/text()',
        namespaces={
            'prima':
            'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'
        })

    for l in lines:
        text += l
        text += "\n"

    return text


class TypeClassifier:
    def __init__(
            self,
            max_words=10000,
            xmldata='/home/leon/Documents/Golden_Agents/saa-notarialTexts/page',
            ydata='/home/leon/Documents/Golden_Agents/page-corrector/classification/data/scans.csv'
    ):

        self.xmldata = xmldata

        self._parseScanData(csvpath=ydata)

        self.tokenizer = Tokenizer(num_words=max_words,
                                   filters='#$%&/;<=>@[\\]^_`{|}~\t',
                                   lower=True,
                                   split=' ',
                                   char_level=False,
                                   oov_token=None)

        if max_words:
            self.max_words = max_words
        else:
            self.max_words = len(self.tokenizer.word_counts)

        # get texts and labels
        texts, labels = zip(
            *self.getTexts(folder=self.xmldata, train_only=True))

        # fit on the corpus (`xmldata`)
        self.tokenizer.fit_on_texts(texts)
        self.X = self.tokenizer.texts_to_sequences(texts)

        # encode class labels
        self.encoder = LabelEncoder()
        self.y = self.encoder.fit_transform(labels)

        # Split in train and text
        X_train, X_test, y_train, y_test = train_test_split(self.X,
                                                            self.y,
                                                            test_size=0.2,
                                                            random_state=42,
                                                            shuffle=True,
                                                            stratify=self.y)

        self.X_train = self.tokenizer.sequences_to_matrix(X_train,
                                                          mode='tfidf')
        self.X_test = self.tokenizer.sequences_to_matrix(X_test, mode='tfidf')

        self.n_classes = len(list(self.encoder.classes_))
        self.y_train = keras.utils.to_categorical(y_train, self.n_classes)
        self.y_test = keras.utils.to_categorical(y_test, self.n_classes)

        # construct model
        self.constructModel()

        # train
        self.train()

    def getTexts(self, folder: str, train_only=True):

        if train_only:
            for f in os.listdir(folder):
                filepath = os.path.join(folder, f)
                scanid = os.path.splitext(f)[0].upper()
                if scanid in self.scan2type:
                    yield xml2text(filepath), self.scan2type[scanid]
        else:
            return (xml2text(os.path.join(folder, f))
                    for f in os.listdir(folder))

    def constructModel(self):

        model = Sequential()
        model.add(Dense(512, input_shape=(self.max_words, ),
                        activation='relu'))
        model.add(Dropout(0.5))
        # model.add(Dense(256, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(self.n_classes, activation='sigmoid'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy', f1])

        print(model.metrics_names)

        self.model = model

        # self.model = keras.Sequential([
        #     keras.layers.Embedding(encoder.vocab_size, 16),
        #     keras.layers.GlobalAveragePooling1D(),
        #     keras.layers.Dense(1, activation='sigmoid')
        # ])

        # self.model.compile(optimizer='adam',
        #                    loss='binary_crossentropy',
        #                    metrics=['accuracy'])

    def train(self, batch_size=32, epochs=10):

        self.history = self.model.fit(self.X_train,
                                      self.y_train,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      epochs=epochs,
                                      verbose=1,
                                      validation_data=(self.X_test,
                                                       self.y_test))

        score = self.model.evaluate(self.X_test,
                                    self.y_test,
                                    batch_size=batch_size,
                                    verbose=1)

        y_val_pred = self.model.predict(self.X_test)
        # y_pred_bool = np.argmax(y_val_pred, axis=1)

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        print('Test f1-score', score[2])

        print(
            classification_report(self.y_test,
                                  y_val_pred,
                                  target_names=self.encoder.classes_))

        # self.history = self.model.fit(train_batches,
        #                               epochs=epochs,
        #                               validation_data=validation_data,
        #                               validation_steps=validation_steps)

    def _parseScanData(self, csvpath: str):

        df = pd.read_csv(csvpath)
        scan2type = dict()
        scan2record = dict()

        for d in df.to_dict(orient='records'):
            scanid = d['scan'].split('/Scan/')[1].upper()

            scan2type[scanid] = d['type']
            scan2record[scanid] = d['record']

        self.scan2type = scan2type
        self.scan2record = scan2record


if __name__ == "__main__":

    C = TypeClassifier()