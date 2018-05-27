import pandas as pd
import numpy as np
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Input, Layer, Dropout, Masking
from keras.layers import GRU, Activation
from keras.models import Model, Sequential
import numpy as np
from keras.models import model_from_json
import time
import argparse
#from keras.utils import to_categorical
import tensorflow as tf
import os


PATH_TO_WORKING_DIR = 'f:/dss18/item_embedding_session_reco/'
PATH_TO_ORIGINAL_DATA = '././data/'
PATH_TO_PROCESSED_DATA = '././pre-train/'
file_2_tokenize_name = PATH_TO_PROCESSED_DATA + 'rcs15_sesss_tr.txt'
file_2_tokenize_name_test = PATH_TO_PROCESSED_DATA + 'rcs15_sesss_ts.txt'
tokenized_file_name = PATH_TO_PROCESSED_DATA + 'rcs15_sesss_tr_tk.txt'
tokenized_file_name_test = PATH_TO_PROCESSED_DATA + 'rcs15_sesss_ts_tk.txt'
tokenized_file_name_X = PATH_TO_PROCESSED_DATA + 'rcs15_sesss_tr_tkX.txt'
tokenized_file_name_Y = PATH_TO_PROCESSED_DATA + 'rcs15_sesss_tr_tkY.txt'
tokenized_file_name_test_X = PATH_TO_PROCESSED_DATA + 'rcs15_sesss_ts_tkX.txt'
tokenized_file_name_test_Y = PATH_TO_PROCESSED_DATA + 'rcs15_sesss_ts_tkY.txt'
glove_vectors_file_name = PATH_TO_PROCESSED_DATA + 'vectors_s_rcs15.txt'
glove_vocab_file_name = PATH_TO_PROCESSED_DATA + 'rcs15_s_vocab.txt'
vocab_size = 14386
sessions_train = 781486
sessions_test = 30726
sequence_length = 3
sequence_len = 3
embedding_size = 50
do_val = 0.2
n_samples = 10
hidden_neurons = 30  #RNN layer output dim
num_iter = 2
N = 10  #recall@
maxW = N+3   #max item clicke we count

def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    '''
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes),dtype=np.bool)
    for i in range(len(y)):
        idx = y[i,0]
        Y[i, idx] = 1
    return Y

def calcRecall(recom,actual,i):
    act_set = set(actual)
    pred_set = set(recom[:i])
    result = (float) (len(act_set & pred_set)) / float(len(act_set))
    return result

def generate(glove_vectors_file_name,glove_vocab_file_name):
    with open(glove_vocab_file_name, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(glove_vectors_file_name, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    return (W_norm, vocab, ivocab)

def run_experiment():
    W, vocab, ivocab = generate(glove_vectors_file_name, glove_vocab_file_name)

    print("loading train")
    y1 = np.loadtxt(tokenized_file_name_Y, dtype='int').reshape((sessions_train, 1))
    Y = to_categorical(y1, len(vocab) + 1)
    X = np.loadtxt(tokenized_file_name_X).reshape((sessions_train,sequence_len,embedding_size))
    out_n = len(vocab)+1

    #create the model here
    print("Creating model...")
    model = Sequential()
    print("Adding GRU ...")
    model.add(GRU(hidden_neurons, input_dim=embedding_size, return_sequences=False))
    print("Adding dropout ...")
    model.add(Dropout(do_val))
    print("adding output layer...")
    model.add(Dense(out_n, input_dim=hidden_neurons))
    print("adding activation...")
    model.add(Activation("softmax"))
    print("compiling...")
    model.compile(optimizer='adam',loss='categorical_crossentropy')
    print("compiled!")

    print(model.summary())

    data_size = X.shape[0]
    train_size = int(data_size)

    X_train = X[0:train_size, :]
    Y_train = Y[0:train_size]

    for II in range(1, num_iter):
        if II < 10:
            step = 1
        else:
            step = 10
        model.fit(X_train, Y_train, batch_size=100, nb_epoch=step, validation_split=0.05)

         # saving the model structures and parameters so it can be loaded later
        z1 = "m3_%d" % (II)
        json_string = model.to_json()
        open(z1+'_my_model_architecture.json', 'w').write(json_string)
        model.save_weights(z1+'_my_model_weights.h5')

def load_test_data():
    print("before load test")

    # load first 3 clicks embedded vectors
    XTS = np.loadtxt(tokenized_file_name_test_X).reshape((sessions_test, sequence_len, embedding_size))
    test_size = int(XTS.shape[0])
    X_test = XTS[0:test_size, :]

    # load next user clicks id
    with open(file_2_tokenize_name_test, 'r') as f:
        full_data = [line.rstrip().split(' ') for line in f]
        data = full_data

    print("after load test")
    return X_test,data

def load_model():
    json_string = 'm3_10_my_model_architecture.json'
    json_file = open(PATH_TO_PROCESSED_DATA + json_string, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = Sequential()
    model = model_from_json(loaded_model_json)
    print(model.summary())

    # load weights into new model
    weights_string = 'm3_10_my_model_weights.h5'
    model.load_weights(PATH_TO_PROCESSED_DATA + weights_string)
    print("Loaded model from disk")

    # evaluate loaded model on test data
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

def calc_pred_recall(pred,click_id):
    W, vocab, ivocab = generate(glove_vectors_file_name, glove_vocab_file_name)
    recall = 0
    rown = 0
    rownc = 0
    recallV = [0] * N

    print("testing model ")
    for row in click_id:
        if len(row) < 4:  # nothing to predict
            rown = rown + 1
            continue  # nothin to predict

        predicted_results = pred[rown]
        dist = predicted_results

        #filter first 3 clicks from results
        input_term = row[0:3]
        for term in input_term:
            if term in vocab:
                index = vocab[term]
                dist[index] = -np.Inf

        #sort top N predictions
        top = np.argsort(-dist)[:N]
        top_original_id = []
        for glv_item_id in top:
            org_item_id = ivocab[glv_item_id]
            top_original_id.append(org_item_id)

        top_original_set = set(top_original_id)
        matched = [click_id for click_id in row[3:maxW] if click_id in top_original_set]
        recall = recall + (float(len(matched)) / (float)(len(row[3:maxW])))
        rownc = rownc + 1
        rown = rown + 1
        for i in range(N):
            recallV[i] = recallV[i] + calcRecall(top_original_id, row[3:maxW], (i + 1))

    recall = (float)(recall) / (float)(rownc)
    print('recall',recall)

    for i in range(N):
        recallV[i] = (float)(recallV[i]) / (float)(rownc)
        print('recall@ ' + str(i+1) + ' ' + str(recallV[i]) + "\n")

def test_prediction():
    X_test,click_data = load_test_data()
    # load and create model
    model = load_model()

    # use first 3 clicks to recommend
    Y_test = model.predict(X_test)
    calc_pred_recall(Y_test,click_data)

def main():
    os.chdir(PATH_TO_WORKING_DIR)
    #run_experiment()
    test_prediction()


if __name__ == "__main__":
    main()