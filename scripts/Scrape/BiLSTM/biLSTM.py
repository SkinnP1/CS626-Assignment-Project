import nltk
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras import callbacks
from keras.layers import Dense, LSTM,Dropout, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from keras.preprocessing.sequence import pad_sequences
from keras import backend as acc


def custom_accuracy():
    def masked_accuracy(y_true, y_pred):
        true_id = acc.argmax(y_true, axis=-1)
        pred_id = acc.argmax(y_pred, axis=-1)
        ignore_mask = acc.cast(acc.not_equal(pred_id,0), 'int32')
        tensor = acc.cast(acc.equal(true_id, pred_id), 'int32') * ignore_mask
        accuracy = acc.sum(tensor) / acc.maximum(acc.sum(ignore_mask), 1)
        return accuracy
    return masked_accuracy



def onehotencoding(tag_data, length):
    train_y = []
    for x in tag_data:
        tag = []
        for i in x:
            zeroes = np.zeros(length)
            zeroes[i] = 1.0
            tag.append(zeroes)
        train_y.append(tag)
    return np.array(train_y)

#    model.add(Activation('sigmoid'))
def bi_lstm_model():    
    model = Sequential()
    model.add(InputLayer(input_shape=(MAX_LEN, )))
    model.add(Embedding(len(windex), 128))
    model.add(Dropout(0.25))
    model.add(Bidirectional(LSTM(128, return_sequences=True,recurrent_dropout=0.3)))
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Dense(len(tindex),activation='softmax')))
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy', custom_accuracy()])
    return model



def serialize(xyz,pred):
    xyz = xyz.reshape(-1,1).flatten()
    x = pd.Series(np.array(pred).reshape(-1,1).flatten(),name = "Predicted")
    index = {i: t for t, i in tindex.items()}
    s = []
    for i in xyz:
        s.append(index[i])
    s= pd.Series(s,name = "Actual")
    return s,x
def custom_matrix(xyz,pred):
    #import seaborn as sns
    s,x = serialize(xyz,pred)
    return(pd.crosstab(s,x,margins=True,margins_name = "Total"))
    #print(sns.heatmap(pd.crosstab(s,x,margins=True,margins_name = "Total",normalize = True),annot = False,cmap="YlGnBu",cbar = False))


def custom_accuracy_score(xyz,pred):
    s,x = serialize(xyz,pred)
    return accuracy_score(s,x)



def reverse(pred, index):
    token_seq = []
    for t_seq in pred:
        token = []
        for tag_c in t_seq:
            token.append(index[np.argmax(tag_c)])
        token_seq.append(token)
    return token_seq

def fmasked_accuracy(y_true, y_pred):
        true_id = acc.argmax(y_true, axis=-1)
        pred_id = acc.argmax(y_pred, axis=-1)
        ignore_mask = acc.cast(acc.not_equal(pred_id,0), 'int32')
        tensor = acc.cast(acc.equal(true_id, pred_id), 'int32') * ignore_mask
        accuracy = acc.sum(tensor) / acc.maximum(acc.sum(ignore_mask), 1)
        return accuracy


trainPath = "../assignment2dataset/train.txt"
wholeTrainSet = []
with open(trainPath) as fp: 
    Lines = fp.readlines() 
    sentence = []
    for line in Lines: 
        #print(line)
        lineList = line.split(" ")
        word = lineList[0]
        if word=="\n":
            wholeTrainSet.append(sentence)
            sentence = []
            continue
        tag = lineList[2][0]
        sentence.append((word,tag))


testPath = "../assignment2dataset/test.txt"

wholeTestSet = []
with open(testPath) as fp: 
    Lines = fp.readlines() 
    sentence = []
    for line in Lines: 
        lineList = line.split(" ")
        word = lineList[0]
        if word=="\n":
            wholeTestSet.append(sentence)
            sentence = []
            continue
        tag = lineList[2][0]
        sentence.append((word,tag))
        
        

X_train , y_train = [],[]
for sent in wholeTrainSet:
    sentence, tags = zip(*sent)
    X_train.append(np.array(sentence))
    y_train.append(np.array(tags))


X_test , y_test = [],[]
for sent in wholeTestSet:
    sentence, tags = zip(*sent)
    X_test.append(np.array(sentence))
    y_test.append(np.array(tags))        
words, tags = set([]), set([])    

for sents in X_train:
    for x in sents:      
        words.add(x.lower())

for tsents in y_train:
    for t in tsents:
        tags.add(t)
 
windex = {w: i + 2 for i, w in enumerate(list(words))}
windex['-PADDING-'] = 0
windex['-UNKNOWN-'] = 1
tindex = {t: i + 1 for i, t in enumerate(list(tags))}
tindex['-PADDING-'] = 0

train_X, test_X, train_y, test_y = [], [], [], [] 
for sents in X_train:
    sindex = []
    for x in sents:
        try:
            sindex.append(windex[x.lower()])
        except KeyError:
            sindex.append(windex['-UNKNOWN-'])
    train_X.append(sindex)
    
for sents in X_test:
    sindex = []
    for x in sents:
        try:
            sindex.append(windex[x.lower()])
        except KeyError:
            sindex.append(windex['-UNKNOWN-'])
    test_X.append(sindex)
 
for sents in y_train:
    train_y.append([tindex[t] for t in sents])
 
for sents in y_test:
    test_y.append([tindex[t] for t in sents])
    
MAX_LEN = len(max(train_X,key=len))
Ptrain_sent = pad_sequences(train_X, maxlen=MAX_LEN, padding='post')
Ptest_sent = pad_sequences(test_X, maxlen=MAX_LEN, padding='post')
Ptrain_tag = pad_sequences(train_y, maxlen=MAX_LEN, padding='post')
Ptest_tag = pad_sequences(test_y, maxlen=MAX_LEN, padding='post')
enc_train_tag = onehotencoding(Ptrain_tag, len(tindex))
enc_test_tag = onehotencoding(Ptest_tag, len(tindex))
model = bi_lstm_model()
model.summary()
callback = callbacks.EarlyStopping(monitor='loss')
history = model.fit(Ptrain_sent, enc_train_tag, callbacks=[callback], batch_size=128, epochs=40, validation_split=0.2)
 
predictions = model.predict(Ptest_sent)
pred = reverse(predictions, {i: t for t, i in tindex.items()})
print("\nhistory of accuracy during training : ",history.history['accuracy'])

predictions = model.predict(Ptest_sent)
pred = reverse(predictions, {i: t for t, i in tindex.items()})
cm = custom_matrix(Ptest_tag,pred)
print(cm)


accuracy_s = custom_accuracy_score(Ptest_tag,pred)
s,x = serialize(Ptest_tag,pred)
print("\naccuracy with Paddings : ",accuracy_s)
q= fmasked_accuracy(enc_test_tag,predictions)
print("\naccuracy without paddings : ",float(q))
