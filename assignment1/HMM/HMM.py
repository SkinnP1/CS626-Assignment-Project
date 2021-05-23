import nltk
from nltk.corpus import brown
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import pprint,time
import string
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
from nltk.stem.snowball import SnowballStemmer


def word_given_tag(word,tag,train_bag):
    tag_list=[]
    for pair in train_bag:
        if pair[1]==tag:
            tag_list.append(pair)
    count_tag=len(tag_list)
    w_given_tag_list=[]
    for pair in tag_list:
        if pair[0]==word:
            w_given_tag_list.append(pair[0])
    count_w_given_tag=len(w_given_tag_list)
    return(count_w_given_tag,count_tag)



def t2_given_t1(t2,t1,train_bag):
    tags=[pair[1] for pair in train_bag]
    count_t1=len([t for t in tags if t==t1])
    count_t2_t1=0
    for i in range(len(tags)-1):
        if tags[i]==t1 and tags[i+1]==t2:
            count_t2_t1+=1
    return (count_t2_t1,count_t1)


def POS_Tagging_using_HMM(words,train_bag,tags_df):
    state=[]
    T=list(set([pair[1] for pair in train_bag]))
    i=0
    
    for key,word in enumerate(words):
        p=[]
        for tag in T:
            if key==0:
                transition_p=tags_df.loc['.',tag]
            else:
                transition_p=tags_df.loc[state[-1],tag]
            
            t=word_given_tag(words[key],tag,train_bag)[1]
            if t!=0:
                emission_p=word_given_tag(words[key],tag,train_bag)[0]/t
                state_probability=emission_p*transition_p
                p.append(state_probability)

        pmax=max(p)
        state_max=T[p.index(pmax)]
        state.append(state_max)
        
    return list(zip(words,state))



def fnmain(train_set,test_set,tags):

    test_untagged_words=[tup[0] for sent in test_set for tup in sent]

    train_tagged_words=[]
    for sent in train_set:
        for tup in sent:
            train_tagged_words.append(tup)
    test_tagged_words=[]
    for sent in test_set:
        for tup in sent:
            test_tagged_words.append(tup)




    tags_matrix=np.zeros((len(tags),len(tags)),dtype='float32')
    for i,t1 in enumerate(list(tags)):
        for j,t2 in enumerate(list(tags)):
            t=t2_given_t1(t2,t1,train_tagged_words)[1]
            if t!=0:
                tags_matrix[i,j]=t2_given_t1(t2,t1,train_tagged_words)[0]/t


    tags_df=pd.DataFrame(tags_matrix,columns=list(tags),index=list(tags))




    test_tagged_words=[tup for sent in test_set for tup in sent]
    tagged_seq=POS_Tagging_using_HMM(test_untagged_words,train_tagged_words,tags_df)
    y_actual=[tup[1] for sent in test_set for tup in sent]
    y_pred=[tup[1] for tup in tagged_seq]


    check=[i for i , j in zip(tagged_seq,test_tagged_words) if i==j]

    acc=len(check)/len(tagged_seq)
    c_matrix=confusion_matrix(y_actual,y_pred,labels=list(tags))#confusion matrix
    df=pd.DataFrame(c_matrix,columns=list(tags),index=list(tags))
    accPOS=np.zeros(len(tags))#per POS tag accuracy check
    for i in range(len(tags)):
        if sum(c_matrix[i])!=0:
            accPOS[i]=c_matrix[i,i]/sum(c_matrix[i])
    return acc,accPOS

#importing brown corpus 
brown_words = []
normalSentence = []
snow = SnowballStemmer('english')
for sentence in brown.tagged_sents(tagset='universal'):
    updatedSentence = []
    for word in sentence :
        if word[0] not in string.punctuation or word[0]==".":
            stemmedWord = snow.stem(word[0])
            tag = word[1]
            updatedTuple = (stemmedWord,tag)
            updatedSentence.append(updatedTuple)
    brown_words.append(updatedSentence)






tags={word[1] for sent in brown_words for word in sent }
accPOS=np.zeros(len(tags))#per POS tag accuracy check
    
    
a=0
avg=0
from sklearn.model_selection import KFold#creating a five fold split
kf=KFold(n_splits=5)
kf
train_set=[]
test_set=[]
for train_index,test_index in kf.split(brown_words):
    for i in train_index:
        train_set.append(brown_words[i])
    for i in test_index:
        test_set.append(brown_words[i])
    a,arr=fnmain(train_set,test_set,tags)
    avg+=a
    accPOS+=arr

accPOS/=5.0
acc_df=pd.DataFrame(accPOS,index=list(tags))
print("average accuracy per POS tag")
print(acc_df)
print("average accuracy")#average accuracy
avg=avg/5.0
print(avg*100)





