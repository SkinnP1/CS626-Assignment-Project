import nltk
import numpy as np
import pickle
import os
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


trainPath = "../assignment2dataset/train.txt"

wholeTrainSet = []
with open(trainPath) as fp: 
    Lines = fp.readlines() 
    sentence = []
    for line in Lines: 
        lineList = line.split(" ")
        word = lineList[0]
        if word=="\n":
            wholeTrainSet.append(sentence)
            sentence = []
            continue
        posTag = lineList[1]
        chunkLabel = lineList[2][0]
        sentence.append((word,posTag,chunkLabel))

prefixesList = ["under", "fore", "mid", "mis", "over", "auto","super","up","uni","un","tri","trans","tele","sym","syn","sub","pre","pro","post","omni","non","mono",
                "micro","macro","intro",
                "intra","inter", "in", "il", "im", "ir", "hyper", "homo", "homeo", "hetero", "extra", "ex", "en",
                "dis", "de", "contra", "contro", "com", "con", "co", "circum", "auto", "anti", "ante", "an"
               ]
prefixesList = sorted(prefixesList,key= lambda word : (-len(word)))

suffixList = []
suffixPath = "./suffix.txt"
with open(suffixPath) as fp: 
    Lines = fp.readlines() 
    for line in Lines: 
        lineList = line.split("\n")
        suffixList.append(lineList[0].lower())
suffixes = list(set(suffixList))
suffixList = sorted(suffixList,key= lambda word : (-len(word)))


######### Local function Creating Features
def defaultFeatureValues() :
    features = {}
    for i in prefixesList :
        features[i] = False
    for i in suffixList :
        features[i] = False 
    features['prev_prev_word'] = "NA"
    features['prev_word'] = "NA"
    features['word'] = "NA"
    features['next_word'] = "NA"
    features['next_next_word'] = "NA"
    features['capitalization'] = False
    features['start_of_sentence'] = False
    features['prev_prev_word_chunk'] = "NA"
    features['prev_word_chunk'] = "NA"
    features['prev_prev_word_tag'] = "NA"
    features['prev_word_tag'] = "NA"
    features['word_tag'] = "NA"
    features['next_word_tag'] = "NA"
    features['next_next_word_tag'] = "NA"
    return features

############# Local function for detecting prefixes
def detectPrefix(word):
    for prefix in prefixesList:
        if word.lower().startswith(prefix) and len(word) - len(prefix) > 1:
            return(prefix)
    return "NO"



############ Local function for detecting suffixes
def detectSuffix(word):
    for suffix in suffixList:
        if word.lower().endswith(suffix) and len(word) - len(suffix) > 1:
            return(suffix)
    return "NO"

############# Converting Data to Train Set 
TrainSet = []
for index,sentence in enumerate(wholeTrainSet) :
    if len(sentence) < 5 :
        continue
    for i in range(len(sentence)):
        word = sentence[i]
        features = defaultFeatureValues()
        prefix = detectPrefix(word[0])
        suffix = detectSuffix(word[0])
        if prefix != "NO":
            features[prefix] = True
        if suffix != "NO":
            features[suffix] = True
        features['capitalization'] = word[0][0].isupper()
        features['word'] = word[0]
        features['word_tag'] = word[1]
        if i == 0 :
            features['start_of_sentence'] = True
            features['next_word'] = sentence[i+1][0]
            features['next_next_word'] = sentence[i+2][0]
            features['next_word_tag'] = sentence[i+1][1]
            features['next_next_word_tag'] = sentence[i+2][1]
        elif i == 1 :
            features['next_word'] = sentence[i+1][0]
            features['next_next_word'] = sentence[i+2][0]
            features['next_word_tag'] = sentence[i+1][1]
            features['next_next_word_tag'] = sentence[i+2][1]
            features['prev_word'] = sentence[i-1][0]
            features['prev_word_tag'] = sentence[i-1][1]
            features['prev_word_chunk'] = sentence[i-1][2]
        elif i == len(sentence) - 2 :
            features['prev_prev_word'] = sentence[i-2][0]
            features['prev_prev_word_tag'] = sentence[i-2][1]
            features['prev_prev_word_chunk'] = sentence[i-2][2]
            features['prev_word_chunk'] = sentence[i-1][2]
            features['prev_word'] = sentence[i-1][0]
            features['prev_word_tag'] = sentence[i-1][1]
            features['next_word'] = sentence[i+1][0]
            features['next_word_tag'] = sentence[i+1][1]
        elif i == len(sentence) - 1 :
            features['prev_prev_word'] = sentence[i-2][0]
            features['prev_prev_word_tag'] = sentence[i-2][1]
            features['prev_prev_word_chunk'] = sentence[i-2][2]
            features['prev_word_chunk'] = sentence[i-1][2]
            features['prev_word'] = sentence[i-1][0]
            features['prev_word_tag'] = sentence[i-1][1]
        else :
            features['prev_prev_word'] = sentence[i-2][0]
            features['prev_prev_word_tag'] = sentence[i-2][1]
            features['prev_prev_word_chunk'] = sentence[i-2][2]
            features['prev_word_chunk'] = sentence[i-1][2]
            features['prev_word'] = sentence[i-1][0]
            features['prev_word_tag'] = sentence[i-1][1]
            features['next_word'] = sentence[i+1][0]
            features['next_word_tag'] = sentence[i+1][1]
            features['next_next_word'] = sentence[i+2][0]
            features['next_next_word_tag'] = sentence[i+2][1]
        TrainSet.append((features,word[2]))


###### Checking if model already present 
if not os.path.isfile("./my_classifier_withPOS.pickle"):
    maxent_classifier = nltk.classify.MaxentClassifier.train(TrainSet, max_iter=30)
    f = open("my_classifier_withPOS.pickle", "wb")
    pickle.dump(maxent_classifier , f)
    f.close() 


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
        posTag = lineList[1]
        tag = lineList[2][0]
        sentence.append((word,posTag,tag))

TestSet = [] 
TestLabel = []
for index,sentence in enumerate(wholeTestSet) :
    if len(sentence) < 5 :
        continue
    for i in range(len(sentence)):
        word = sentence[i]
        features = defaultFeatureValues()
        prefix = detectPrefix(word[0])
        suffix = detectSuffix(word[0])
        if prefix != "NO":
            features[prefix] = True
        if suffix != "NO":
            features[suffix] = True
        features['capitalization'] = word[0][0].isupper()
        features['word'] = word[0]
        features['word_tag'] = word[1]
        if i == 0 :
            features['start_of_sentence'] = True
            features['next_word'] = sentence[i+1][0]
            features['next_next_word'] = sentence[i+2][0]
            features['next_word_tag'] = sentence[i+1][1]
            features['next_next_word_tag'] = sentence[i+2][1]
        elif i == 1 :
            features['next_word'] = sentence[i+1][0]
            features['next_next_word'] = sentence[i+2][0]
            features['next_word_tag'] = sentence[i+1][1]
            features['next_next_word_tag'] = sentence[i+2][1]
            features['prev_word'] = sentence[i-1][0]
            features['prev_word_tag'] = sentence[i-1][1]
            features['prev_word_chunk'] = sentence[i-1][2]
        elif i == len(sentence) - 2 :
            features['prev_prev_word'] = sentence[i-2][0]
            features['prev_prev_word_tag'] = sentence[i-2][1]
            features['prev_prev_word_chunk'] = sentence[i-2][2]
            features['prev_word_chunk'] = sentence[i-1][2]
            features['prev_word'] = sentence[i-1][0]
            features['prev_word_tag'] = sentence[i-1][1]
            features['next_word'] = sentence[i+1][0]
            features['next_word_tag'] = sentence[i+1][1]
        elif i == len(sentence) - 1 :
            features['prev_prev_word'] = sentence[i-2][0]
            features['prev_prev_word_tag'] = sentence[i-2][1]
            features['prev_prev_word_chunk'] = sentence[i-2][2]
            features['prev_word_chunk'] = sentence[i-1][2]
            features['prev_word'] = sentence[i-1][0]
            features['prev_word_tag'] = sentence[i-1][1]
        else :
            features['prev_prev_word'] = sentence[i-2][0]
            features['prev_prev_word_tag'] = sentence[i-2][1]
            features['prev_prev_word_chunk'] = sentence[i-2][2]
            features['prev_word_chunk'] = sentence[i-1][2]
            features['prev_word'] = sentence[i-1][0]
            features['prev_word_tag'] = sentence[i-1][1]
            features['next_word'] = sentence[i+1][0]
            features['next_word_tag'] = sentence[i+1][1]
            features['next_next_word'] = sentence[i+2][0]
            features['next_next_word_tag'] = sentence[i+2][1]
        TestLabel.append(word[2])
        TestSet.append(features)


loaded_model = pickle.load(open("my_classifier_withPOS.pickle", 'rb'))
predicted = [loaded_model.classify(i) for i in TestSet]

print("////////// Precision Recall F_Score and Main for Overall using Macro Average /////////////")
print(precision_recall_fscore_support(TestLabel, predicted,average="macro"))

print()

print("////////// Precision Recall F_Score and Main for B using Macro Average /////////////")
print(precision_recall_fscore_support(TestLabel, predicted,labels=['B'],average="macro"))

print()

print("////////// Precision Recall F_Score and Main for I using Macro Average /////////////")
print(precision_recall_fscore_support(TestLabel, predicted,labels=['I'],average="macro"))

print()
print("Confusion Matrix")
print(confusion_matrix(TestLabel, predicted,labels=['B','I','O']))