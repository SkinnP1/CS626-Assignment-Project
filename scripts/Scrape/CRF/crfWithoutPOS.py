
import nltk
import numpy as np
from sklearn_crfsuite import metrics
from sklearn.metrics import precision_recall_fscore_support



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
    return features

def detectPrefix(word):
    for prefix in prefixesList:
        if word.lower().startswith(prefix) and len(word) - len(prefix) > 1:
            return(prefix)
    return "NO"

def detectSuffix(word):
    for suffix in suffixList:
        if word.lower().endswith(suffix) and len(word) - len(suffix) > 1:
            return(suffix)
    return "NO"

X_train = []
y_train = []
for index,sentence in enumerate(wholeTrainSet) :
   # print(index)
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
        if i == 0 :
            features['start_of_sentence'] = True
            features['next_word'] = sentence[i+1][0]
            features['next_next_word'] = sentence[i+2][0]
        elif i == 1 :
            features['next_word'] = sentence[i+1][0]
            features['next_next_word'] = sentence[i+2][0]
            features['prev_word'] = sentence[i-1][0]
            features['prev_word_chunk'] = sentence[i-1][2]
        elif i == len(sentence) - 2 :
            features['prev_prev_word'] = sentence[i-2][0]
            features['prev_prev_word_chunk'] = sentence[i-2][2]
            features['prev_word_chunk'] = sentence[i-1][2]
            features['prev_word'] = sentence[i-1][0]
            features['next_word'] = sentence[i+1][0]
        elif i == len(sentence) - 1 :
            features['prev_prev_word'] = sentence[i-2][0]
            features['prev_prev_word_chunk'] = sentence[i-2][2]
            features['prev_word_chunk'] = sentence[i-1][2]
            features['prev_word'] = sentence[i-1][0]
        else :
            features['prev_prev_word'] = sentence[i-2][0]
            features['prev_prev_word_chunk'] = sentence[i-2][2]
            features['prev_word_chunk'] = sentence[i-1][2]
            features['prev_word'] = sentence[i-1][0]
            features['next_word'] = sentence[i+1][0]
            features['next_next_word'] = sentence[i+2][0]
        X_train.append([features])
        y_train.append(word[2])

        
        
        

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
        TestSet.append([features])
        

X_train[1]
from sklearn_crfsuite import CRF
 
model = CRF()
model.fit(X_train, y_train)




 
y_pred = model.predict(TestSet)

print("////////// Precision Recall F_Score and Main for Overall using Macro Average /////////////")
print(precision_recall_fscore_support(TestLabel, y_pred,average="macro"))

print()

print("////////// Precision Recall F_Score and Main for B using Macro Average /////////////")
print(precision_recall_fscore_support(TestLabel, y_pred,labels=['B'],average="macro"))

print()

print("////////// Precision Recall F_Score and Main for I using Macro Average /////////////")
print(precision_recall_fscore_support(TestLabel, y_pred,labels=['I'],average="macro"))

print()
print("Confusion Matrix")
print(confusion_matrix(TestLabel, y_pred,labels=['B','I','O']))

print()

print("////////// Precision Recall F_Score and Main for O using Macro Average /////////////")
print(precision_recall_fscore_support(TestLabel, y_pred,labels=['O'],average="macro"))

