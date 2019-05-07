import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics.pairwise import cosine_similarity


# To compare the deep learning based features, term frequency based bag-of-words model features are constructed as follows:

# TODO 
# These parameteres are needed  
updated_train_data, updated_train_owner, updated_test_data, vocabulary , y_test, rankK = [],[],[],[],[],10

train_data = []
for item in updated_train_data:
    train_data.append(' '.join(item))
    
test_data = []
for item in updated_test_data:
    test_data.append(' '.join(item))

vocab_data = []
for item in vocabulary:
    vocab_data.append(item)

# Extract tf based bag of words representation
tfidf_transformer = TfidfTransformer(use_idf=False)
count_vect = CountVectorizer(min_df=1, vocabulary= vocab_data,dtype=np.int32)

train_counts = count_vect.fit_transform(train_data)       
train_feats = tfidf_transformer.fit_transform(train_counts)
print(train_feats.shape)

test_counts = count_vect.transform(test_data)
test_feats = tfidf_transformer.transform(test_counts)
print(test_feats.shape)
print("=======================")
# Four baseline classifiers are built over the bag-of-words features:

# Naive Bayes
# Support Vector Machine
# Cosine similarity
# Softmax classifier
# All the classifiers are implemented using the scikit package of python. The Naive Bayes classifier is implemented as follows:
classifierModel = MultinomialNB(alpha=0.01)        
classifierModel = OneVsRestClassifier(classifierModel).fit(train_feats, updated_train_owner)
predict = classifierModel.predict_proba(test_feats)  
classes = classifierModel.classes_  

accuracy = []
sortedIndices = []
pred_classes = []
for ll in predict:
    sortedIndices.append(sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True))
for k in range(1, rankK+1):
    id = 0
    trueNum = 0
    for sortedInd in sortedIndices:            
        if y_test[id] in classes[sortedInd[:k]]:
            trueNum += 1
            pred_classes.append(classes[sortedInd[:k]])
        id += 1
    accuracy.append((float(trueNum) / len(predict)) * 100)
print(accuracy)
# The implementation of Support Vector Machine is as follows:

classifierModel = svm.SVC(probability=True, verbose=False, decision_function_shape='ovr', random_state=42)
classifierModel.fit(train_feats, updated_train_owner)
predict = classifierModel.predict(test_feats)
classes = classifierModel.classes_ 

accuracy = []
sortedIndices = []
pred_classes = []
for ll in predict:
    sortedIndices.append(sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True))
for k in range(1, rankK+1):
    id = 0
    trueNum = 0
    for sortedInd in sortedIndices:            
        if y_test[id] in classes[sortedInd[:k]]:
            trueNum += 1
            pred_classes.append(classes[sortedInd[:k]])
        id += 1
    accuracy.append((float(trueNum) / len(predict)) * 100)
print(accuracy)
# The implementation of cosine similarity based classification is provided as follows:
trainls = updated_train_owner

predict = cosine_similarity(test_feats, train_feats)
classes = np.array(trainls)
classifierModel = []

accuracy = []
sortedIndices = []
pred_classes = []
for ll in predict:
    sortedIndices.append(sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True))
for k in range(1, rankK+1):
    id = 0
    trueNum = 0
    for sortedInd in sortedIndices:            
        if y_test[id] in classes[sortedInd[:k]]:
            trueNum += 1
            pred_classes.append(classes[sortedInd[:k]])
        id += 1
    accuracy.append((float(trueNum) / len(predict)) * 100)
print(accuracy)
# The softmax (regression) based classification is performed as follows:

classifierModel = LogisticRegression(solver='lbfgs', penalty='l2', tol=0.01)
classifierModel = OneVsRestClassifier(classifierModel).fit(train_feats, updated_train_owner)
predict = classifierModel.predict(test_feats)
classes = classifierModel.classes_ 

accuracy = []
sortedIndices = []
pred_classes = []
for ll in predict:
    sortedIndices.append(sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True))
for k in range(1, rankK+1):
    id = 0
    trueNum = 0
    for sortedInd in sortedIndices:            
        if y_test[id] in classes[sortedInd[:k]]:
            trueNum += 1
            pred_classes.append(classes[sortedInd[:k]])
        id += 1
accuracy.append((float(trueNum) / len(predict)) * 100)
print(accuracy)
