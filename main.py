# The entire implementation is done using Python. The complete script to reproduce the results from the entire paper can be downloaded from here.
# Let us walk through the implementation of our approach. The required packages for our implementation are:

# Stanford NLTK
# Gensim for word2vec
# Keras with Tensorflow backend
# Scikit-learn from Python
# The required packages can be imported into python as follows:

import numpy as np
np.random.seed(1337)
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, BatchNormalization, Flatten, Input, RepeatVector, Permute, multiply, Lambda, Activation
from keras.layers.merge import concatenate
from keras.layers.wrappers import Wrapper
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics.pairwise import cosine_similarity


#1. Word2vec parameters
min_word_frequency_word2vec = 5
embed_size_word2vec = 200
context_window_word2vec = 5

#2. Classifier hyperparameters
numCV = 10
max_sentence_len = 50
min_sentence_length = 15
rankK = 10
batch_size = 32

# Load preprocessed data

wordvec_model = Word2Vec.load("./data/chrome/word2vec.model")
vocabulary = wordvec_model.wv.vocab
vocab_size = len(vocabulary)

all_data = np.load("./data/chrome/all_data.npy", allow_pickle=True)
all_owner = np.load("./data/chrome/all_owner.npy", allow_pickle=True)

# The ten times chronological cross validation split is performed as follows:

totalLength = len(all_data)
splitLength = totalLength // (numCV + 1)

for i in range(1, numCV+1):
    train_data = all_data[:i*splitLength-1]
    test_data = all_data[i*splitLength:(i+1)*splitLength-1]
    train_owner = all_owner[:i*splitLength-1]
    test_owner = all_owner[i*splitLength:(i+1)*splitLength-1]
    # For the ith cross validation set, remove all the words that is not present in the vocabulary

    # i = 1 # Denotes the cross validation set number
    updated_train_data = []    
    updated_train_data_length = []    
    updated_train_owner = []
    final_test_data = []
    final_test_owner = []
    for j, item in enumerate(train_data):
        current_train_filter = [word for word in item if word in vocabulary]
        if len(current_train_filter)>=min_sentence_length:  
            updated_train_data.append(current_train_filter)
            updated_train_owner.append(train_owner[j])  
        
    for j, item in enumerate(test_data):
        current_test_filter = [word for word in item if word in vocabulary]  
        if len(current_test_filter)>=min_sentence_length:
            final_test_data.append(current_test_filter)    	  
            final_test_owner.append(test_owner[j])   
    # For the ith cross validation set, remove those classes from the test set, for whom the train data is not available.

    # i = 1 # Denotes the cross validation set number
    # Remove data from test set that is not there in train set
    train_owner_unique = set(updated_train_owner)
    test_owner_unique = set(final_test_owner)
    unwanted_owner = list(test_owner_unique - train_owner_unique)
    updated_test_data = []
    updated_test_owner = []
    updated_test_data_length = []
    for j in range(len(final_test_owner)):
        if final_test_owner[j] not in unwanted_owner:
            updated_test_data.append(final_test_data[j])
            updated_test_owner.append(final_test_owner[j])

    unique_train_label = list(set(updated_train_owner))
    train_label = unique_train_label
    classes = np.array(unique_train_label)
    # Create the data matrix and labels required for the deep learning model training and softmax classifier as follows:

    X_train = np.empty(shape=[len(updated_train_data), max_sentence_len, embed_size_word2vec], dtype='float32')
    Y_train = np.empty(shape=[len(updated_train_owner),1], dtype='int32')
    # 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3 
    for j, curr_row in enumerate(updated_train_data):
        sequence_cnt = 0         
        for item in curr_row:
            if item in vocabulary:
                X_train[j, sequence_cnt, :] = wordvec_model[item] 
                sequence_cnt = sequence_cnt + 1                
                if sequence_cnt == max_sentence_len-1:
                    break                
        for k in range(sequence_cnt, max_sentence_len):
            X_train[j, k, :] = np.zeros((1,embed_size_word2vec))        
        Y_train[j,0] = unique_train_label.index(updated_train_owner[j])

    X_test = np.empty(shape=[len(updated_test_data), max_sentence_len, embed_size_word2vec], dtype='float32')
    Y_test = np.empty(shape=[len(updated_test_owner),1], dtype='int32')
    # 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3 
    for j, curr_row in enumerate(updated_test_data):
        sequence_cnt = 0          
        for item in curr_row:
            if item in vocabulary:
                X_test[j, sequence_cnt, :] = wordvec_model[item] 
                sequence_cnt = sequence_cnt + 1                
                if sequence_cnt == max_sentence_len-1:
                        break                
        for k in range(sequence_cnt, max_sentence_len):
            X_test[j, k, :] = np.zeros((1,embed_size_word2vec))        
        Y_test[j,0] = unique_train_label.index(updated_test_owner[j])
        
    y_train = np_utils.to_categorical(Y_train, len(unique_train_label))
    y_test = np_utils.to_categorical(Y_test, len(unique_train_label))
    # Construct the architecture for deep bidirectional RNN model using Keras library as follows:

    input_1 = Input(shape=(max_sentence_len,embed_size_word2vec), dtype='float32')
    # sequence_embed = Embedding(vocab_size, embed_size_word2vec, input_length=max_sentence_len)(input)

    forwards_1 = LSTM(512, return_sequences=True, dropout=0.2)(input_1)

    attention_1 = Dense(1, activation='tanh')(forwards_1)
    attention_1 = Flatten()(attention_1)
    attention_1 = Activation('softmax')(attention_1)
    attention_1 = RepeatVector(512)(attention_1)
    attention_1 = Permute([2, 1])(attention_1)

    sent_representation_1 = multiply([forwards_1, attention_1])


    after_dp_forward_5 = BatchNormalization()(sent_representation_1)

    backwards_1 = LSTM(512, return_sequences=True, dropout=0.2, go_backwards=True)(input_1)
    
    attention_2 = Dense(1, activation='tanh')(backwards_1)
    attention_2 = Flatten()(attention_2)
    attention_2 = Activation('softmax')(attention_2)
    attention_2 = RepeatVector(512)(attention_2)
    attention_2 = Permute([2, 1])(attention_2)

    sent_representation_2 = multiply([backwards_1, attention_2])

    after_dp_backward_5 = BatchNormalization()(sent_representation_2)
                
    merged = concatenate([after_dp_forward_5, after_dp_backward_5])
    flat = Flatten()(merged)
    after_merge = Dense(1000, activation='relu')(flat)
    after_dp = Dropout(0.4)(after_merge)
    output = Dense(len(train_label), activation='softmax')(after_dp)                
    model = Model(input=input_1, output=output)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])

    model.summary()
    # Train the deep learning model and test using the classifier as follows:

    early_stopping = EarlyStopping(monitor='loss', patience=2)
    hist = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=50, callbacks=[early_stopping])              
    
    predict = model.predict(X_test)        
    accuracy = []
    sortedIndices = []
    pred_classes = []
    for ll in predict:
        sortedIndices.append(sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True))
    for k in range(1, rankK+1):
        id = 0
        trueNum = 0
        for sortedInd in sortedIndices:            
            pred_classes.append(classes[sortedInd[:k]])
            if np.argmax(y_test[id]) in sortedInd[:k]:
                trueNum += 1
            id += 1
        accuracy.append((float(trueNum) / len(predict)) * 100)
    print('Test accuracy:', accuracy)       

    train_result = hist.history        
    print(train_result)

# To compare the deep learning based features, term frequency based bag-of-words model features are constructed as follows:

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
