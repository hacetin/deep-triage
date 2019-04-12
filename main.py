# The entire implementation is done using Python. The complete script to reproduce the results from the entire paper can be downloaded from here.
# Let us walk through the implementation of our approach. The required packages for our implementation are:

# Stanford NLTK
# Gensim for word2vec
# Keras with Tensorflow backend
# Scikit-learn from Python
# The required packages can be imported into python as follows:

import numpy as np
np.random.seed(1337)
import json, re, nltk, string
from nltk.corpus import wordnet
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge, BatchNormalization, TimeDistributed, InputSpec
from keras.layers.wrappers import Wrapper
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras import backend as K
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics.pairwise import cosine_similarity


# The Soft Attention layer is implemented as follows: 
# Adapted from code written by braingineer

def make_safe(x):
    return K.clip(x, K.common._EPSILON, 1.0 - K.common._EPSILON)

class ProbabilityTensor(Wrapper):
    """ function for turning 3d tensor to 2d probability matrix, which is the set of a_i's """
    def __init__(self, dense_function=None, *args, **kwargs):
        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        #layer = TimeDistributed(dense_function) or TimeDistributed(Dense(1, name='ptensor_func'))
        layer = TimeDistributed(Dense(1, name='ptensor_func'))
        super(ProbabilityTensor, self).__init__(layer, *args, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.input_spec = [InputSpec(shape=input_shape)]
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis.')

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ProbabilityTensor, self).build()

    def get_output_shape_for(self, input_shape):
        # b,n,f -> b,n 
        #       s.t. \sum_n n = 1
        if isinstance(input_shape, (list,tuple)) and not isinstance(input_shape[0], int):
            input_shape = input_shape[0]

        return (input_shape[0], input_shape[1])

    def squash_mask(self, mask):
        if K.ndim(mask) == 2:
            return mask
        elif K.ndim(mask) == 3:
            return K.any(mask, axis=-1)

    def compute_mask(self, x, mask=None):
        if mask is None:
            return None
        return self.squash_mask(mask)

    def call(self, x, mask=None):
        energy = K.squeeze(self.layer(x), 2)
        p_matrix = K.softmax(energy)
        if mask is not None:
            mask = self.squash_mask(mask)
            p_matrix = make_safe(p_matrix * mask)
            p_matrix = (p_matrix / K.sum(p_matrix, axis=-1, keepdims=True))*mask
        return p_matrix

    def get_config(self):
        config = {}
        base_config = super(ProbabilityTensor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SoftAttentionConcat(ProbabilityTensor):
    '''This will create the context vector and then concatenate it with the last output of the LSTM'''
    def get_output_shape_for(self, input_shape):
        # b,n,f -> b,f where f is weighted features summed across n
        return (input_shape[0], 2*input_shape[2])

    def compute_mask(self, x, mask=None):
        if mask is None or mask.ndim==2:
            return None
        else:
            raise Exception("Unexpected situation")

    def call(self, x, mask=None):
        # b,n,f -> b,f via b,n broadcasted
        p_vectors = K.expand_dims(super(SoftAttentionConcat, self).call(x, mask), 2)
        expanded_p = K.repeat_elements(p_vectors, K.int_shape(x)[2], axis=2)
        context = K.sum(expanded_p * x, axis=1)
        last_out = x[:, -1, :]
        return K.concatenate([context, last_out])



# The JSON file location containing the data for deep learning model training and classifier training and testing are provided as follows:

open_bugs_json = './data/chrome/deep_data.json'
closed_bugs_json = './data/chrome/classifier_data_0.json'
# The hyperparameters required for the entire code can be initialized upfront as follows:

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
# The bugs are loaded from the JSON file and the preprocessing is performed as follows:

with open(open_bugs_json) as data_file:
    data = json.load(data_file, strict=False)

all_data = []
for item in data:
    #1. Remove \r 
    current_title = item['issue_title'].replace('\r', ' ')
    current_desc = item['description'].replace('\r', ' ')    
    #2. Remove URLs
    current_desc = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', current_desc)    
    #3. Remove Stack Trace
    start_loc = current_desc.find("Stack trace:")
    current_desc = current_desc[:start_loc]    
    #4. Remove hex code
    current_desc = re.sub(r'(\w+)0x\w+', '', current_desc)
    current_title= re.sub(r'(\w+)0x\w+', '', current_title)    
    #5. Change to lower case
    current_desc = current_desc.lower()
    current_title = current_title.lower()    
    #6. Tokenize
    current_desc_tokens = nltk.word_tokenize(current_desc)
    current_title_tokens = nltk.word_tokenize(current_title)
    #7. Strip trailing punctuation marks    
    current_desc_filter = [word.strip(string.punctuation) for word in current_desc_tokens]
    current_title_filter = [word.strip(string.punctuation) for word in current_title_tokens]      
    #8. Join the lists
    current_data = current_title_filter + current_desc_filter
    current_data = list(filter(None, current_data))
    all_data.append(current_data)  
# A vocabulary is constructed and the word2vec model is learnt using the preprocessed data. The word2vec model provides a semantic word representation for every word in the vocabulary.

wordvec_model = Word2Vec(all_data, min_count=min_word_frequency_word2vec, size=embed_size_word2vec, window=context_window_word2vec)
vocabulary = wordvec_model.wv.vocab
vocab_size = len(vocabulary)
# The data used for training and testing the classifier is loaded and the preprocessing is performed as follows:

with open(closed_bugs_json) as data_file:
    data = json.load(data_file, strict=False)

all_data = []
all_owner = []    
for item in data:
    #1. Remove \r 
    current_title = item['issue_title'].replace('\r', ' ')
    current_desc = item['description'].replace('\r', ' ')
    #2. Remove URLs
    current_desc = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', current_desc)
    #3. Remove Stack Trace
    start_loc = current_desc.find("Stack trace:")
    current_desc = current_desc[:start_loc]
    #4. Remove hex code
    current_desc = re.sub(r'(\w+)0x\w+', '', current_desc)
    current_title= re.sub(r'(\w+)0x\w+', '', current_title)
    #5. Change to lower case
    current_desc = current_desc.lower()
    current_title = current_title.lower()
    #6. Tokenize
    current_desc_tokens = nltk.word_tokenize(current_desc)
    current_title_tokens = nltk.word_tokenize(current_title)
    #7. Strip punctuation marks
    current_desc_filter = [word.strip(string.punctuation) for word in current_desc_tokens]
    current_title_filter = [word.strip(string.punctuation) for word in current_title_tokens]       
    #8. Join the lists
    current_data = current_title_filter + current_desc_filter
    current_data = filter(None, current_data)
    all_data.append(current_data)
    all_owner.append(item['owner'])
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

    train_label = updated_train_owner
    unique_train_label = list(set(updated_train_owner))
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

    input = Input(shape=(max_sentence_len,), dtype='int32')
    sequence_embed = Embedding(vocab_size, embed_size_word2vec, input_length=max_sentence_len)(input)

    forwards_1 = LSTM(1024, return_sequences=True, dropout_U=0.2)(sequence_embed)
    attention_1 = SoftAttentionConcat()(forwards_1)
    after_dp_forward_5 = BatchNormalization()(attention_1)

    backwards_1 = LSTM(1024, return_sequences=True, dropout_U=0.2, go_backwards=True)(sequence_embed)
    attention_2 = SoftAttentionConcat()(backwards_1)
    after_dp_backward_5 = BatchNormalization()(attention_2)
                
    merged = merge([after_dp_forward_5, after_dp_backward_5], mode='concat', concat_axis=-1)
    after_merge = Dense(1000, activation='relu')(merged)
    after_dp = Dropout(0.4)(after_merge)
    output = Dense(len(train_label), activation='softmax')(after_dp)                
    model = Model(input=input, output=output)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])

    # Train the deep learning model and test using the classifier as follows:

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    hist = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=200)              
        
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
            if y_test[id] in classes[sortedInd[:k]]:
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
