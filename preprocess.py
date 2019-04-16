import numpy as np
np.random.seed(1337)
import json, re, nltk, string
from nltk.corpus import wordnet
from gensim.models import Word2Vec

# The JSON file location containing the data for deep learning model training and classifier training and testing are provided as follows:

open_bugs_json = './data/chrome/deep_data.json'
closed_bugs_json = './data/chrome/classifier_data_0.json'
# The hyperparameters required for the entire code can be initialized upfront as follows:

#1. Word2vec parameters
min_word_frequency_word2vec = 5
embed_size_word2vec = 200
context_window_word2vec = 5

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

# A vocabulary is constructed and the word2vec model is learnt using the preprocessed data. The word2vec model provides a semantic word representation for every word in the vocabulary.

wordvec_model.save("./data/chrome/word2vec.model")

np.save("./data/chrome/all_data.npy", all_data)
np.save("./data/chrome/all_owner.npy", all_owner)