import numpy as np
import json, re, nltk, string
from nltk.corpus import wordnet
from gensim.models import Word2Vec

np.random.seed(1337)


def clean_word_list(item):
    # 1. Remove \r
    current_title = item["issue_title"].replace("\r", " ")
    current_desc = item["description"].replace("\r", " ")
    # 2. Remove URLs
    current_desc = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "",
        current_desc,
    )
    # 3. Remove Stack Trace
    start_loc = current_desc.find("Stack trace:")
    current_desc = current_desc[:start_loc]
    # 4. Remove hex code
    current_desc = re.sub(r"(\w+)0x\w+", "", current_desc)
    current_title = re.sub(r"(\w+)0x\w+", "", current_title)
    # 5. Change to lower case
    current_desc = current_desc.lower()
    current_title = current_title.lower()
    # 6. Tokenize
    current_desc_tokens = nltk.word_tokenize(current_desc)
    current_title_tokens = nltk.word_tokenize(current_title)
    # 7. Strip trailing punctuation marks
    current_desc_filter = [
        word.strip(string.punctuation) for word in current_desc_tokens
    ]
    current_title_filter = [
        word.strip(string.punctuation) for word in current_title_tokens
    ]
    # 8. Join the lists
    current_data = current_title_filter + current_desc_filter
    current_data = [x for x in current_data if x]  # list(filter(None, current_data))

    return current_data


def preprocess_dataset(dataset_name):
    print("Preprocessing {0} dataset: Start".format(dataset_name))
    # The JSON file location containing the data for deep learning model training
    open_bugs_json = "./data/{0}/deep_data.json".format(dataset_name)

    # Word2vec parameters
    min_word_frequency_word2vec = 5
    embed_size_word2vec = 200
    context_window_word2vec = 5

    # The bugs are loaded from the JSON file and the preprocessing is performed

    with open(open_bugs_json) as data_file:
        text = data_file.read()
        # Fix json files for mozilla core and mozilla firefox
        text = text.replace('" : NULL', '" : "NULL"')
        data = json.loads(text, strict=False)

    all_data = []
    for item in data:
        current_data = clean_word_list(item)
        all_data.append(current_data)

    print("Preprocessing {0} dataset: Word2Vec model".format(dataset_name))
    # A vocabulary is constructed and the word2vec model is learnt using the preprocessed data. The word2vec model provides a semantic word representation for every word in the vocabulary.
    wordvec_model = Word2Vec(
        all_data,
        min_count=min_word_frequency_word2vec,
        size=embed_size_word2vec,
        window=context_window_word2vec,
    )

    # Save word2vec model to use in the model again and again
    wordvec_model.save("./data/{0}/word2vec.model".format(dataset_name))

    # The data used for training and testing the classifier is loaded and the preprocessing is performed
    for min_train_samples_per_class in [0, 5, 10, 20]:
        print(
            "Preprocessing {0} dataset: Classifier data {1}".format(
                dataset_name, min_train_samples_per_class
            )
        )
        closed_bugs_json = "./data/{0}/classifier_data_{1}.json".format(
            dataset_name, min_train_samples_per_class
        )

        with open(closed_bugs_json) as data_file:
            text = data_file.read()
            # Fix json files for mozilla core and mozilla firefox
            text = text.replace('" : NULL', '" : "NULL"')
            data = json.loads(text, strict=False)

        all_data = []
        all_owner = []
        for item in data:
            current_data = clean_word_list(item)
            all_data.append(current_data)
            all_owner.append(item["owner"])

        # Save all data arrays to use in the model again and again
        np.save(
            "./data/{0}/all_data_{1}.npy".format(
                dataset_name, min_train_samples_per_class
            ),
            all_data,
        )
        np.save(
            "./data/{0}/all_owner_{1}.npy".format(
                dataset_name, min_train_samples_per_class
            ),
            all_owner,
        )


def preprocess_all_datasets():
    preprocess_dataset("google_chromium")
    preprocess_dataset("mozilla_core")
    preprocess_dataset("mozilla_firefox")


def preprocess_datasets_combined(dataset_name1, dataset_name2):
    print(
        "Preprocessing {0} and {1} dataset: Start".format(dataset_name1, dataset_name2)
    )
    # The JSON file location containing the data for deep learning model training
    open_bugs_json1 = "./data/{0}/deep_data.json".format(dataset_name1)
    open_bugs_json2 = "./data/{0}/deep_data.json".format(dataset_name2)

    # Word2vec parameters
    min_word_frequency_word2vec = 5
    embed_size_word2vec = 200
    context_window_word2vec = 5

    # The bugs are loaded from the JSON file and the preprocessing is performed

    with open(open_bugs_json1) as data_file:
        text = data_file.read()
        # Fix json files for mozilla core and mozilla firefox
        text = text.replace('" : NULL', '" : "NULL"')
        data1 = json.loads(text, strict=False)

    with open(open_bugs_json2) as data_file:
        text = data_file.read()
        # Fix json files for mozilla core and mozilla firefox
        text = text.replace('" : NULL', '" : "NULL"')
        data2 = json.loads(text, strict=False)

    merged_data = data1 + data2
    all_data = [clean_word_list(item) for item in merged_data]

    print(
        "Preprocessing {0} and {1} dataset: Word2Vec model".format(
            dataset_name1, dataset_name2
        )
    )
    # A vocabulary is constructed and the word2vec model is learnt using the preprocessed data. The word2vec model provides a semantic word representation for every word in the vocabulary.
    wordvec_model = Word2Vec(
        all_data,
        min_count=min_word_frequency_word2vec,
        size=embed_size_word2vec,
        window=context_window_word2vec,
    )

    # Save word2vec model to use in the model again and again
    sorted_dnames = sorted([dataset_name1, dataset_name2])
    wordvec_model.save(
        "./data/combined/word2vec_{0}_{1}.model".format(
            sorted_dnames[0], sorted_dnames[1]
        )
    )


def preprocess_all_datasets_combined():
    preprocess_datasets_combined("google_chromium", "mozilla_core")
    preprocess_datasets_combined("google_chromium", "mozilla_firefox")
    preprocess_datasets_combined("mozilla_core", "mozilla_firefox")
