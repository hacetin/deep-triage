import numpy as np
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import (
    Dense,
    Dropout,
    Embedding,
    LSTM,
    GRU,
    BatchNormalization,
    Flatten,
    Input,
    RepeatVector,
    Permute,
    multiply,
    Lambda,
    Activation,
)
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import backend as K
from dataset import chronological_cv

np.random.seed(1337)


def dnrnna_model(
    input_shape, num_output, num_rnn_unit=512, num_dense_unit=1000, rnn_type="gru"
):
    """ Deep bidirectional RNN model using Keras library
        
        # Example
        ```python
            dnrnna_model((50, 200), 1061)
        ```
        # Arguments
        input_shape: Tuple for model input as (max_sentence_len, embed_size_word2vec)
        num_output: Number of unique labels in the data
        num_rnn_unit: Number of rnn units
        num_dense_unit: Number of dense layer units
        rnn_type: One of "lstm" and "gru"
    """
    if rnn_type not in ["lstm", "gru"]:
        print("Wrong RNN type.")
        return

    input_1 = Input(shape=input_shape, dtype="float32")

    if rnn_type == "lstm":
        forwards_1 = LSTM(num_rnn_unit, return_sequences=True, dropout=0.2)(input_1)
    else:
        forwards_1 = GRU(num_rnn_unit, return_sequences=True, dropout=0.2)(input_1)
    attention_1 = Dense(1, activation="tanh")(forwards_1)
    attention_1 = Flatten()(attention_1)  # squeeze (None,50,1)->(None,50)
    attention_1 = Activation("softmax")(attention_1)
    attention_1 = RepeatVector(num_rnn_unit)(attention_1)
    attention_1 = Permute([2, 1])(attention_1)
    attention_1 = multiply([forwards_1, attention_1])
    attention_1 = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(num_rnn_unit,))(
        attention_1
    )

    last_out_1 = Lambda(lambda xin: xin[:, -1, :])(forwards_1)
    sent_representation_1 = concatenate([last_out_1, attention_1])

    after_dp_forward_5 = BatchNormalization()(sent_representation_1)
    if rnn_type == "lstm":
        backwards_1 = LSTM(
            num_rnn_unit, return_sequences=True, dropout=0.2, go_backwards=True
        )(input_1)
    else:
        backwards_1 = GRU(
            num_rnn_unit, return_sequences=True, dropout=0.2, go_backwards=True
        )(input_1)

    attention_2 = Dense(1, activation="tanh")(backwards_1)
    attention_2 = Flatten()(attention_2)
    attention_2 = Activation("softmax")(attention_2)
    attention_2 = RepeatVector(num_rnn_unit)(attention_2)
    attention_2 = Permute([2, 1])(attention_2)
    attention_2 = multiply([backwards_1, attention_2])
    attention_2 = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(num_rnn_unit,))(
        attention_2
    )

    last_out_2 = Lambda(lambda xin: xin[:, -1, :])(backwards_1)
    sent_representation_2 = concatenate([last_out_2, attention_2])

    after_dp_backward_5 = BatchNormalization()(sent_representation_2)

    merged = concatenate([after_dp_forward_5, after_dp_backward_5])
    after_merge = Dense(num_dense_unit, activation="relu")(merged)
    after_dp = Dropout(0.4)(after_merge)
    output = Dense(num_output, activation="softmax")(after_dp)
    model = Model(input=input_1, output=output)
    model.compile(
        loss="categorical_crossentropy", optimizer=Adam(lr=1e-4), metrics=["accuracy"]
    )

    # model.summary()

    return model


def topk_accuracy(prediction, y_test, classes, rank_k=10):
    accuracy = []
    sortedIndices = []
    pred_classes = []
    for ll in prediction:
        sortedIndices.append(
            sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True)
        )
    for k in range(1, rank_k + 1):
        id = 0
        trueNum = 0
        for sortedInd in sortedIndices:
            pred_classes.append(classes[sortedInd[:k]])
            if np.argmax(y_test[id]) in sortedInd[:k]:
                trueNum += 1
            id += 1
        accuracy.append((float(trueNum) / len(prediction)) * 100)

    return accuracy


def run_dbrnna_chronological_cv(
    dataset_name,
    min_train_samples_per_class,
    num_cv,
    rnn_type="lstm",
    merged_wordvec_model=False,
):
    """ Chronological cross validation for DBRNN-A model

        # Example
        ```python
            run_dbrnna_chronological_cv("google_chromium", 0, 10)
        ```
        # Arguments
        dataset_name: Available datasets  are "google_chromium", "mozilla_core", "mozilla_firefox"
        min_train_samples_per_class: This is a dataet parameter, and needs to be one of 0, 5, 10 and 20
        num_cv: Number of chronological cross validation
        rnn_type: RNN model to use in keras model, one of "lstm" and "gru"
        merged_wordvec_model: If `True`, use open bugs from all datasets 
    """

    if min_train_samples_per_class not in [0, 5, 10, 20]:
        print("Wrong min train samples per class")
        return

    if num_cv < 1:
        print("Wrong number of chronological cross validation (num_cv)")
        return

    if dataset_name not in ["google_chromium", "mozilla_core", "mozilla_firefox"]:
        print("Wrong dataset name")
        return

    # Word2vec parameters
    embed_size_word2vec = 200

    # Classifier hyperparameters
    max_sentence_len = 50
    rank_k = 10
    batch_size = 2048
    if rnn_type == "gru":
        batch_size = int(batch_size * 1.5)

    slices = chronological_cv(
        dataset_name, min_train_samples_per_class, num_cv, merged_wordvec_model
    )

    slice_results = {}
    top_rank_k_accuracies = []
    for i, (X_train, y_train, X_test, y_test, classes) in enumerate(slices):
        model = dnrnna_model(
            (max_sentence_len, embed_size_word2vec), len(classes), rnn_type=rnn_type
        )

        # Train the deep learning model and test using the classifier
        early_stopping = EarlyStopping(monitor="val_loss", patience=3)
        hist = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            batch_size=batch_size,
            epochs=500,
            callbacks=[early_stopping],
        )

        prediction = model.predict(X_test)
        accuracy = topk_accuracy(prediction, y_test, classes, rank_k=rank_k)
        print("CV{}, top1 - ... - top{} accuracy: ".format(i + 1, rank_k), accuracy)

        train_result = hist.history
        train_result["test_topk_accuracies"] = accuracy
        slice_results[i + 1] = train_result
        top_rank_k_accuracies.append(accuracy[-1])
    
    print("Top{0} accuracies for all CVs: {1}".format(rank_k, top_rank_k_accuracies))
    print("Average top{0} accuracy: {1}".format(rank_k, sum(top_rank_k_accuracies)/rank_k))
    return slice_results
