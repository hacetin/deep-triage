import numpy as np
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import (
    Dense,
    Dropout,
    Embedding,
    LSTM,
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
from dataset import (
    google_chromium_chronological_cv,
    mozilla_core_chronological_cv,
    mozilla_firefox_chronological_cv,
)

np.random.seed(1337)


def dnrnna_model(input_shape, num_output, num_lstm_unit=512, num_dense_unit=1000):
    """Construct the architecture for deep bidirectional RNN model using Keras library"""

    input_1 = Input(shape=input_shape, dtype="float32")

    forwards_1 = LSTM(num_lstm_unit, return_sequences=True, dropout=0.2)(input_1)

    attention_1 = Dense(1, activation="tanh")(forwards_1)
    attention_1 = Flatten()(attention_1)  # squeeze (None,50,1)->(None,50)
    attention_1 = Activation("softmax")(attention_1)
    attention_1 = RepeatVector(num_lstm_unit)(attention_1)
    attention_1 = Permute([2, 1])(attention_1)
    attention_1 = multiply([forwards_1, attention_1])
    attention_1 = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(512,))(
        attention_1
    )

    last_out_1 = Lambda(lambda xin: xin[:, -1, :])(forwards_1)
    sent_representation_1 = concatenate([last_out_1, attention_1])

    after_dp_forward_5 = BatchNormalization()(sent_representation_1)

    backwards_1 = LSTM(
        num_lstm_unit, return_sequences=True, dropout=0.2, go_backwards=True
    )(input_1)

    attention_2 = Dense(1, activation="tanh")(backwards_1)
    attention_2 = Flatten()(attention_2)
    attention_2 = Activation("softmax")(attention_2)
    attention_2 = RepeatVector(num_lstm_unit)(attention_2)
    attention_2 = Permute([2, 1])(attention_2)
    attention_2 = multiply([backwards_1, attention_2])
    attention_2 = Lambda(lambda xin: K.sum(xin, axis=1))(attention_2)

    last_out_2 = Lambda(lambda xin: xin[:, -1, :])(backwards_1)
    sent_representation_2 = concatenate([last_out_2, attention_2])

    after_dp_backward_5 = BatchNormalization()(sent_representation_2)

    merged = concatenate([after_dp_forward_5, after_dp_backward_5])
    # flat = Flatten()(merged)
    after_merge = Dense(num_dense_unit, activation="relu")(merged)
    after_dp = Dropout(0.4)(after_merge)
    output = Dense(num_output, activation="softmax")(after_dp)
    model = Model(input=input_1, output=output)
    model.compile(
        loss="categorical_crossentropy", optimizer=Adam(lr=1e-4), metrics=["accuracy"]
    )

    model.summary()

    return model


def run_dbrnna_chronological_cv(dataset_name, min_train_samples_per_class, num_cv):
    """ Chronological cross validation for DBRNN-A model

        # Example
        ```python
            run_dbrnna_chronological_cv("google_chromium", 0, 10)
        ```
        # Arguments
        dataset_name: Available datasets  are "google_chromium", "mozilla_core", "mozilla_firefox"
        min_train_samples_per_class: This is a dataet parameter, and needs to be one of 0, 5, 10 and 20
        num_cv: Number of chronological cross validation
    """

    if min_train_samples_per_class not in [0, 5, 10, 20]:
        print("Wrong min train samples per class")
        return

    if num_cv < 2:
        print("Wrong number of chronological cross validation (num_cv)")
        return

    # Word2vec parameters
    embed_size_word2vec = 200

    # Classifier hyperparameters
    max_sentence_len = 50
    rank_k = 10
    batch_size = 2048

    slices = None
    if dataset_name == "google_chromium":
        slices = google_chromium_chronological_cv(min_train_samples_per_class, num_cv)
    elif dataset_name == "mozilla_core":
        slices = mozilla_core_chronological_cv(min_train_samples_per_class, num_cv)
    elif dataset_name == "mozilla_firefox":
        slices = mozilla_firefox_chronological_cv(min_train_samples_per_class, num_cv)
    else:
        print("Wrong dataset name")
        return

    slice_results = {}
    for i, (X_train, y_train, X_test, y_test, classes) in enumerate(slices):
        model = dnrnna_model((max_sentence_len, embed_size_word2vec), len(classes))

        # Train the deep learning model and test using the classifier
        early_stopping = EarlyStopping(monitor="val_loss", patience=2)
        hist = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            batch_size=batch_size,
            epochs=500,
            callbacks=[early_stopping],
        )

        predict = model.predict(X_test)
        accuracy = []
        sortedIndices = []
        pred_classes = []
        for ll in predict:
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
            accuracy.append((float(trueNum) / len(predict)) * 100)

        train_result = hist.history
        train_result["test_topk_accuracies"] = accuracy
        slice_results[i] = train_result

    return slice_results
