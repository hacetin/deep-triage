Install equired packages

`pip install -U nltk gensim tensorflow keras scikit-learn` 

Download data to "data" folder.

Changed lines:

```python
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge, BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping
current_data = list(filter(None, current_data))
vocabulary = wordvec_model.wv.vocab
splitLength = totalLength // (numCV + 1)
train_label = updated_train_owner
trainls = updated_train_owner
```

Not working yet. Error message:

    Traceback (most recent call last):
      File "/home/alper/workspaces/cs559_project/main.py", line 297, in <module>
        after_dp_forward_5 = BatchNormalization()(attention_1)
      File "/home/alper/anaconda3/envs/cs559_project/lib/python3.7/site-packages/keras/engine/base_layer.py", line 440, in __call__
        self.assert_input_compatibility(inputs)
      File "/home/alper/anaconda3/envs/cs559_project/lib/python3.7/site-packages/keras/engine/base_layer.py", line 311, in assert_input_compatibility
        str(K.ndim(x)))
    ValueError: Input 0 is incompatible with layer batch_normalization_1: expected ndim=3, found ndim=2