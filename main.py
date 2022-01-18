import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pydotplus

from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dense, Activation, Dropout, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split


df = pd.read_csv('serbian_names.csv', sep=',')
df = df[['Gender', 'Forename']].drop_duplicates()
df.columns = ['gender', 'name']

names = df['name'].apply(lambda x: x.lower())
gender = df['gender']

df.head()

maxlen = 20
labels = 2

print("Male : " + str(sum(gender == 'M')))
print("Female : " + str(sum(gender == 'F')))

vocab = set(' '.join([str(i) for i in names]))
vocab.add('END')
len_vocab = len(vocab)

# print(vocab)
# print(len_vocab)

char_index = dict((c, i) for i, c in enumerate(vocab))
print(char_index)


# Builds an empty line with a 1 at the index of character
def set_flag(i):
    tmp = np.zeros(len_vocab)
    tmp[i] = 1
    return list(tmp)


# Truncate names and create the matrix
def prepare_X(X):
    new_list = []
    trunc_train_name = [str(i)[0:maxlen] for i in X]

    for i in trunc_train_name:
        tmp = [set_flag(char_index[j]) for j in str(i)]
        for k in range(0, maxlen - len(str(i))):
            tmp.append(set_flag(char_index["END"]))
        new_list.append(tmp)

    return new_list


X = prepare_X(names.values)


# Label Encoding of y
def prepare_y(y):
    new_list = []
    for i in y:
        if i == 'M':
            new_list.append([1, 0])
        else:
            new_list.append([0, 1])

    return new_list


y = prepare_y(gender)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Building the model
model = Sequential()
model.add(Bidirectional(
    LSTM(512, return_sequences=True),
    backward_layer=LSTM(512, return_sequences=True, go_backwards=True),
    input_shape=(maxlen, len_vocab))
)
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(512)))
model.add(Dropout(0.2))
model.add(Dense(2, activity_regularizer=l2(0.002)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

plot_model(model, to_file='model_2.png', show_shapes=True, expand_nested=True)

callback = EarlyStopping(monitor='val_loss', patience=5)
mc = ModelCheckpoint('best_model_9.h5', monitor='val_loss', mode='min', verbose=1)
reduce_lr_acc = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=2, verbose=1, min_delta=1e-4, mode='max')

batch_size = 256
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=35, verbose=1, validation_data=(X_test, y_test),
                    callbacks=[callback, mc, reduce_lr_acc])

plt.figure(figsize=(12, 8))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

new_names = ["Marika", "Ozren", "Milkica", "Maca", "Spas"]
X_pred = prepare_X([e.lower() for e in new_names])
prediction = model.predict(X_pred)
print(prediction)
