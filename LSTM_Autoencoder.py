from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from keras.layers import Dense, Dropout,RepeatVector,TimeDistributed

import matplotlib.pyplot as plt

import math


def history_visualization(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    # plt.savefig('png/LSTM/model_loss_for_lstm.png')
    plt.show()



def building_autoencoder_model(x_train):
    # Building model
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(256, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, activation='relu', return_sequences=True))

    model.add(LSTM(64, activation='relu', return_sequences=False))
    model.add(RepeatVector(x_train.shape[1]))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(256, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(x_train.shape[2])))

    """model.add(Dropout(0.2))
    model.add(Dense(x_train.shape[1]))"""

    model.compile(optimizer='adam', loss='mse')
    history = model.fit(x_train, x_train, epochs=1, batch_size=64, validation_split=0.1, verbose=1)

    history_visualization(history)

    # save model
    model.save(f'lstm_autoencoder_model.h5')

    print("model fitted and saved")
    return model, history


def df_to_X_for_lstm(df_train, df_test):
    # Prepare train datasets
    window_size = 6
    x_train = []
    n_future = 0  # Number of days we want to look into the future based on the past days.

    for i in range(window_size, len(df_train) - n_future + 1):
        x_train.append(df_train.iloc[i - window_size:i])

    # Prepare test datasets
    x_test = []
    for i in range(window_size, len(df_test) - n_future + 1):
        x_test.append(df_test.iloc[i - window_size:i])

    return np.array(x_train).astype('float32'), np.array(x_test).astype('float32')


def main():
    df_train = pd.read_csv('df_train.csv')
    df_test = pd.read_csv('df_test.csv')

    df_train = df_train["Açılış"]
    df_test = df_test["Açılış"]

    x_train, x_test = df_to_X_for_lstm(df_train, df_test)

    model, history = building_autoencoder_model(x_train)

    #model = load_model('lstm_autoencoder_model.h5')

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    testPredict = model.predict(x_test)

    testPredict = np.reshape(testPredict, (testPredict.shape[0], testPredict.shape[1], 1))

    testPredict = np.mean(testPredict, axis=1)
    df_test = df_test[0:x_test.shape[0]]

    plt.plot(testPredict, label='Predict')
    plt.legend()
    plt.plot(df_test, label='Actual')
    plt.legend()
    plt.title('Predict vs Actual')
    # plt.savefig('png/LSTM/predictVSactual_for_lstm.png')
    plt.show()



main()
