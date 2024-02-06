import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from keras.layers import Dense, Dropout
from sklearn.metrics import r2_score, mean_squared_error
import math
from tensorflow.keras.models import load_model


# Normalized Root Mean Squared Error (NRMSE)
def normalized_rmse(true_values, predicted_values):
    range_true = max(true_values) - min(true_values)
    rmse = math.sqrt(mean_squared_error(true_values, predicted_values))
    nrmse = rmse / range_true
    return nrmse

def performance_metrics_visualization(y_test, testPredict):
    testPredict_vis = testPredict.reshape(testPredict.shape[0] * testPredict.shape[1], 1)
    y_test_vis = y_test.reshape(y_test.shape[0] * y_test.shape[1], 1)


    r_squared = r2_score(y_test_vis, testPredict_vis)
    print("R-Squared:", r_squared)

    # Mean Squared Error (MSE)
    mse = mean_squared_error(y_test_vis, testPredict_vis)
    print("MSE:", mse)

    # Root Mean Squared Error (RMSE)
    rmse = math.sqrt(mse)
    print("RMSE:", rmse)

    nrmse = normalized_rmse(y_test_vis, testPredict_vis)
    print("NRMSE:", nrmse)

    plt.plot(testPredict_vis[:, 0], label='Predict')
    plt.title('Predict')
    plt.savefig('png/predict.png')
    plt.legend()
    plt.show()

    plt.plot(y_test_vis[:, 0], label='Test')
    plt.title('Actual')
    plt.savefig('png/actual.png')
    plt.legend()
    plt.show()

    plt.plot(testPredict_vis[:, 0], label='Predict')
    plt.legend()
    plt.plot(y_test_vis[:, 0], label='Test')
    plt.legend()
    plt.title('Predict vs Actual')
    plt.savefig('png/predictVSactual.png')
    plt.show()

def lstm_history_visualization(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('png/model_loss.png')
    plt.show()

def building_lstm_model(x_train, y_train):
    # Building model
    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
    model.add(LSTM(64, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(y_train.shape[1]))

    model.compile(optimizer='adam', loss='mse')
    history = model.fit(x_train, y_train, epochs=6, batch_size=16, validation_split=0.1, verbose=1)

    lstm_history_visualization(history)

    # save model
    model.save(f'lstm_model.h5')

    print("model fitted and saved")
    return model,history

def df_to_X_and_y_for_lstm(df_train,df_test):
    # Prepare train datasets
    window_size = 6
    x_train = []
    y_train = []
    n_future = 0  # Number of days we want to look into the future based on the past days.

    for i in range(window_size, len(df_train) - n_future + 1):
        x_train.append(df_train.iloc[i - window_size:i, 0:df_train.shape[1]])
        y_train.append(df_train.iloc[i + n_future - 1:i + n_future, 0])

    # Prepare test datasets
    x_test = []
    y_test = []
    for i in range(window_size, len(df_test) - n_future + 1):
        x_test.append(df_test.iloc[i - window_size:i, 0:df_test.shape[1]])
        y_test.append(df_test.iloc[i + n_future - 1:i + n_future, 0])

    return np.array(x_train).astype('float32'), np.array(x_test).astype('float32'),np.array(y_train).astype('float32'), np.array(y_test).astype('float32')
def plot_train_and_test(df_train,df_test):
    # Visualize dependent variable for df_test and df_train
    plt.plot(df_test["Tarih"], df_test["Açılış"])
    plt.title('Test Data USD-EUR')
    plt.savefig('png/test_data.png')
    plt.show()

    plt.plot(df_train["Tarih"], df_train["Açılış"])
    plt.title('Train Data USD-EUR')
    plt.savefig('png/train_data.png')
    plt.show()

def data_preprocessing(df):
    # Columns what we need are selected
    df = df[["Tarih", "Açılış"]]

    # convert datetime to date column
    df["Tarih"] = pd.to_datetime(df["Tarih"], format='%d.%m.%Y')

    # Convert float dependent variable
    df["Açılış"] = df["Açılış"].str.replace(',', '.')
    df["Açılış"] = df["Açılış"].astype(float)

    # Sort values ascending according to Tarih column
    df = df.sort_values(by='Tarih', ascending=True)

    # Reset index
    df = df.reset_index(drop=True)

    return df

def main():
    # data load from csv
    df = pd.read_csv('usd_eur.csv')

    df = data_preprocessing(df)

    # Split dataframe to df_train and df_test
    df_train = df[df["Tarih"] <= '2023-10-31']
    df_test = df[df["Tarih"] > '2023-10-31']

    plot_train_and_test(df_train, df_test)

    df_train = df_train.set_index("Tarih")
    df_test = df_test.set_index("Tarih")

    x_train,x_test,y_train,y_test=df_to_X_and_y_for_lstm(df_train, df_test)

    model , history = building_lstm_model(x_train, y_train)
    #model = load_model('lstm_model.h5')


    # Predict and visualize model performance
    testPredict = model.predict(x_test)


    performance_metrics_visualization(y_test,testPredict)

main()