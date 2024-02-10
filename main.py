import pandas as pd
import matplotlib.pyplot as plt
import subprocess





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

def LSTM():
    subprocess.run(["python", "LSTM.py"])
def Prophet():
    subprocess.run(["python", "Prophet.py"])

def main():
    # data load from csv
    df = pd.read_csv('usd_eur.csv')

    df = data_preprocessing(df)

    # Split dataframe to df_train and df_test
    df_train = df[df["Tarih"] <= '2023-10-31']
    df_test = df[df["Tarih"] > '2023-10-31']

    df_train.to_csv('df_train.csv')
    df_test.to_csv('df_test.csv')

    #LSTM()
    #Prophet()


main()