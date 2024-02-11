from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt



def univariate(df_train,df_test):
    df_train = df_train[["Tarih", "Açılış"]]
    df_test = df_test[["Tarih", "Açılış"]]

    df_train.rename(columns={"Açılış": 'y', "Tarih": 'ds'}, inplace=True)
    df_test.rename(columns={"Açılış":'y',"Tarih":'ds'},inplace=True)

    model=Prophet()
    model.fit(df_train)

    forecast=model.predict(df_test)

    plt.plot(df_test['ds'], df_test['y'], color='red', label='actual')
    plt.plot(df_test['ds'], forecast['yhat_upper'], color='blue', label='yhat_upper')
    plt.plot(df_test['ds'], forecast['yhat_lower'], color='yellow', label='yhat_lower')
    plt.plot(df_test['ds'], forecast['yhat'], color='orange', label='Predict')

    plt.savefig('png/Prophet/model_result_univariate.png')
    plt.title('Univariate Forecasting')

    plt.legend()
    plt.show()


def multivariate(df_train,df_test):
    df_train=df_train[["Tarih","Açılış","Yüksek","Düşük"]]
    df_test=df_test[["Tarih","Açılış","Yüksek","Düşük"]]

    df_train.rename(columns={"Açılış": 'y', "Tarih": 'ds'}, inplace=True)
    df_test.rename(columns={"Açılış": 'y', "Tarih": 'ds'}, inplace=True)

    model=Prophet()
    model.add_regressor('Yüksek',standardize=False)
    model.add_regressor('Düşük', standardize=False)

    model.fit(df_train)

    forecast=model.predict(df_test)

    plt.plot(df_test['ds'], df_test['y'], color='red', label='actual')
    plt.plot(df_test['ds'], forecast['yhat_upper'], color='blue', label='yhat_upper')
    plt.plot(df_test['ds'], forecast['yhat_lower'], color='yellow', label='yhat_lower')
    plt.plot(df_test['ds'], forecast['yhat'], color='orange', label='Predict')

    plt.title('Multivariate Forecasting')
    plt.savefig('png/Prophet/model_result_multivariate.png')

    plt.legend()
    plt.show()


def main():
    df_train = pd.read_csv('df_train.csv')
    df_test = pd.read_csv('df_test.csv')

    univariate(df_train,df_test)
    multivariate(df_train,df_test)
main()