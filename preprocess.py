import os, sys
import pandas as pd
from statsmodels.sandbox.tsa.varma import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX

path = ".\\Log\\"
dirs = os.listdir(path)
pd.set_option('display.max_columns', None)
frame = pd.DataFrame()
list_ = []
ar_x =[]
ar_y=[]
ma_x=[]
ma_y=[]
ama_x=[]
ama_y=[]
arima_x=[]
arima_y=[]
sarima_x=[]
sarima_y=[]
var_x=[]
var_y=[]
varmax_x=[]
varmax_y=[]
hwes_x=[]
hwes_y=[]

import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from random import random
import numpy as np
for subdir in dirs:
    subpath = os.listdir(path+subdir)
    for files in subpath:
        if files == "ibmq_ourense.csv":
        # print(path)
        # print(subdir)
        # print(files)
            df = pd.read_csv(path+subdir+"\\"+files,index_col=None,header=0)
            # print(df.iloc[:,2])
            array1 = df.iloc[:,1]
            array2 = df.iloc[:,4]
            data1 = np.array(array1)
            data2 = np.array(array2)
            # print(data)
            # # contrived dataset
            # data = [x + random() for x in range(1, 100)]
            # # fit model
            import warnings
            warnings.simplefilter("ignore")
            model_ar = AutoReg(data1, lags=1)
            model_fit_ar = model_ar.fit()
            # # make prediction

            ar_yhat1 = model_fit_ar.predict(len(data1), len(data1))
            # print(ar_yhat1)
            ar_x.append(ar_yhat1[0])
            model_ar2 = AutoReg(data2, lags=1)
            model_fit_ar2 = model_ar2.fit()
            # # make prediction

            ar_yhat2 = model_fit_ar2.predict(len(data2), len(data2))
            # print(ar_yhat2)
            ar_y.append(ar_yhat2[0])

            model_ma = ARMA(data1, order=(0, 1))
            model_fit_ma = model_ma.fit(disp=False)
            # make prediction
            ma_yhat = model_fit_ma.predict(len(data1), len(data1))
            # print(ma_yhat)
            ma_x.append(ma_yhat)

            model_ma2 = ARMA(data2, order=(0, 1))
            model_fit_ma2 = model_ma2.fit(disp=False)
            # make prediction
            ma_yhat2 = model_fit_ma2.predict(len(data2), len(data2))
            # print(ma_yhat)
            ma_y.append(ma_yhat2)

            # model_ama = ARMA(data1, order=(2, 1))
            # model_fit_ama = model_ama.fit(disp=False)
            # # make prediction
            # ama_yhat = model_fit_ama.predict(len(data1), len(data1))
            # # print(ma_yhat)
            # ama_x.append(ama_yhat)
            #
            # model_ama2 = ARMA(data2, order=(2, 1))
            # model_fit_ama2 = model_ama2.fit(disp=False)
            # # make prediction
            # ama_yhat2 = model_fit_ama2.predict(len(data2), len(data2))
            # # print(ma_yhat)
            # ama_y.append(ama_yhat2)

            # model_arima = ARIMA(data1, order=(1, 1, 1))
            # model_fit_arima = model_arima.fit(disp=False)
            # # make prediction
            # yhat_arima = model_fit_arima.predict(len(data1), len(data1), typ='levels')
            # arima_x.append(yhat_arima)
            #
            # model_arima2 = ARIMA(data2, order=(1, 1, 1))
            # model_fit_arima2 = model_arima2.fit(disp=False)
            # # make prediction
            # yhat_arima2 = model_fit_arima2.predict(len(data2), len(data2), typ='levels')
            # arima_y.append(yhat_arima2)

            # model_sarima = SARIMAX(data1, order=(1, 1, 1), seasonal_order=(1, 1, 1, 1))
            # model_fit_sarima = model_sarima.fit(disp=False)
            #  # make prediction
            # yhat_sarima = model_fit_sarima.predict(len(data1), len(data1))
            # sarima_x.append(yhat_sarima)
            #
            # model_sarima2 = SARIMAX(data2, order=(1, 1, 1), seasonal_order=(1, 1, 1, 1))
            # model_fit_sarima2 = model_sarima2.fit(disp=False)
            # # make prediction
            # yhat_sarima2 = model_fit_sarima2.predict(len(data2), len(data2))
            # sarima_y.append(yhat_sarima2)

            # model_var = VAR(data1)
            # model_fit_var = model_var.fit()
            # # make prediction
            # yhat_var = model_fit_var.forecast(model_fit_var.y, steps=1)
            # var_x.append(yhat_var)
            #
            # model_var2 = VAR(data2)
            # model_fit_var2 = model_var2.fit()
            # # make prediction
            # yhat_var2 = model_fit_var2.forecast(model_fit_var2.y, steps=1)
            # var_y.append(yhat_var2)
            # data = list()
            # for i in range(100):
            #     v1 = random()
            #     v2 = v1 + random()
            #     row = [v1, v2]
            #     data.append(row)
            # data_exog = [x + random() for x in range(100)]
            # model_varmax = VARMAX(data, exog=data_exog, order=(1, 1))
            # model_fit_varmax = model_varmax.fit(disp=False)
            # # make prediction
            # data_exog2 = [[100]]
            # yhat_varmax = model_fit_varmax.forecast(exog=data_exog2)
            # varmax_x.append(yhat_varmax)
            #
            # model_varmax2 = VARMAX(data, exog=data_exog, order=(1, 1))
            # model_fit_varmax2 = model_varmax2.fit(disp=False)
            # # make prediction
            # data_exog2 = [[100]]
            # yhat_varmax2 = model_fit_varmax2.forecast(exog=data_exog2)
            # varmax_y.append(yhat_varmax2)

            # print(data1)
            # print(data2)
            model_hwes = ExponentialSmoothing(data1)
            model_fit_hwes = model_hwes.fit()
            # make prediction
            yhat_hwes = model_fit_hwes.predict(len(data1), len(data1))
            hwes_x.append(yhat_hwes)
            # print(yhat_hwes)
            model_hwes2 = ExponentialSmoothing(data2)
            model_fit_hwes2 = model_hwes2.fit()
            # make prediction
            yhat_hwes2 = model_fit_hwes2.predict(len(data2), len(data2))
            hwes_y.append(yhat_hwes2)
            # print(yhat_hwes2)
        # list_.append(df)

plt.plot(ar_x, ar_y,'ro')
plt.savefig("Autoreg.png")
plt.plot(ma_x, ma_y,'ro')
plt.savefig("ARMA.png")
# plt.plot(ama_x, ama_y,'ro')
# plt.plot(arima_x, arima_y,'ro')
# plt.plot(sarima_x, sarima_y,'ro')
# plt.plot(var_x, var_y,'ro')
# plt.plot(varmax_x, varmax_y,'ro')
plt.plot(hwes_x, hwes_y,'ro')
plt.savefig("ExponentialSmoothing.png")

# frame = pd.concat(list_)
