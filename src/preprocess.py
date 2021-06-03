import os

import pandas as pd

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
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
for subdir in dirs:
    subpath = os.listdir(path+subdir)
    for files in subpath:
        if files == "ibmq_ourense.csv":
            df = pd.read_csv(path+subdir+"\\"+files,index_col=None,header=0)
            # print(df.iloc[:,2])
            array1 = df.iloc[:,1]
            array2 = df.iloc[:,4]
            data1 = np.array(array1)
            data2 = np.array(array2)
            import warnings
            warnings.simplefilter("ignore")
            model_ar = AutoReg(data1, lags=1)
            model_fit_ar = model_ar.fit()

            ar_yhat1 = model_fit_ar.predict(len(data1), len(data1))

            ar_x.append(ar_yhat1[0])
            model_ar2 = AutoReg(data2, lags=1)
            model_fit_ar2 = model_ar2.fit()


            ar_yhat2 = model_fit_ar2.predict(len(data2), len(data2))
            ar_y.append(ar_yhat2[0])

            model_ma = ARMA(data1, order=(0, 1))
            model_fit_ma = model_ma.fit(disp=False)

            ma_yhat = model_fit_ma.predict(len(data1), len(data1))

            ma_x.append(ma_yhat)

            model_ma2 = ARMA(data2, order=(0, 1))
            model_fit_ma2 = model_ma2.fit(disp=False)

            ma_yhat2 = model_fit_ma2.predict(len(data2), len(data2))

            ma_y.append(ma_yhat2)
            model_hwes = ExponentialSmoothing(data1)
            model_fit_hwes = model_hwes.fit()

            yhat_hwes = model_fit_hwes.predict(len(data1), len(data1))
            hwes_x.append(yhat_hwes)

            model_hwes2 = ExponentialSmoothing(data2)
            model_fit_hwes2 = model_hwes2.fit()

            yhat_hwes2 = model_fit_hwes2.predict(len(data2), len(data2))
            hwes_y.append(yhat_hwes2)


plt.plot(ar_x, ar_y,'ro')
plt.savefig("Autoreg.png")
plt.plot(ma_x, ma_y,'ro')
plt.savefig("ARMA.png")
plt.plot(hwes_x, hwes_y,'ro')
plt.savefig("ExponentialSmoothing.png")

