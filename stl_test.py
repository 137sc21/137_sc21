import os, sys
import pandas as pd
from statsmodels.sandbox.tsa.varma import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from random import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
sns.set_style('darkgrid')

plt.rc('figure',figsize=(16,12))
plt.rc('font',size=13)
from statsmodels.tsa.seasonal import STL


import warnings
import re

import numpy as np
import pandas as pd
import pyflux as pf
from datetime import datetime
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
from datetime import datetime

def add_stl_plot(fig, res, legend):
    """Add 3 plots from a second STL fit"""
    axs = fig.get_axes()
    comps = ['trend', 'seasonal', 'resid']
    for ax, comp in zip(axs[1:], comps):
        series = getattr(res, comp)
        if comp == 'resid':
            ax.plot(series, marker='o', linestyle='none')
        else:
            ax.plot(series)
            if comp == 'trend':
                ax.legend(legend, frameon=False)




def adf_test(timeseries):
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used',
                                             'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


def kpss_test(timeseries):
    print('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
    print(kpss_output)

macine_list = ["ibmq_16_melbourne.csv","ibmq_ourense.csv","ibmq_vigo.csv"]
for machines in macine_list:

    date_analysis = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    machine_name=""
    path = ".\\Log\\"
    dirs = os.listdir(path)
    pd.set_option('display.max_columns', None)
    frame = pd.DataFrame()
    Original_T1 =[]
    Original_T2 =[]
    Original_Frequency= []
    Original_Readout =[]
    Original_SQU3 = []
    Original_CNOT_Keys=[]
    Original_CNOT_Values=[]
    Final_T1= []
    Final_T2= []
    Final_Frequncy= []
    Final_Readout= []
    Final_SQU3= []
    Final_CNOT = []
    for subdir in dirs:
        subpath = os.listdir(path+subdir)
        # print(subdir)
        for files in subpath:
            if files == machines:
                name = machines.split(".")
                machine_name = name[0]
                # print(machine_name)
            # if files == "ibmq_ourense.csv":
            # if files == "ibmq_vigo.csv":
            # if files == "ibmq_5_yorktown - ibmqx2.csv":
                def isfloat(value):
                    try:
                        float(value)
                        return True
                    except ValueError:
                        return False


                df = pd.read_csv(path+subdir+"\\"+files,index_col=None,header=0)
                T1 = df.iloc[:,1]
                T2 = df.iloc[:,2]
                Freqency = df.iloc[:,3]
                Readout = df.iloc[:,4]
                SQU3 = df.iloc[:,5]
                CNOT = df.iloc[:,6]
                df.insert(0, 'timestamp', str(subdir))
                df.timestamp = pd.to_datetime(str(subdir), format='%Y-%m-%d')
                # print(df)
                df.index = df.timestamp
                # print(df)
                # df.drop('Month', axis=1, inplace=True)
                # df.iloc[:,1].plot()
                for cnot_parts in CNOT:
                    cnot_key_value = re.split(',',str(cnot_parts))
                    for i in cnot_key_value:
                        # print("===================================================================")
                        # individual = str(i)
                        indivitual = re.split(':',i)
                        for keys in indivitual:
                            if keys.lower().startswith('nan'):
                                indivitual.remove(keys)
                            keys = keys.strip()
                            if keys.lower().startswith('cx'):
                                Original_CNOT_Keys.append(keys)
                            if isfloat(keys):
                                if np.isnan(float(keys)):
                                    pass
                                else:
                                    Original_CNOT_Values.append(keys)
                        # print(indivitual)
                        # print(Original_CNOT_Keys)
                        # print(Original_CNOT_Values)
                        #
                        # print("===================================================================")
                    # cnot_key = re.split(':',str(cnot_key_value))
                    # for each in cnot_key:
                    #     unwanted = [';', ':', '!', "*",'[',']',"'",' ']
                    #     for i in unwanted:
                    #         each = each.replace(i, '')
                    #     splitagain = re.split(',', str(each))
                    #     # splitagain.lower()
                    #     splitagain = str(splitagain)
                    #     if splitagain.startswith('cx'):
                    #         # print(each)
                    #         print("BOO")
                    #     if isinstance(splitagain,float):
                    #         print("===================================================================")
                    #         print(splitagain)
                    #     print("===================================================================")
                    #     print(splitagain)
                # # CNOT = CNOT.str.split(pat=',')
                # CNOT = (CNOT.str.split(expand=True))
                # print(CNOT)
                T1_data = np.array(T1)
                T2_data = np.array(T2)
                Freqency_data= np.array(Freqency)
                Readout_data = np.array(Readout)
                SQU3_data = np.array(SQU3)

                Original_T1.append(T1_data)
                Original_T2.append(T2_data)
                Original_Frequency.append(Freqency_data)
                Original_Readout.append(Readout_data)
                Original_SQU3.append(SQU3_data)






                #adf_test(df['T1 (µs)'])
                #kpss_test(df['T1 (µs)'])
                #df['T1'] = df['T1 (µs)'] - df['T1 (µs)'].shift(1)
                #df['T1'].dropna().plot()
                # plt.cla()
                # plt.show()
                # print("===================================================================")
                # print( df['T1'])
                # print("===================================================================")
                #n = 7
                #df['T1'] = df['T1 (µs)'] - df['T1 (µs)'].shift(n)
                #df['T1_log'] = np.log(df['T1 (µs)'])
                #df['T1_log_diff'] = df['T1_log'] - df['T1_log'].shift(1)
                #df['T1_log_diff'].dropna().plot()
    # plt.cla()
    #plt.show()
                # df.index = df['Qubit'];
                # df.loc[(df['T1 (µs)'] >= 40.05), 'T1_stat'] = 1;
                # df.loc[(df['T1 (µs)'] < 40.05), 'T1_stat'] = 0;
                # df.loc[(df['T2 (µs)'] >= 74.00), 'T2_stat'] = 1;
                # df.loc[(df['T2 (µs)'] < 74.00), 'T2_stat'] = 0;
                # df.loc[(df['Frecuency (GHz)'] >= 5.00), 'freq_stat'] = 1;
                # df.loc[(df['Frecuency (GHz)'] < 5.00), 'freq_stat'] = 0;
                # df.loc[(df['Readout error'] >= 4.00), 'read_stat'] = 1;
                # df.loc[(df['Readout error'] < 4.00), 'read_stat'] = 0;
                # df.loc[(df['Single-qubit U3 error rate'] >= 2.00), 'single_stat'] = 1;
                # df.loc[(df['Single-qubit U3 error rate'] < 2.00), 'single_stat'] = 0;
                # plt.figure(figsize=(15, 5));
                # plt.plot(df.index, df['T1 (µs)']);
                # plt.ylabel('Qubit T1 T2 Status');
                # plt.title('Time Based ARMAX Analysis');
                # plt.plot();
                # plt.show()
                # #plt.cla()
                # print(df.shape)
                # df=df.rename(columns={'Single-qubit U3 error rate': 'single'})
                # print(df.columns.values.tolist())
                # model = pf.ARIMAX(data=df, formula='timestamp~T1_stat+T2_stat+read_stat+single_stat+T1_stat:T2_stat',
                #                   ar=1, ma=1, family=pf.Normal())
                # print(model.data_length)
                # x = model.fit("MLE")
                # x.summary()
                #
                # model.plot_fit(figsize=(15,10))
                # plt.show()
               # model.plot_predict(h=10, oos_data=df.iloc[-12:], past_values=100, figsize=(15, 5))
                #plt.show()

                # print(T1_data.shape)
                # print(T2_data.shape)
                # print(Freqency_data.shape)
                # print(SQU3_data.shape)
                # print(Readout_data.shape)
                numpy_data = np.array([T1_data,T2_data,Freqency_data,SQU3_data,Readout_data])
                df2 = pd.DataFrame(data=numpy_data.T,index=T1_data, columns=["T1", "T2","Freq","SQU3","ROErr"])
                # print(df2)
                # df2.insert(0, 'timestamp', str(subdir))
                # df2.timestamp = pd.to_datetime(str(subdir), format='%Y-%m-%d')
                # # print(df)
                # df2.index = df2.timestamp
                # print(df2)
                co2 = df2.index
                co2 = pd.Series(co2, index=pd.date_range('1-1-2020', periods=len(co2), freq='M'), name='CO2')
                co2.describe()
                stl = STL(co2, seasonal=13)
                res = stl.fit()
                fig = res.plot()
                df=df.replace(to_replace=r'Q', value='', regex=True)


                # df.replace(to_replace=r'^Q.$', value=i, regex=True)
                # print(df)
                elec_equip = df['Qubit']
                stl = STL(elec_equip, period=12, robust=True)
                res_robust = stl.fit()
                fig = res_robust.plot()
                res_non_robust = STL(elec_equip, period=12, robust=False).fit()
                add_stl_plot(fig, res_non_robust, ['Robust', 'Non-robust'])
                fig = plt.figure(figsize=(16, 5))
                lines = plt.plot(res_robust.weights, marker='o', linestyle='none')
                ax = plt.gca()
                xlim = ax.set_xlim(elec_equip.index[0], elec_equip.index[-1])
                stl = STL(elec_equip, period=12, seasonal_deg=0, trend_deg=0, low_pass_deg=0, robust=True)
                res_deg_0 = stl.fit()
                fig = res_robust.plot()
                add_stl_plot(fig, res_deg_0, ['Degree 1', 'Degree 0'])
                plt.show()
                # temp = df.reset_index()
                # df_stl_month = temp[['index', 'timestamp']].set_index('index')
                # decomp = decompose(df_stl_month.values, period=30)
                #
                # trend = decomp.trend
                # seasonal = decomp.seasonal
                # residual = decomp.resid
                # df['T1 (µs)'] = df.timeseries - df.approx_trend.values
                #
                # plt.figure(figsize=(30, 10))
                # plt.rcParams.update({'font.size': 22})
                # plt.grid()
                # plt.plot(X_index, df.detrended_series, marker='', linestyle='-', label='T1 (µs) Time Data',
                #          linewidth=4)
                # plt.title("T1 data from Jan to Feb")
                # plt.xlabel("Period")
                # plt.ylabel("Value")
                # plt.xticks(rotation=90)
                # plt.legend()
                # plt.show()
                # seasonal_comp = df.groupby('month')['month', 'T1 (µs)'].mean().reset_index(drop=True)
                #
                # plt.figure(figsize=(30, 10))
                # plt.rcParams.update({'font.size': 22})
                # plt.grid()
                # plt.plot(seasonal_comp.month.astype(str), seasonal_comp.detrended_series, marker='', linestyle='-',
                #          label='Periodic', linewidth=4)
                # plt.title("Avg. Period values (across all Collection)")
                # plt.xlabel("Period")
                # plt.ylabel("avg. Value")
                # plt.legend()
                # plt.show()
                # seasonal = pd.DataFrame()
                # for i in range(int(df.timeseries.shape[0] / seasonal_comp.shape[0])):
                #     seasonal = pd.concat([seasonal, seasonal_comp])
                #
                # df['timestamp'] = seasonal.detrended_series.values
                #
                # plt.figure(figsize=(30, 10))
                # plt.rcParams.update({'font.size': 22})
                # plt.grid()
                # plt.plot(X_index, df.seasonal, marker='', linestyle='-', label='seasonality', linewidth=4)
                # plt.title("Seasonality across all Collection")
                # plt.xlabel("Period")
                # plt.ylabel("Value")
                # plt.xticks(rotation=90)
                # plt.legend()
                # plt.show()
                # print("===================================================================")
                # print( df['T1_log_diff'])
                # print("===================================================================")
                # adf_test(df['T2 (µs)'])
                # adf_test(df['Frecuency (GHz)'])
                # adf_test(df['Readout error'])
                # adf_test(df['Single-qubit U3 error rate'])
                # adf_test(df['CNOT error rate'])
                # T1_Model =  kpss(T1_data, regression='c')
                # T1_Fit = T1_Model.fit(disp=False)
                # Yhat_T1 = T1_Fit.predict(len(T1_data), len(T1_data))
                # Final_T1.append(Yhat_T1)
                #
                # T2_Model = kpss(T1_data, regression='c')
                # T2_Fit = T2_Model.fit(disp=False)
                # Yhat_T2 = T2_Fit.predict(len(T2_data), len(T2_data))
                # Final_T2.append(Yhat_T2)
                #
                # Frequency_Model = kpss(T1_data, regression='c')
                # Freqency_Fit = Frequency_Model.fit(disp=False)
                # Yhat_Frequency = Freqency_Fit.predict(len(Freqency_data), len(Freqency_data))
                # Final_Frequncy.append(Yhat_Frequency)
                #
                # Readout_Model = kpss(T1_data, regression='c')
                # Readout_Fit = Readout_Model.fit(disp=False)
                # Yhat_Readout = Readout_Fit.predict(len(Readout_data), len(Readout_data))
                # Final_Readout.append(Yhat_Readout)
                #
                # SQU3_Model = kpss(T1_data, regression='c')
                # SQU3_Fit = SQU3_Model.fit(disp=False)
                # Yhat_SQU3 = SQU3_Fit.predict(len(SQU3_data), len(SQU3_data))
                # Final_SQU3.append(Yhat_SQU3)

    # directory = date_analysis
    # parent_dir = "./Result/"+machine_name+"/"
    # path = os.path.join(parent_dir, directory)
    # os.mkdir(path)
    #
    #
    #
    # plt.plot(Final_T1, label='T1')
    # # plt.scatter(Original_T1,Original_T2,label='Original T1 T2')
    # plt.plot(Final_T2, label='T2')
    # plt.boxplot(Original_T1)
    # plt.boxplot(Original_T2)
    # plt.legend()
    # plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/T1_T2_Changes.png")
    #
    # plt.cla()
    #
    # plt.plot(Final_Frequncy,label='Frequency')
    # plt.legend()
    # plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/Frequency_Changes.png")
    #
    # plt.cla()
    #
    # plt.plot(Final_SQU3,label='Single Qubit U3 Error')
    # plt.legend()
    # plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/SQU3_Changes.png")
    #
    # plt.cla()
    #
    # plt.plot(Final_Readout,label='Readout Error')
    # plt.legend()
    # plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/Readout_Changes.png")
    #
    # plt.cla()
    #
    # plt.plot(Final_T1, Final_T2,'ro')
    # plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/T1_T2.png")
    #
    # plt.cla()
    #
    # plt.plot(Final_T1, Final_Frequncy,'ro')
    # plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/T1_Frequency.png")
    #
    # plt.cla()
    #
    # plt.plot(Final_T1, Final_Readout,'ro')
    # plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/T1_Readout.png")
    #
    # plt.cla()
    #
    # plt.plot(Final_T1, Final_SQU3,'ro')
    # plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/T1_SQU3.png")
    #
    # plt.cla()
    #
    # plt.plot(Final_T2, Final_Frequncy,'ro')
    # plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/T2_Frequency.png")
    #
    # plt.cla()
    #
    # plt.plot(Final_T2, Final_Readout,'ro')
    # plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/T2_Readout.png")
    #
    # plt.cla()
    #
    # plt.plot(Final_T2, Final_SQU3,'ro')
    # plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/T2_SQU3.png")
    #
    # plt.cla()
    #
    # plt.plot(Final_SQU3, Final_Readout,'ro')
    # plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/SQU3_Readout.png")
    #
    # plt.cla()
    #
    # plt.plot(Final_Frequncy, Final_SQU3,'ro')
    # plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/Frequency_SQU3.png")
    #
    # plt.cla()
    #
    # plt.plot(Final_Frequncy, Final_Readout,'ro')
    # plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/Frequency_Readout.png")
    #
    # img_path = "./Result/"+machine_name+"/"+date_analysis+"/"
    # img_dirs = os.listdir(img_path)
    # from PIL import Image
    # imageList = []
    # for files in img_dirs:
    #     image = Image.open(str("./Result/"+machine_name+"/"+date_analysis+"/"+files))
    #     im1 = image.convert('RGB')
    #     imageList.append(im1)
    # im1.save(r"./Result/"+machine_name+"/"+date_analysis+"/Result.pdf",save_all=True, append_images=imageList)