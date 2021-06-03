import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import ccf
from statsmodels.tsa.stattools import kpss

warnings.simplefilter("ignore")
from datetime import datetime

def ccf_test(lcTime, lcIntA, lcIntB):
    corr = ccf(lcIntA, lcIntB)
    plt.plot(lcTime, corr)
    plt.xlabel(r"$\tau(s)$", fontsize=14)
    plt.ylabel(r"$\rho(\tau)$", fontsize=14)
    plt.title(r"Cross-correlation $\rho(\tau)$ of two LCs", fontsize=14)
    return plt
    # plt.show()

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
                for cnot_parts in CNOT:
                    cnot_key_value = re.split(',',str(cnot_parts))
                    for i in cnot_key_value:
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

                plt = ccf_test(df['timestamp'],df['T1 (µs)'],df['T2 (µs)'])
                df['T1'] = df['T1 (µs)'] - df['T1 (µs)'].shift(1)
                df['T1'].dropna().plot()
                n = 7
                df['T1'] = df['T1 (µs)'] - df['T1 (µs)'].shift(n)
                df['T1_log'] = np.log(df['T1 (µs)'])
                df['T1_log_diff'] = df['T1_log'] - df['T1_log'].shift(1)
                df['T1_log_diff'].dropna().plot()
    plt.show()