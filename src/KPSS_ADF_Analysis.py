import os

import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

register_matplotlib_converters()
sns.set_style('darkgrid')

plt.rc('figure',figsize=(16,12))
plt.rc('font',size=13)
from statsmodels.tsa.seasonal import STL


import warnings
import re

import numpy as np
import pandas as pd
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

                numpy_data = np.array([T1_data,T2_data,Freqency_data,SQU3_data,Readout_data])
                df2 = pd.DataFrame(data=numpy_data.T,index=T1_data, columns=["T1", "T2","Freq","SQU3","ROErr"])

                co2 = df2.index
                co2 = pd.Series(co2, index=pd.date_range('1-1-2020', periods=len(co2), freq='M'), name='CO2')
                co2.describe()
                stl = STL(co2, seasonal=13)
                res = stl.fit()
                fig = res.plot()
                df=df.replace(to_replace=r'Q', value='', regex=True)

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
