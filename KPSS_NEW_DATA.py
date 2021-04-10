import os, sys

import matplotlib
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
import warnings
import re
warnings.simplefilter("ignore")
from datetime import datetime


T1_KPSS_dataframe = pd.DataFrame()
T2_KPSS_dataframe = pd.DataFrame()
Frequency_KPSS_dataframe = pd.DataFrame()
SQU3_KPSS_dataframe = pd.DataFrame()
Readout_KPSS_dataframe = pd.DataFrame()
CNOT_KPSS_dataframe = pd.DataFrame()

T1_ADF_dataframe = pd.DataFrame()
T2_ADF_dataframe = pd.DataFrame()
Frequency_ADF_dataframe = pd.DataFrame()
SQU3_ADF_dataframe = pd.DataFrame()
Readout_ADF_dataframe = pd.DataFrame()
CNOT_ADF_dataframe = pd.DataFrame()

appended_T1_adf =[]
appended_T1_kpss =[]
appended_T2_adf =[]
appended_T2_kpss =[]
appended_Frequency_adf =[]
appended_Frequency_kpss =[]
appended_SQU3_adf =[]
appended_SQU3_kpss =[]
appended_Readout_adf =[]
appended_Readout_kpss =[]
appended_CNOT_adf =[]
appended_CNOT_kpss =[]
def adf_test(timeseries):
    # print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used',
                                             'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return(dfoutput)


def kpss_test(timeseries):
    # print('Results of KPSS Test:')

    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value

    # kpss_output = kpss_output.cumsum()
    # dataframe=(kpss_output.to_frame().T)

    # print(dataframe)
    # dataframe.plot(kind='bar',y='p-value',x='timestamp',legend=False,figsize=(8,8),xlabel='ADF Statistical Test',ylabel='Value')
    return(kpss_output)

macine_list = ["ibmq_16_melbourne_calibrations.csv",  "ibmq_athens_calibrations.csv",
               "ibmq_belem_calibrations.csv",
               "ibmq_dublin_calibrations.csv", "ibmq_guadalupe_calibrations.csv", "ibmq_kolkata_calibrations.csv",
               "ibmq_lima_calibrations.csv", "ibmq_manhattan_calibrations.csv", "ibmq_montreal_calibrations.csv",
               "ibmq_mumbai_calibrations.csv", "ibmq_paris_calibrations.csv", "ibmq_quito_calibrations.csv",
                "ibmq_santiago_calibrations.csv", "ibmq_sydney_calibrations.csv",
               "ibmq_toronto_calibrations.csv", "ibmq_vigo_calibrations.csv", "ibmqx2_calibrations.csv"]
for machines in macine_list:
    date_analysis = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    machine_name = ""
    path = ".\\Full_Log\\"
    dirs = os.listdir(path)
    pd.set_option('display.max_columns', None)
    frame = pd.DataFrame()
    df2 = pd.DataFrame()
    Original_Qubit = []
    Original_T1 = []
    Original_T2 = []
    Original_Frequency = []
    Original_Readout = []
    Original_SQU3 = []
    Original_CNOT_Keys = []
    Original_CNOT_Values = []
    appended_data = []
    for subdir in dirs:
        subpath = os.listdir(path + subdir)
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


                # print(name)
                # print(subdir)
                df = pd.read_csv(path + subdir + "\\" + files, index_col=None, header=0)
                df = df.rename(columns={"√x (sx) error": "Sqrt-x (sx) error"})
                df = df.rename(columns={"Readout assignment error": "Readout error"})
                # df = df.rename(columns={"Single - qubit U2 error": "Sqrt-x (sx) error"})

                # df = df.rename(columns={"Single-qubit Pauli-X error": "Sqrt-x (sx) error"})
                # for col in df.columns:
                #     print(col)
                try:
                    Qubit = df['Qubit']
                    T1 = df['T1 (µs)']
                    T2 = df['T2 (µs)']
                    Freqency = df['Frequency (GHz)']
                    Readout = df['Readout error']
                    SQU3 = df['Sqrt-x (sx) error']
                    CNOT = df['CNOT error']
                    df['Qubit'] = df['Qubit'].fillna(0)
                except:
                    pass
                # try:
                #     T1 = df['T1 (µs)']
                #     T2 = df['T2 (µs)']
                #     Freqency = df['Frequency (GHz)']
                #     Readout = df['Readout assignment error']
                #     SQU3 = df['Sqrt-x (sx) error']
                #     CNOT = df['CNOT error']
                # except:
                #     pass
                # try:
                #     T1 = df['T1 (µs)']
                #     T2 = df['T2 (µs)']
                #     Freqency = df['Frequency (GHz)']
                #     Readout = df['Readout assignment error']
                #     SQU3 = df['√x (sx) error']
                #     CNOT = df['CNOT error']
                # except:
                #     pass
                # try:
                #     T1 = df['T1 (µs)']
                #     T2 = df['T2 (µs)']
                #     Freqency = df['Frequency (GHz)']
                #     Readout = df['Readout assignment error']
                #     SQU3 = df['Single-qubit Pauli-X error']
                #     CNOT = df['CNOT error']
                # except:
                #     pass

                df.insert(0, 'timestamp', str(subdir))
                df.timestamp = pd.to_datetime(str(subdir), format='%m-%d-%Y %H-%M-%S')
                # print(df)

                df.index = df.timestamp

                # print(df)
                # df.drop('Month', axis=1, inplace=True)
                # df.iloc[:,1].plot()
                for cnot_parts in CNOT:
                    cnot_key_value = re.split(',', str(cnot_parts))
                    for i in cnot_key_value:
                        # print("===================================================================")
                        # individual = str(i)
                        indivitual = re.split(':', i)
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
                        T1_data = np.array(T1)
                        T2_data = np.array(T2)
                        Freqency_data = np.array(Freqency)
                        Readout_data = np.array(Readout)
                        SQU3_data = np.array(SQU3)

                        Original_T1.append(T1_data)
                        Original_T2.append(T2_data)
                        Original_Frequency.append(Freqency_data)
                        Original_Readout.append(Readout_data)
                        Original_SQU3.append(SQU3_data)
                adf_res = adf_test(df['T1 (µs)'])
                result = kpss_test(df['T1 (µs)'])
                stodataframe = pd.DataFrame()
                stodataframe = result.to_frame().T.__deepcopy__()
                stodataframe.insert(0, 'timestamp', str(subdir))
                stodataframe.timestamp = pd.to_datetime(str(subdir), format='%m-%d-%Y %H-%M-%S')
                stodataframe.index = stodataframe.timestamp

                stodataframe2 = pd.DataFrame()
                stodataframe2 = adf_res.to_frame().T.__deepcopy__()
                stodataframe2.insert(0, 'timestamp', str(subdir))
                stodataframe2.timestamp = pd.to_datetime(str(subdir), format='%m-%d-%Y %H-%M-%S')
                stodataframe2.index = stodataframe2.timestamp

                appended_T1_kpss.append(stodataframe)
                appended_T1_adf.append((stodataframe2))

                adf_res = adf_test(df['T2 (µs)'])
                result = kpss_test(df['T2 (µs)'])
                stodataframe = pd.DataFrame()
                stodataframe = result.to_frame().T.__deepcopy__()
                stodataframe.insert(0, 'timestamp', str(subdir))
                stodataframe.timestamp = pd.to_datetime(str(subdir), format='%m-%d-%Y %H-%M-%S')
                stodataframe.index = stodataframe.timestamp

                stodataframe2 = pd.DataFrame()
                stodataframe2 = adf_res.to_frame().T.__deepcopy__()
                stodataframe2.insert(0, 'timestamp', str(subdir))
                stodataframe2.timestamp = pd.to_datetime(str(subdir), format='%m-%d-%Y %H-%M-%S')
                stodataframe2.index = stodataframe2.timestamp

                appended_T2_kpss.append(stodataframe)
                appended_T2_adf.append((stodataframe2))

                adf_res = adf_test(df['Frequency (GHz)'])
                result = kpss_test(df['Frequency (GHz)'])
                stodataframe = pd.DataFrame()
                stodataframe = result.to_frame().T.__deepcopy__()
                stodataframe.insert(0, 'timestamp', str(subdir))
                stodataframe.timestamp = pd.to_datetime(str(subdir), format='%m-%d-%Y %H-%M-%S')
                stodataframe.index = stodataframe.timestamp

                stodataframe2 = pd.DataFrame()
                stodataframe2 = adf_res.to_frame().T.__deepcopy__()
                stodataframe2.insert(0, 'timestamp', str(subdir))
                stodataframe2.timestamp = pd.to_datetime(str(subdir), format='%m-%d-%Y %H-%M-%S')
                stodataframe2.index = stodataframe2.timestamp

                appended_Frequency_kpss.append(stodataframe)
                appended_Frequency_adf.append((stodataframe2))

                adf_res = adf_test(df['Readout error'])
                result = kpss_test(df['Readout error'])
                stodataframe = pd.DataFrame()
                stodataframe = result.to_frame().T.__deepcopy__()
                stodataframe.insert(0, 'timestamp', str(subdir))
                stodataframe.timestamp = pd.to_datetime(str(subdir), format='%m-%d-%Y %H-%M-%S')
                stodataframe.index = stodataframe.timestamp

                stodataframe2 = pd.DataFrame()
                stodataframe2 = adf_res.to_frame().T.__deepcopy__()
                stodataframe2.insert(0, 'timestamp', str(subdir))
                stodataframe2.timestamp = pd.to_datetime(str(subdir), format='%m-%d-%Y %H-%M-%S')
                stodataframe2.index = stodataframe2.timestamp

                appended_Readout_kpss.append(stodataframe)
                appended_Readout_adf.append((stodataframe2))

                adf_res = adf_test(df['Sqrt-x (sx) error'])
                result = kpss_test(df['Sqrt-x (sx) error'])
                stodataframe = pd.DataFrame()
                stodataframe = result.to_frame().T.__deepcopy__()
                stodataframe.insert(0, 'timestamp', str(subdir))
                stodataframe.timestamp = pd.to_datetime(str(subdir), format='%m-%d-%Y %H-%M-%S')
                stodataframe.index = stodataframe.timestamp

                stodataframe2 = pd.DataFrame()
                stodataframe2 = adf_res.to_frame().T.__deepcopy__()
                stodataframe2.insert(0, 'timestamp', str(subdir))
                stodataframe2.timestamp = pd.to_datetime(str(subdir), format='%m-%d-%Y %H-%M-%S')
                stodataframe2.index = stodataframe2.timestamp

                appended_SQU3_kpss.append(stodataframe)
                appended_SQU3_adf.append((stodataframe2))


                # Cnot_dataframe = pd.DataFrame([Original_CNOT_Keys,Original_CNOT_Values])
                # Cnot_dataframe.insert(0, 'timestamp', str(subdir))
                # Cnot_dataframe.timestamp = pd.to_datetime(str(subdir), format='%Y-%m-%d')
                # Cnot_dataframe.index = Cnot_dataframe.timestamp
                # print(Cnot_dataframe)

                adf_res = adf_test(Original_CNOT_Values)
                result = kpss_test(Original_CNOT_Values)
                stodataframe = pd.DataFrame()
                stodataframe = result.to_frame().T.__deepcopy__()
                stodataframe.insert(0, 'timestamp', str(subdir))
                stodataframe.timestamp = pd.to_datetime(str(subdir), format='%m-%d-%Y %H-%M-%S')
                stodataframe.index = stodataframe.timestamp

                stodataframe2 = pd.DataFrame()
                stodataframe2 = adf_res.to_frame().T.__deepcopy__()
                stodataframe2.insert(0, 'timestamp', str(subdir))
                stodataframe2.timestamp = pd.to_datetime(str(subdir), format='%m-%d-%Y %H-%M-%S')
                stodataframe2.index = stodataframe2.timestamp

                appended_CNOT_kpss.append(stodataframe)
                appended_CNOT_adf.append((stodataframe2))

                # appended_data.append(stodataframe)
                # appended_data_adf.append((stodataframe2))

                # print(result)
                #
                # result.to_frame().T.plot(kind='line', legend=True, figsize=(8, 8), xlabel='ADF Statistical Test',ylabel='Value')

                df['T1'] = df['T1 (µs)'] - df['T1 (µs)'].shift(1)
                # df['T1'].dropna().plot()
                # plt.cla()
                # plt.show()
                # print("===================================================================")
                # print( df['T1'])
                # print("===================================================================")

                # df2 = adf_res.to_frame().T
                # df2.insert(0, 'timestamp', str(subdir))
                # df2.timestamp = pd.to_datetime(str(subdir), format='%Y-%m-%d')
                # print(df)
                # df2.index = df2.timestamp
                # df3 = pd.Series(kpss_res)
                n = 7
                df['T1'] = df['T1 (µs)'] - df['T1 (µs)'].shift(n)
                df['T1_log'] = np.log(df['T1 (µs)'])
                df['T1_log_diff'] = df['T1_log'] - df['T1_log'].shift(1)
                # df['T1_log_diff'].dropna().plot()
                df = df.replace(np.nan, 0)
                # plt.polar(df['T1_log_diff'])
                # adf_test(df['T1_log_diff'].dropna())
                # kpss_test(df['T1_log_diff'].dropna())
                # from pandas.plotting import scatter_matrix
                # scatter_matrix(df2, alpha=0.2, figsize=(8, 8), diagonal="kde")
                df['T1_log_diff']=df['T1_log_diff']
                # print(df2)
                #['Test Statistic', 'p-value', 'Lags Used']
    # df2.plot(kind='line',y='p-value',x='timestamp',legend=True,figsize=(8,8),xlabel='ADF Statistical Test',ylabel='Value')
                # print(df['T1_log_diff'])
                # plt.hist(df2['p-value'])

                # df2.dropna().plot()
                # plt.show()
    # plt.cla()

    T1_KPSS_dataframe = pd.concat(appended_T1_kpss)
    T1_ADF_dataframe = pd.concat(appended_T1_adf)

    T2_KPSS_dataframe = pd.concat(appended_T2_kpss)
    T2_ADF_dataframe = pd.concat(appended_T2_adf)

    Frequency_KPSS_dataframe = pd.concat(appended_Frequency_kpss)
    Frequency_ADF_dataframe = pd.concat(appended_Frequency_adf)

    SQU3_KPSS_dataframe = pd.concat(appended_SQU3_kpss)
    SQU3_ADF_dataframe  = pd.concat(appended_SQU3_adf)

    Readout_KPSS_dataframe = pd.concat(appended_Readout_kpss)
    Readout_ADF_dataframe = pd.concat(appended_Readout_adf)

    CNOT_KPSS_dataframe = pd.concat(appended_CNOT_kpss)
    CNOT_ADF_dataframe = pd.concat(appended_CNOT_adf)
    # dataframe = pd.concat(appended_data)
    # dataframe2 = pd.concat(appended_data_adf)
    # print(dataframe)
    directory = date_analysis
    parent_dir = "./Result/"+machine_name+"/"
    path = os.path.join(parent_dir, directory)
    os.mkdir(path)

    # plt.show()
    fig, axes = plt.subplots(nrows=3, ncols=2)
    T1_KPSS_dataframe.plot(color='blue', xticks=[], ax=axes[0, 0], kind='line', y='p-value', x='timestamp',
                           subplots=True, legend=False, figsize=(15, 10), xlabel='T1')
    T2_KPSS_dataframe.plot(color='blue', xticks=[], ax=axes[0, 1], kind='line', y='p-value', x='timestamp',
                           subplots=True, legend=False, figsize=(15, 10), xlabel='T2')
    Readout_KPSS_dataframe.plot(color='blue', xticks=[], ax=axes[1, 0], kind='line', y='p-value', x='timestamp',
                                subplots=True, legend=False, figsize=(15, 10), xlabel='Readout', ylabel='P-Value')
    Frequency_KPSS_dataframe.plot(color='blue', xticks=[], ax=axes[1, 1], kind='line', y='p-value', x='timestamp',
                                  subplots=True, legend=False, figsize=(15, 10), xlabel='Frequency')
    SQU3_KPSS_dataframe.plot(color='blue', xticks=[], ax=axes[2, 0], kind='line', y='p-value', x='timestamp',
                             subplots=True, legend=False, figsize=(15, 10), xlabel='SQU3')
    CNOT_KPSS_dataframe.plot(color='blue', xticks=[], ax=axes[2, 1], kind='line', y='p-value', x='timestamp',
                             subplots=True, legend=False, figsize=(15, 10), xlabel='CNOT')
    # plt.savefig("./Result/" + machine_name + "/" + date_analysis + "/ALL_KPSS.png")
    fig.tight_layout()
    # plt.cla()
    # fig, axes = plt.subplots(nrows=3, ncols=2)
    # T1_ADF_dataframe.plot(color='red', xticks=[], ax=axes[0, 0], kind='line', y='p-value', x='timestamp', subplots=True,
    #                       legend=False, figsize=(15, 10), xlabel='T1')
    # T2_ADF_dataframe.plot(color='red', xticks=[], ax=axes[0, 1], kind='line', y='p-value', x='timestamp', subplots=True,
    #                       legend=False, figsize=(15, 10), xlabel='T2')
    # Frequency_ADF_dataframe.plot(color='red', xticks=[], ax=axes[1, 0], kind='line', y='p-value', x='timestamp',
    #                              subplots=True, legend=False, figsize=(15, 10), xlabel='Frequency', ylabel='P-Value')
    # Readout_ADF_dataframe.plot(color='red', xticks=[], ax=axes[1, 1], kind='line', y='p-value', x='timestamp',
    #                            subplots=True, legend=False, figsize=(15, 10), xlabel='Readout')
    # SQU3_ADF_dataframe.plot(color='red', xticks=[], ax=axes[2, 0], kind='line', y='p-value', x='timestamp',
    #                         subplots=True, legend=False, figsize=(15, 10), xlabel='SQU3')
    # CNOT_ADF_dataframe.plot(color='red', xticks=[], ax=axes[2, 1], kind='line', y='p-value', x='timestamp',
    #                         subplots=True, legend=False, figsize=(15, 10), xlabel='CNOT')
    # from matplotlib.lines import Line2D
    # cmap = plt.cm.coolwarm
    # custom_lines = [Line2D([0], [0], color=cmap(1.0), lw=4),
    #                 Line2D([1], [1], color=cmap(1.0), lw=4)]
    fig.legend(['KPSS 2020-2021', 'KPSS 2019'],loc='lower left')
    plt.savefig("./Result/" + machine_name + "/" + date_analysis + "/2020_ADF_KPSS.png")

                          #                xlabel='T1 ADF Statistical Test', ylabel='Value')
    # T1_ADF_dataframe.plot(kind='line',y=['p-value','Test Statistic'], x='timestamp', subplots=True, legend=True, figsize=(8, 8),
    #                xlabel='T1 ADF Statistical Test', ylabel='Value')
    # # plt.show()
    # # plt.cla()
    # plt.savefig("./Result/" + machine_name + "/" + date_analysis + "/T1_ADF.png")
    #
    #
    # # plt.show()
    # plt.savefig("./Result/" + machine_name + "/" + date_analysis + "/T2_KPSS.png")
    # T2_ADF_dataframe.plot(kind='line',y=['p-value','Test Statistic'], x='timestamp', subplots=True, legend=True, figsize=(8, 8),
    #                xlabel='T2 ADF Statistical Test', ylabel='Value')
    # # plt.show()
    # plt.savefig("./Result/" + machine_name + "/" + date_analysis + "/T2_ADF.png")
    # # plt.cla()
    #
    # # plt.show()
    # plt.savefig("./Result/" + machine_name + "/" + date_analysis + "/Readout_KPSS.png")
    # Readout_ADF_dataframe.plot(kind='line',y=['p-value','Test Statistic'], x='timestamp', subplots=True, legend=True, figsize=(8, 8),
    #                xlabel='Readout ADF Statistical Test', ylabel='Value')
    # plt.savefig("./Result/" + machine_name + "/" + date_analysis + "/Readout_ADF.png")
    # # plt.show()
    # # plt.cla()
    #
    # plt.savefig("./Result/" + machine_name + "/" + date_analysis + "/Frequency_KPSS.png")
    # Frequency_ADF_dataframe.plot(kind='line',y=['p-value','Test Statistic'], x='timestamp', subplots=True, legend=True, figsize=(8, 8),
    #                xlabel='Frequency ADF Statistical Test', ylabel='Value')
    # plt.savefig("./Result/" + machine_name + "/" + date_analysis + "/Frequency_ADF.png")
    # # plt.cla()
    #
    # plt.savefig("./Result/" + machine_name + "/" + date_analysis + "/SQU3_KPSS.png")
    # SQU3_ADF_dataframe.plot(kind='line',y=['p-value','Test Statistic'], x='timestamp', subplots=True, legend=True, figsize=(8, 8),
    #                xlabel='SQU3 ADF Statistical Test', ylabel='Value')
    # plt.savefig("./Result/" + machine_name + "/" + date_analysis + "/SQU3_ADF.png")
    # # plt.cla()
    #
    # plt.savefig("./Result/" + machine_name + "/" + date_analysis + "/CNOT_KPSS.png")
    # CNOT_ADF_dataframe.plot(kind='line',y=['p-value','Test Statistic'], x='timestamp', subplots=True, legend=True, figsize=(8, 8),
    #                xlabel='CNOT ADF Statistical Test', ylabel='Value')
    # plt.savefig("./Result/" + machine_name + "/" + date_analysis + "/CNOT_ADF.png")
    # plt.cla()
    # plt.savefig("./Result/" + machine_name + "/" + date_analysis + "/T1_T2_Changes.png")
    img_path = "./Result/"+machine_name+"/"+date_analysis+"/"
    img_dirs = os.listdir(img_path)
    from PIL import Image
    imageList = []
    for files in img_dirs:
        image = Image.open(str("./Result/"+machine_name+"/"+date_analysis+"/"+files))
        im1 = image.convert('RGB')
        imageList.append(im1)
    im1.save(r"./Result/"+machine_name+"/"+date_analysis+"/ADF_KPSS_Result.pdf",save_all=True, append_images=imageList)
    # plt.box()
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