import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

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

macine_list = ["ibmq_16_melbourne.csv","ibmq_ourense.csv","ibmq_vigo.csv","ibmq_5_yorktown - ibmqx2"]
for machines in macine_list:

    date_analysis = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    machine_name=""
    path = ".\\Log\\"
    dirs = os.listdir(path)
    pd.set_option('display.max_columns', None)
    frame = pd.DataFrame()
    df2 = pd.DataFrame()
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
                df = df.rename(columns={"Frecuency (GHz)": "Frequency (GHz)"})


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

                adf_res = adf_test(df['T1 (??s)'])
                result = kpss_test(df['T1 (??s)'])
                stodataframe = pd.DataFrame()
                stodataframe = result.to_frame().T.__deepcopy__()
                stodataframe.insert(0, 'timestamp', str(subdir))
                stodataframe.timestamp = pd.to_datetime(str(subdir), format='%Y-%m-%d')
                stodataframe.index = stodataframe.timestamp

                stodataframe2 = pd.DataFrame()
                stodataframe2 = adf_res.to_frame().T.__deepcopy__()
                stodataframe2.insert(0, 'timestamp', str(subdir))
                stodataframe2.timestamp = pd.to_datetime(str(subdir), format='%Y-%m-%d')
                stodataframe2.index = stodataframe2.timestamp

                appended_T1_kpss.append(stodataframe)
                appended_T1_adf.append((stodataframe2))

                adf_res = adf_test(df['T2 (??s)'])
                result = kpss_test(df['T2 (??s)'])
                stodataframe = pd.DataFrame()
                stodataframe = result.to_frame().T.__deepcopy__()
                stodataframe.insert(0, 'timestamp', str(subdir))
                stodataframe.timestamp = pd.to_datetime(str(subdir), format='%Y-%m-%d')
                stodataframe.index = stodataframe.timestamp

                stodataframe2 = pd.DataFrame()
                stodataframe2 = adf_res.to_frame().T.__deepcopy__()
                stodataframe2.insert(0, 'timestamp', str(subdir))
                stodataframe2.timestamp = pd.to_datetime(str(subdir), format='%Y-%m-%d')
                stodataframe2.index = stodataframe2.timestamp

                appended_T2_kpss.append(stodataframe)
                appended_T2_adf.append((stodataframe2))

                adf_res = adf_test(df['Frequency (GHz)'])
                result = kpss_test(df['Frequency (GHz)'])
                stodataframe = pd.DataFrame()
                stodataframe = result.to_frame().T.__deepcopy__()
                stodataframe.insert(0, 'timestamp', str(subdir))
                stodataframe.timestamp = pd.to_datetime(str(subdir), format='%Y-%m-%d')
                stodataframe.index = stodataframe.timestamp

                stodataframe2 = pd.DataFrame()
                stodataframe2 = adf_res.to_frame().T.__deepcopy__()
                stodataframe2.insert(0, 'timestamp', str(subdir))
                stodataframe2.timestamp = pd.to_datetime(str(subdir), format='%Y-%m-%d')
                stodataframe2.index = stodataframe2.timestamp

                appended_Frequency_kpss.append(stodataframe)
                appended_Frequency_adf.append((stodataframe2))

                adf_res = adf_test(df['Readout error'])
                result = kpss_test(df['Readout error'])
                stodataframe = pd.DataFrame()
                stodataframe = result.to_frame().T.__deepcopy__()
                stodataframe.insert(0, 'timestamp', str(subdir))
                stodataframe.timestamp = pd.to_datetime(str(subdir), format='%Y-%m-%d')
                stodataframe.index = stodataframe.timestamp

                stodataframe2 = pd.DataFrame()
                stodataframe2 = adf_res.to_frame().T.__deepcopy__()
                stodataframe2.insert(0, 'timestamp', str(subdir))
                stodataframe2.timestamp = pd.to_datetime(str(subdir), format='%Y-%m-%d')
                stodataframe2.index = stodataframe2.timestamp

                appended_Readout_kpss.append(stodataframe)
                appended_Readout_adf.append((stodataframe2))

                adf_res = adf_test(df['Single-qubit U3 error rate'])
                result = kpss_test(df['Single-qubit U3 error rate'])
                stodataframe = pd.DataFrame()
                stodataframe = result.to_frame().T.__deepcopy__()
                stodataframe.insert(0, 'timestamp', str(subdir))
                stodataframe.timestamp = pd.to_datetime(str(subdir), format='%Y-%m-%d')
                stodataframe.index = stodataframe.timestamp

                stodataframe2 = pd.DataFrame()
                stodataframe2 = adf_res.to_frame().T.__deepcopy__()
                stodataframe2.insert(0, 'timestamp', str(subdir))
                stodataframe2.timestamp = pd.to_datetime(str(subdir), format='%Y-%m-%d')
                stodataframe2.index = stodataframe2.timestamp

                appended_SQU3_kpss.append(stodataframe)
                appended_SQU3_adf.append((stodataframe2))


                adf_res = adf_test(Original_CNOT_Values)
                result = kpss_test(Original_CNOT_Values)
                stodataframe = pd.DataFrame()
                stodataframe = result.to_frame().T.__deepcopy__()
                stodataframe.insert(0, 'timestamp', str(subdir))
                stodataframe.timestamp = pd.to_datetime(str(subdir), format='%Y-%m-%d')
                stodataframe.index = stodataframe.timestamp

                stodataframe2 = pd.DataFrame()
                stodataframe2 = adf_res.to_frame().T.__deepcopy__()
                stodataframe2.insert(0, 'timestamp', str(subdir))
                stodataframe2.timestamp = pd.to_datetime(str(subdir), format='%Y-%m-%d')
                stodataframe2.index = stodataframe2.timestamp

                appended_CNOT_kpss.append(stodataframe)
                appended_CNOT_adf.append((stodataframe2))


                df['T1'] = df['T1 (??s)'] - df['T1 (??s)'].shift(1)

                n = 7
                df['T1'] = df['T1 (??s)'] - df['T1 (??s)'].shift(n)
                df['T1_log'] = np.log(df['T1 (??s)'])
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
    T1_KPSS_dataframe.plot(color='red', xticks=[], ax=axes[0, 0], kind='line', y='p-value', x='timestamp',
                           subplots=True, legend=False, figsize=(15, 10), xlabel='T1')
    T2_KPSS_dataframe.plot(color='red', xticks=[], ax=axes[0, 1], kind='line', y='p-value', x='timestamp',
                           subplots=True, legend=False, figsize=(15, 10), xlabel='T2')
    Readout_KPSS_dataframe.plot(color='red', xticks=[], ax=axes[1, 0], kind='line', y='p-value', x='timestamp',
                                subplots=True, legend=False, figsize=(15, 10), xlabel='Readout', ylabel='P-Value')
    Frequency_KPSS_dataframe.plot(color='red', xticks=[], ax=axes[1, 1], kind='line', y='p-value', x='timestamp',
                                  subplots=True, legend=False, figsize=(15, 10), xlabel='Frequency' )
    SQU3_KPSS_dataframe.plot(color='red', xticks=[], ax=axes[2, 0], kind='line', y='p-value', x='timestamp',
                             subplots=True, legend=False, figsize=(15, 10), xlabel='SQU3')
    CNOT_KPSS_dataframe.plot(color='red', xticks=[], ax=axes[2, 1], kind='line', y='p-value', x='timestamp',
                             subplots=True, legend=False, figsize=(15, 10), xlabel='CNOT')
    # plt.savefig("./Result/" + machine_name + "/" + date_analysis + "/ALL_KPSS.png")
    fig.tight_layout()

    plt.savefig("./Result/" + machine_name + "/" + date_analysis + "/2019_ADF_KPSS.png")


    img_path = "./Result/"+machine_name+"/"+date_analysis+"/"
    img_dirs = os.listdir(img_path)
    from PIL import Image
    imageList = []
    for files in img_dirs:
        image = Image.open(str("./Result/"+machine_name+"/"+date_analysis+"/"+files))
        im1 = image.convert('RGB')
        imageList.append(im1)
    im1.save(r"./Result/"+machine_name+"/"+date_analysis+"/ADF_KPSS_Result.pdf",save_all=True, append_images=imageList)
