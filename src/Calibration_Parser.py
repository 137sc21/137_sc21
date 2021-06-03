import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
from datetime import datetime

macine_list = ["ibmq_16_melbourne_calibrations.csv","ibmq_armonk_calibrations.csv","ibmq_athens_calibrations.csv","ibmq_belem_calibrations.csv","ibmq_bogota_calibrations.csv","ibmq_casablanca_calibrations.csv","ibmq_dublin_calibrations.csv","ibmq_guadalupe_calibrations.csv","ibmq_kolkata_calibrations.csv","ibmq_lima_calibrations.csv","ibmq_manhattan_calibrations.csv","ibmq_montreal_calibrations.csv","ibmq_mumbai_calibrations.csv","ibmq_paris_calibrations.csv","ibmq_quito_calibrations.csv","ibmq_rome_calibrations.csv","ibmq_santiago_calibrations.csv","ibmq_sydney_calibrations.csv","ibmq_toronto_calibrations.csv","ibmq_vigo_calibrations.csv","ibmqx2_calibrations.csv"]
for machines in macine_list:
    date_analysis = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    machine_name = ""
    path = "Data\\"
    dirs = os.listdir(path)
    pd.set_option('display.max_columns', None)
    frame = pd.DataFrame()
    df2 = pd.DataFrame()
    Original_Qubit =[]
    Original_T1 = []
    Original_T2 = []
    Original_Frequency = []
    Original_Readout = []
    Original_SQU3 = []
    Original_CNOT_Keys = []
    Original_CNOT_Values = []
    appended_data = []
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

                # print(name)
                # print(subdir)
                df = pd.read_csv(path + subdir + "\\" + files, index_col=None, header=0)
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
                try:
                    T1 = df['T1 (µs)']
                    T2 = df['T2 (µs)']
                    Freqency = df['Frequency (GHz)']
                    Readout = df['Readout assignment error']
                    SQU3 = df['Sqrt-x (sx) error']
                    CNOT = df['CNOT error']
                except:
                    pass

                df.insert(0, 'timestamp', str(subdir))
                df.timestamp = pd.to_datetime(str(subdir), format='%m-%d-%Y %H-%M-%S')
                # print(df)

                df.index = df.timestamp
                df = df.rename(columns={"Frecuency (GHz)": "Frequency (GHz)"})
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
                appended_data.append(df)
                # print(appended_data)
        try:
            final = pd.concat(appended_data,ignore_index=True)
            final.to_csv(machine_name + '.csv')
        except:
            pass
        # final.to_csv(machine_name+'.csv')