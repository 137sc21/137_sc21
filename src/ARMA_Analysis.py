import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARMA

warnings.simplefilter("ignore")
from datetime import datetime
macine_list = ["ibmq_16_melbourne.csv","ibmq_ourense.csv","ibmq_vigo.csv","ibmq_5_yorktown - ibmqx2.csv"]
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
    Original_CNOT=[]
    Final_T1= []
    Final_T2= []
    Final_Frequncy= []
    Final_Readout= []
    Final_SQU3= []
    Final_CNOT = []
    for subdir in dirs:
        subpath = os.listdir(path+subdir)
        for files in subpath:
            if files == machines:
                name = machines.split(".")
                machine_name = name[0]
                df = pd.read_csv(path+subdir+"\\"+files,index_col=None,header=0)
                T1 = df.iloc[:,1]
                T2 = df.iloc[:,2]
                Freqency = df.iloc[:,3]
                Readout = df.iloc[:,4]
                SQU3 = df.iloc[:,5]
                CNOT = df.iloc[:,6]
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



                T1_Model =  ARMA(T1_data, order=(0, 1))
                T1_Fit = T1_Model.fit(disp=False)
                Yhat_T1 = T1_Fit.predict(len(T1_data), len(T1_data))
                Final_T1.append(Yhat_T1)

                T2_Model = ARMA(T2_data, order=(0, 1))
                T2_Fit = T2_Model.fit(disp=False)
                Yhat_T2 = T2_Fit.predict(len(T2_data), len(T2_data))
                Final_T2.append(Yhat_T2)

                Frequency_Model = ARMA(Freqency_data, order=(0, 1))
                Freqency_Fit = Frequency_Model.fit(disp=False)
                Yhat_Frequency = Freqency_Fit.predict(len(Freqency_data), len(Freqency_data))
                Final_Frequncy.append(Yhat_Frequency)

                Readout_Model = ARMA(Readout_data, order=(0, 1))
                Readout_Fit = Readout_Model.fit(disp=False)
                Yhat_Readout = Readout_Fit.predict(len(Readout_data), len(Readout_data))
                Final_Readout.append(Yhat_Readout)

                SQU3_Model = ARMA(SQU3_data, order=(0, 1))
                SQU3_Fit = SQU3_Model.fit(disp=False)
                Yhat_SQU3 = SQU3_Fit.predict(len(SQU3_data), len(SQU3_data))
                Final_SQU3.append(Yhat_SQU3)

    directory = date_analysis
    parent_dir = "./Result/"+machine_name+"/"
    path = os.path.join(parent_dir, directory)
    os.mkdir(path)



    plt.plot(Final_T1, label='T1')
    # plt.scatter(Original_T1,Original_T2,label='Original T1 T2')
    plt.plot(Final_T2, label='T2')
    plt.boxplot(Original_T1)
    plt.boxplot(Original_T2)
    plt.legend()
    plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/T1_T2_Changes.png")

    plt.cla()

    plt.plot(Final_Frequncy,label='Frequency')
    plt.legend()
    plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/Frequency_Changes.png")

    plt.cla()

    plt.plot(Final_SQU3,label='Single Qubit U3 Error')
    plt.legend()
    plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/SQU3_Changes.png")

    plt.cla()

    plt.plot(Final_Readout,label='Readout Error')
    plt.legend()
    plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/Readout_Changes.png")

    plt.cla()

    plt.plot(Final_T1, Final_T2,'ro')
    plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/T1_T2.png")

    plt.cla()

    plt.plot(Final_T1, Final_Frequncy,'ro')
    plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/T1_Frequency.png")

    plt.cla()

    plt.plot(Final_T1, Final_Readout,'ro')
    plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/T1_Readout.png")

    plt.cla()

    plt.plot(Final_T1, Final_SQU3,'ro')
    plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/T1_SQU3.png")

    plt.cla()

    plt.plot(Final_T2, Final_Frequncy,'ro')
    plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/T2_Frequency.png")

    plt.cla()

    plt.plot(Final_T2, Final_Readout,'ro')
    plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/T2_Readout.png")

    plt.cla()

    plt.plot(Final_T2, Final_SQU3,'ro')
    plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/T2_SQU3.png")

    plt.cla()

    plt.plot(Final_SQU3, Final_Readout,'ro')
    plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/SQU3_Readout.png")

    plt.cla()

    plt.plot(Final_Frequncy, Final_SQU3,'ro')
    plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/Frequency_SQU3.png")

    plt.cla()

    plt.plot(Final_Frequncy, Final_Readout,'ro')
    plt.savefig("./Result/"+machine_name+"/"+date_analysis+"/Frequency_Readout.png")

    img_path = "./Result/"+machine_name+"/"+date_analysis+"/"
    img_dirs = os.listdir(img_path)
    from PIL import Image
    imageList = []
    for files in img_dirs:
        image = Image.open(str("./Result/"+machine_name+"/"+date_analysis+"/"+files))
        im1 = image.convert('RGB')
        imageList.append(im1)
    im1.save(r"./Result/"+machine_name+"/"+date_analysis+"/Result.pdf",save_all=True, append_images=imageList)