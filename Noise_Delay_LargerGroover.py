import numpy as np
from qiskit.tools.monitor import job_monitor
from qiskit import *
import time
from datetime import date, datetime
import csv

fields = ['Backend', 'Date', 'Jobs in Queue', 'Start Time', 'Finish Time']
with open('Noise_Delay_LargerCircuit.csv', 'a+') as f:
    write = csv.writer(f)
    write.writerow(fields)
f.close()
machine_list = ['ibmq_belem','ibmq_quito','ibmq_athens','ibmq_lima','ibmq_5_yorktown','ibmq_santiago']
IBMQ.save_account('0662badfba46c90ce9ab6dd85bac6bbbd5867ab02d69dcf14092dce237c620fc2c0d6ff9b3c29e35f9c066105475c4712b0f7908a03c37f81895a2f09928f5db',overwrite=True)
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-research', group='kent-1', project='main')
while True:
    for each in machine_list:

        backend = provider.get_backend(each)
        status = backend.status()
        jobs_in_queue = status.pending_jobs

        n = 5
        grover_circuit = QuantumCircuit(n)


        def initialize_s(qc, qubits):
            """Apply a H-gate to 'qubits' in qc"""
            for q in qubits:
                qc.h(q)
            return qc


        grover_circuit = initialize_s(grover_circuit, [0, 1,2,3,4])
        grover_circuit.cz(0, 1)
        grover_circuit.h([0, 1])
        grover_circuit.cz(2, 3)
        grover_circuit.cz(3, 4)
        grover_circuit.z([0, 1])
        grover_circuit.z([3, 4])
        grover_circuit.cz(0, 1)
        grover_circuit.h([0, 1])


        today = date.today()
        now = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        start = time.time()
        job = execute(grover_circuit, backend)
        job_monitor(job, interval=2)
        results = job.result()
        answer = results.get_counts(grover_circuit)
        finish_time = (time.time() - start)
        name = backend.name()
        print(jobs_in_queue)
        print("=========")
        print(start)
        print("=========")
        print(finish_time)
        print('=========')
        print(answer)
        print('=========')

        rows = [name, now, jobs_in_queue, start,finish_time]
        with open('Noise_Delay_LargerCircuit.csv','a+') as f:
            write = csv.writer(f)
            # write.writerow(fields)
            write.writerow(rows)
    time.sleep(1800)