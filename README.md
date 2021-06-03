# Temporal Analysis of Quantum Errors in NISQ Computers: an Empirical Study

TAQE was developed as a set of python tools for analysis of features on IBM-Q Quantum Computers using statistical methods.  
TAQE includes different analysis methods and set of different parser to evaluate and clean the quantum calibration data. The initial calibration data from IBM-Q stored in Data, the result of our experiments is in the Result folder and the source code of each individual analysis including parser and delay analysis is in SRC folder.



### Data Collection

#### Calibration Data for each features such as T1, T2, Readout, SQU3, and CNOT Error rate was collected from 21 IBM-Q machines in different time intervals: 

| Time Interval           | # of Features and # of IBM-Q Machines |
|-------------------------|---------------------------------------|
| 09/11/2019              | 7,  1                                 |
| 09/12/2019 - 09/13/2019 | 7,  2                                 |
| 09/14/2019 - 11/11/2019 | 7,  4                                 |
| 06/26/2020 - 09/15/2020 | 4,  1                                 |
| 12/02/2020 - 02/17/2021 | 7,  11                                |
| 02/18/2021 - 03/24/2021 | 7,  21                                |


### Installation 
#### Option 1:

```shell
pip install -r requirements.txt
```
#### Option 2: 
```shell
git clone https://github.com/137sc21/137_sc21.git
cd 137_sc21
conda create --name TAQE python=3.7
source activate TAQE
python setup.py install
```
### Setup and Verify TAQE

1. Install TAQE
2. Run `preprocess.py`
3. Run `XXX_Analysis.py` for desired analysis method

### Prerequisites
- Anaconda(Optional)
- matplotlib==3.3.3
- pandas==1.1.4
- numpy==1.19.3
- statsmodels==0.12.0
- pillow==8.0.1
- qiskit==0.22.0
- ipython==7.16.1
- pyflux==0.4.15
- seaborn==0.11.1


### License 

#### BSD 3-Clause License

##### Copyright (c) 2021, 137sc21 All rights reserved.

###### Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

```THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.```
