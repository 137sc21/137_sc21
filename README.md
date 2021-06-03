# 137_sc21
# Temporal Analysis of Quantum Errors in NISQ Computers: an Empirical Study

TAQE was developed as sets of python tools for analysis of features on IBM-Q Quantum Computers using statistical analysis.  
TAQE includes five different analysis and three different parser to evaluate and clean the quantum calibration data. The initial calibration data from IBM-Q stored in Data, the result of our experiments is in the Result folder and the source code of each individual analysis including parser and delay analysis is in SRC folder. 

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
