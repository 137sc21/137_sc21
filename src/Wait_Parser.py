import pandas as pd
import os
basePath = os.path.dirname(os.path.abspath(__file__))
from io import StringIO
def CustomParser(data):
    import json
    j1 = json.loads(data)
    return j1
lines = []
with open('Data/Analysis_Part_One/Athens') as f:
    lines = (f.read().splitlines())
TESTDATA = StringIO(lines[0])
df = pd.DataFrame(lines)
df4 = pd.read_csv(TESTDATA,sep=",")
print(df4)
print(df4)

