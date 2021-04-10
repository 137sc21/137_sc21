import pandas as pd
import os
basePath = os.path.dirname(os.path.abspath(__file__))
# print(basePath)
import sys
from io import StringIO

def CustomParser(data):
    import json
    j1 = json.loads(data)
    return j1

lines = []
with open('Athens') as f:
    lines = (f.read().splitlines())

# print(lines[6])
TESTDATA = StringIO(lines[0])


df = pd.DataFrame(lines)
df4 = pd.read_csv(TESTDATA,sep=",")
# df.columns = ['json_element']
import json
print(df4)
print(df4)
# df2 = (df[6:7].to_json())
# df3 = pd.DataFrame()
# df3 = df2

# df_final = pd.json_normalize(df[6:7].apply(json.loads))
# pd.read_csv(str(lines[6]))
# df2 = pd.json_normalize(df[6:7].apply(json.loads))
# print(df_final)
