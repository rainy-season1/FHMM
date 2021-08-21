
import csv
import pandas as pd
import numpy as np
from os import listdir
from os.path import *

TIMESTAMP_COLUMN_NAME = "TS"
TIMEZONE = "America/Vancouver"

def csv_reader(csv_name,n):
	y=[]
	with open(csv_name, 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			y.append(row[n])
	f.close()
	return y



def csv_wirter(csv_name,value_list):
	with open(csv_name, 'w') as f:
		writer = csv.writer(f)
		writer.writerow(value_list)
	f.close()
    
def csv_to_data(csv_name):
    df=pd.read_csv(csv_name)
    df.columns = [x.replace(" ", "") for x in df.columns]
    df.index = pd.to_datetime(df[TIMESTAMP_COLUMN_NAME], unit='s', utc=True)
    df = df.drop(TIMESTAMP_COLUMN_NAME, 1)
    df = df.tz_convert(TIMEZONE)
    df = df.apply(pd.to_numeric, errors='ignore')
    df = df.dropna()
    df = df.astype(np.float32)
    return df

if __name__=="__main__":
    #数据集所在路径input_path
    #数据集转换后输出路径out_path
    input_path='E:/data//'
    out_path='E:/NILM/pkl//'
    files=[f for f in listdir(input_path) if isfile(join(input_path, f)) and
             '.csv' in f]
    del_files=['FRG.csv','HTW.csv','WHE.csv','WHG.csv','WHW.csv']
    files=list(set(files)-set(del_files))
    files.sort()
    print(files)
    name=[splitext(x)[0] for x in files]
    print(name)
    data_set=[]
    for i, csv_file in enumerate(files):
        data=csv_to_data(join(input_path,csv_file))
        data_set.append(data)
    all_data=pd.concat(data_set,axis=1)
    all_data.columns=(pd.MultiIndex.from_product([name,
                                        ['V','I','f','DPF','APF','P','Pt','Q','Qt','S','St']],
                                        names=['appliance','physical_quantity']))
    all_data.to_pickle(join(out_path,'WHOLE.pkl'))

    
#[B1E,B2E,BME,CDE,CWE,DNE,DWE,EBE,EQE,FGE,FRE,GRE,HPE,HTE,OFE,OUE,TVE,UTE,WOE]

# a=pd.concat([B1E,B2E,BME,CDE,CWE,DNE,DWE,EBE,EQE,FGE,FRE,GRE,HPE,HTE,OFE,OUE,TVE,UTE,WOE],axis=1)
# a.columns=(pd.MultiIndex.from_product([['B1E','B2E','BME','CDE','CWE','DNE','DWE','EBE','EQE','FGE',
#                                         'FRE','GRE','HPE','HTE','OFE','OUE','TVE','UTE','WOE'],
#                                        ['V','I','f','DPF','APF','P','Pt','Q','Qt','S','St']],
#                                        names=['appliance','physical_quantity']))

# a.to_pickle('E:/NILM/pkl/WHOLE.pkl')







