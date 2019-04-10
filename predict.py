from sklearn.externals import joblib
import os
import csv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

test_example_filefolder="F:\\方向二题目一数据包\\stage1_dataset\\test_api_txt_with_blank"
wordlist_filename="F:\\方向二题目一数据包\\stage1_dataset\\train\\topwordlist.txt"   #top65的api序列文件
wordlist=[]
with open(wordlist_filename,'r') as f:
    worddata=f.readlines()
    for i in worddata:
        wordlist.append(i)
sample_list=[]
list_test=os.listdir(test_example_filefolder)
for i in list_test:
    with open(test_example_filefolder+'\\'+i,'r') as f:
        temp_api_list=f.readlines()
        tempvec=[]
        for j in temp_api_list:
            j=j.lower()
            list_str=j.split(' ')
        for k in wordlist:
            k=k[:-1]
            tempvec.append(list_str.count(k))
        #print(tempvec)
        tempvec.append(1)
        sample_list.append(tempvec)
slist = xgb.DMatrix(sample_list)
new_tree = joblib.load("save/xgboost.pkl")
list = new_tree.predict(slist)

rootdir='F:\\方向二题目一数据包\\stage1_dataset\\test'
dic = os.listdir(rootdir)
file_list=[]
#file_list.append('id')
for i in dic:
    i=i[:-4]
    file_list.append(i)
name=['id','safe_type']
# test = pd.DataFrame(file_list, list, columns=name)
# print(test)
# test.to_csv('F:\\方向二题目一数据包\\stage1_dataset\\test.csv')
print(file_list)
with open('F:\\方向二题目一数据包\\stage1_dataset\\testxg.csv','w+') as f:
    writer = csv.writer(f,lineterminator='\n')
    writer.writerows(zip(file_list,list))

