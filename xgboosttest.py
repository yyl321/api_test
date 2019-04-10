import os
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import numpy as np
from sklearn.metrics import classification_report
wordlist_filename="F:\\方向二题目一数据包\\stage1_dataset\\train\\topwordlist.txt"   #top65的api序列文件
wordlist=[]
with open(wordlist_filename,'r') as f:
    worddata=f.readlines()
    for i in worddata:
        wordlist.append(i)
black_example_filefolder="F:\\方向二题目一数据包\\stage1_dataset\\train\\black_examples_with_blank" #恶意样本
white_example_filefolder="F:\\方向二题目一数据包\\stage1_dataset\\train\\white_examples_with_blank" #正常样本
sample_list=[]
label_list=[]
####
list_black=os.listdir(black_example_filefolder)
for i in list_black:
    with open(black_example_filefolder+'\\'+i,'r') as f:
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
        label_list.append(1)
list_white=os.listdir(white_example_filefolder)
for i in list_white:
    with open(white_example_filefolder+'\\'+i,'r') as f:
        temp_api_list=f.readlines()
        tempvec=[]
        for j in temp_api_list:
            j=j.lower()
            list_str=j.split(' ')
        for k in wordlist:
            k=k[:-1]
            tempvec.append(list_str.count(k))
        tempvec.append(1)
        sample_list.append(tempvec)
        label_list.append(0)
print(len(sample_list))
X_train,X_test,y_train,y_test = train_test_split(sample_list,label_list,test_size=0.2,random_state=1234568)
# model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, objective='multi:softmax')
# model.fit(X_train, y_train)
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 3,
    'gamma': 0.01,
    'max_depth': 50,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

plst = params.items()

dtrain = xgb.DMatrix(X_train,y_train)
num_rounds = 500
model = xgb.train(plst, dtrain, num_rounds)

dtest = xgb.DMatrix(X_test)
ans =model.predict(dtest)
print(ans)
joblib.dump(model,'save/xgboost.pkl')

# 计算准确率
cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    if ans[i] == y_test[i]:
        cnt1 += 1
    else:
        cnt2 += 1

print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))

# 显示重要特征
plot_importance(model)
plt.show()