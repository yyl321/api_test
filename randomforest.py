import os
from sklearn import tree
#import pandas as ps
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
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
X_train,X_test,y_train,y_test = train_test_split(sample_list,label_list,test_size=0.25,random_state=50)
# r_range = range(10,30)
# r_scores=[]
# for i in r_range:
tree = RandomForestClassifier(n_estimators=120, max_depth=30, random_state=0)
tree.fit(X_train,y_train)
print(tree.score(X_test,y_test))

#print(y_test)
#joblib.dump(tree,'save/tree.pkl')
#     scores = cross_val_score(tree,sample_list,label_list,cv=10,scoring='accuracy')
#     r_scores.append(scores.mean())
# print('Train score:{:.3f}'.format(tree.score(X_train,y_train)))
# print('Test score:{:.3f}'.format(tree.score(X_test,y_test)))
# train_sizes, train_loss, test_loss = learning_curve(RandomForestClassifier(max_depth=30), sample_list, label_list, cv=10, scoring='neg_mean_squared_error',
#                                                 train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
# train_loss_mean = -np.mean(train_loss, axis=1)
# test_loss_mean = -np.mean(test_loss, axis=1)
#
#
# plt.plot(train_sizes,train_loss_mean, 'o-', color="r",
#          label = "Training")
# plt.plot(train_sizes,test_loss_mean, 'o-', color="g",
#          label = "Cross-validation")
#
# plt.xlabel("Training examples")
# plt.ylabel("Loss")
# plt.legend(loc="best")
# plt.show()

# print(r_scores)
# plt.plot(r_range,r_scores)
# plt.xlabel('Ram')
# plt.ylabel('cross_cal_score')
# plt.show()