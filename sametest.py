import csv

with open('F:\\方向二题目一数据包\\stage1_dataset\\test.csv','r') as f:
    reader = csv.reader(f)
    column1 = [row[1] for row in reader]
print(type(column1[0]))

with open('F:\\方向二题目一数据包\\stage1_dataset\\result.csv','r') as f:
    reader = csv.reader(f)
    column2 = [row[1] for row in reader]
print(type(column2[0]))
print(column2[1])

print(len(column1))
count=0
for i in range(0,len(column1)):
    tmp1=int(column1[i])
    tmp =int(eval(column2[i]))
    if tmp1!=tmp:
        count+=1
        print(i)
print(count)