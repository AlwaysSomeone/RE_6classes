import numpy as np

#读取relation2id
relation2id = {}
id2relation = {}
f = open('./origin_data/relation2id.txt', 'r', encoding='utf-8')
while True:
    content = f.readline()
    if content == '':
        break
    content = content.strip().split()
    relation2id[content[0]] = int(content[1])
    id2relation[int(content[1])] = content[0]
f.close()


#读取测试文件真实关系列表
allans = np.load('./data/allans.npy')
allans = np.reshape(allans, (250,6))
#print(allans)
ans = []
allans = list(allans)
for i in allans:
    i = list(i)
    ans.append(i.index(max(i)))
for i in range(len(ans)):
    ans[i] = ans[i]+1
    #print(id2relation[ans[i]])
#print(ans)


#读取预测关系列表
allprob = np.load('./out/allprob_iter_200.npy')
allprob = np.reshape(allprob, (250,6))
allprob = list(allprob)
#print(allprob)
pred = []
for i in allprob:
    #top3_id = i.argsort()[-3:][::-1]
    #print(top3_id)
    #top1_id = i.argsort()[-1:][::-1]
    #print(top1_id)
    i = list(i)
    #print(i.index(max(i)))
    pred.append(i.index(max(i)))
for i in range(len(pred)):
    pred[i] = pred[i]+1
    #print(id2relation[pred[i]])
#print(pred)


#比较两个列表并计数
count = 0
wrong = []
for i in range(len(ans)):
    if ans[i] == pred[i]:
        count = count + 1
    else:
        wrong.append(i)
#print(wrong)

#打印错误结果
num = -1
f = open('./origin_data/test.txt', 'r', encoding='utf-8')
while True:
    content = f.readline()
    if content == '':
        break
    num = num + 1
    #print(num)
    if num in wrong:
        #print(num)
        content = content.strip().split()
        en1 = content[0]
        en2 = content[1]
        relation = content[2]
        sentence = content[3]
        print(sentence)
        print('两实体为：'+ en1 + ' 和 ' + en2)
        print('实际关系为：'+ relation)
        print( '预测关系为：' + id2relation[pred[num]])
        print('\n')