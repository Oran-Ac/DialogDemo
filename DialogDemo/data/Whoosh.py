

import os
from whoosh.index import create_in
from whoosh.fields import *
from jieba.analyse import ChineseAnalyzer
from whoosh.writing import AsyncWriter
import json
from whoosh.index import open_dir



#创建scheam对象
'''
utter:完整句子，用于全匹配
keyword： 用结巴分词，用于关键词匹配
'''
scheam = Schema(
                                utter = TEXT(stored = True),
                                keywords = TEXT(stored = True,analyzer = ChineseAnalyzer()) )




#创建schema的索引
if not os.path.exists("index"):
    os.mkdir("index")
    idx = create_in("index",scheam)


for path in ['test','train','valid']:
    file_path = "./LCCC/LCCC-base_"+path+".json" #导入json文件
    with open(file_path,'rb') as f:
        data = json.load(f)
    print(f'the data set contains: # {len(data)}')
    idx = open_dir("index")
    for i in range(0,len(data)-1):
        for utter in data[i]:
            writer = idx.writer()
            utter =  utter.replace(" ","") #去掉json中的空格
            writer.add_document(utter = utter,keywords = utter )
            writer.commit()
    idx.close()

'''
此处的导入方式：一条存一句
''' 


