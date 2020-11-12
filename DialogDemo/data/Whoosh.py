

import os
from whoosh.index import create_in
from whoosh.index import open_dir
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
                                context = TEXT(stored = True),
                                response = TEXT(stored=True),
                                context_keywords = TEXT(stored = True,analyzer = ChineseAnalyzer()),
                                response_keywords = TEXT(stored = True,analyzer = ChineseAnalyzer()) 
)

#创建schema的索引
if not os.path.exists("index"):
    os.mkdir("index")
    idx = create_in("index",scheam)
    idx.close()


for path in ['test','train','valid']:
    file_path = "./LCCC/LCCC-base_"+path+".json" #导入json文件
    with open(file_path,'rb') as f:
        data = json.load(f)
    print(f'the data set contains: # {len(data)}')
    idx = open_dir("index")
    for i in range(len(data)):
        for j in range(len(data[i])-1):
            writer = idx.writer()
            context = data[i][j].replace(" ","")
            response = data[i][j+1].replace(" ","")
            writer.add_document(context = context,response = response,context_keywords = context,response_keywords=response)
            writer.commit()
    idx.close()
'''
此处的导入方式：一条存一句
''' 


