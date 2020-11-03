#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
from whoosh.index import create_in
from whoosh.fields import *
from jieba.analyse import ChineseAnalyzer
from whoosh.writing import AsyncWriter
import json


# In[4]:


#创建scheam对象
'''
utter:完整句子，用于全匹配
keyword： 用结巴分词，用于关键词匹配
'''
scheam = Schema(
                                utter = TEXT(stored = True),
                                keywords = TEXT(stored = True,analyzer = ChineseAnalyzer()) )


# In[5]:


#创建schema的索引
if not os.path.exists("index"):
    os.mkdir("index")
idx = create_in("index",scheam)


# In[6]:


file_path = "./LCCC/LCCC-base_train.json" #导入json文件


# In[7]:


with open(file_path,'rb') as f:
    data = json.load(f)


# In[8]:


print(f'the data set contains: # {len(data)}')


# In[9]:


writer = AsyncWriter(index=idx)
'''
由于不是很清楚后面retrieval的时候到底怎么存，所以这里先导入1001条，方便后续修改
此处的导入方式：一条存一句
''' 
for i in range(0,1000):
    for utter in data[i]:
        utter =  utter.replace(" ","") #去掉json中的空格
        writer.add_document(utter = utter,keywords = utter )
writer.commit()


# In[ ]:


# 采用indexWriter写入数据
#writer = AsyncWriter(index=idx)
#for utters in data:
#    for utter in data:
#        writer.add_document(utter = utter,keywords = utter )
#writer.commit()


# In[61]:


# =============查找测试================
from whoosh.qparser import QueryParser
parser = QueryParser("keyword", idx.schema)


# In[69]:


with idx.searcher() as searcher:
    query = parser.parse(u"回")
    results = searcher.search(query)


# In[71]:


for i in range(min(10,len(result))):
    print(result[i])


# In[67]:


len(result)
#============查找测试结束==================


# In[ ]:





# In[ ]:





# In[ ]:




