#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
from whoosh.index import open_dir
from whoosh.fields import *
from whoosh.query import*
from jieba.analyse import ChineseAnalyzer
from whoosh.writing import AsyncWriter
import json


# In[47]:


class WhooshChat:
    '''basic whoosh searcher'''
    '''
    scheam = Schema(utter = TEXT(stored = True),keywords = TEXT(stored = True,analyzer = ChineseAnalyzer()) )
    '''
    def __init__(self,index_name):
        self.idx = open_dir(index_name)
    '''
    输入：msgs：信息（string），topics（[]）,sample（查询返回条数）
    返回：查询结果([])
    '''
    def search(self,msgs=None,topics=None,sample = 10):
        query = []
        if msgs:
            query.append(Term("content",msgs,boost = 1)) #句子的增强因子boost = 1
        if topics:
            for topic in topics:
                query.append(Term("keywords",topic,boost=7)) # 关键词的增强因子为7
        query = Or(query)
        result = []
        
        with self.idx.searcher() as searcher:
            res = searcher.search(query,limit=sample)
            for hit in res:
                data ={
                "utter":hit["utter"]
                }
                result.append(data)
        return result
            
            
        


# In[49]:


# ==================测试代码部分 =============
whoosh = WhooshChat('index')
whoosh.search(topics=['你','我','他','秃头'])


# In[ ]:




