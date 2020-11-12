

import os
from whoosh.index import open_dir
from whoosh.fields import *
from whoosh.query import*
from jieba.analyse import ChineseAnalyzer
from whoosh.writing import AsyncWriter
import json


# In[89]:


class WhooshChat:
    '''basic whoosh searcher'''
    '''
    scheam = Schema(
                                context = TEXT(stored = True),
                                response = TEXT(stored=True),
                                context_keywords = TEXT(stored = True,analyzer = ChineseAnalyzer()),
                                response_keywords = TEXT(stored = True,analyzer = ChineseAnalyzer()) 
)
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
            query.append(Term("context",msgs,boost = 1)) #句子的增强因子boost = 1
        if topics:
            for topic in topics:
                query.append(Term("context_keywords",topic,boost=7)) # 关键词的增强因子为7
        query = Or(query)
        result = []
        
        with self.idx.searcher() as searcher:
            res = searcher.search(query,limit=sample)
            print(res)
            for hit in res:
                data ={
                "utter":hit["response"]
                }
                result.append(data)
        return result
    
    def talk(self, msgs, topic=None):
        print(topic)
        rest = self.search(msgs, sample=1, topics=topic)
        if not rest:
            print('f[!] there is no response in the dataset')
        else:
            rest = rest[0]['utter']
        return rest
            
            
        


# In[90]:


# ==================测试代码部分 =============
if __name__ == 'main':
    whoosh = WhooshChat('index')
    whoosh.search(topics=['你','我','他'])





