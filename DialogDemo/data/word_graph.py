#!/usr/bin/env python
# coding: utf-8

# In[1]:


from LAC import LAC
import json
from tqdm import tqdm
import re
from gensim.models.keyedvectors import KeyedVectors
import os
import pickle
import networkx as nx


# In[2]:


'''
参数说明：
1）stopword_file：指明stopwords.txt的路径

功能说明：
1）加载stopword

文件格式:
词1'\n'
词2'\n'
...

'''
stopword_file = 'stopwords.txt'
def load_stopwords():
    with open(stopword_file) as f:
        words = f.read().split('\n')
        word = [word for word in words if word.strip()]
    return word


# In[3]:


"""
功能说明：
去除掉不何规范的词
"""
def filter(words):
    def isChinese():
        for word in words:
            if not '\u4e00' <= word <='\u9fff':
                return False
        return True
    def HaveDigital():
        if bool(re.search(r'\d',words)):
            return True
        else:
            return False
    def HaveAlpha():
        for word in words:
            if word.encode().isalpha():
                return True
        return False
    def HaveSpecialWord():
        for word in words:
            if word in set('一二三四五六七八九十月日周年区东西南北。，|；“”‘’！~·、：=-+#￥%……&*（）【】@？.,?[]{}()!$^`";:'):
                return True
        return False
    def CheckTag():
        word,pos = lac.run(words)
        if len(word) != 1:
            return False
        if pos[0] in set(['n', 'nz', 'nw', 'v', 'vn', 'a', 'ad', 'an']): #!!!这里是去掉了（ORG,PER,LOC)
            return True
        else: 
            return False
    
    def CheckLength():
        '''only save tokens which length from 2 to 4'''
        if 1 < len(words) < 5:
            return True
        else:
            return False
    
    return isChinese() and (not HaveDigital()) and (not HaveAlpha()) and (not HaveSpecialWord()) and CheckLength() and CheckTag()        


# In[4]:


'''
功能说明：
将处理好的词向量重新写入txt文件

文件格式
词的个数 300维向量
词1    向量
词2    向量
...    ...
'''
def write_new_w2v(words,path):  
    with open(path,'w') as f:
        f.write(f'{len(words)} 300\n')     
        for word in tqdm(words):
            vec = w2v[word].tolist()
            string = f'{word} {" ".join(map(str,vec))}\n'
            f.write(string)


# In[10]:


'''
功能说明：
1）从chinese_w2v.txt中读取词向量，去除stopword和不规范词后，写入chinese_w2v_base.txt文件
'''
if not os.path.exists('chinese_w2v_base.txt'): #处理得到 chinese_w2v_base.txt文件
    
    w2v = KeyedVectors.load_word2vec_format('chinese_w2v.txt',binary=False) #False表示为text格式，True表示为 bin格式
    print(f'[!] laod the word2vec from chinese_w2v.txt')
    
    # remove the unsatisfied word
    wordlist  = w2v.index2word
    new_wordlist = [word for word in tqdm(wordlist) if filter(word)]
    print(f'[!] squeeze the wordlist from {len(wordlist)} to {len(new_wordlist)}')
    
    # remove the stop word
    new_wordlist_ = list(set(new_wordlist)-set(stopWords))
    print(f'[!] squeeze the wordlist from {len(new_wordlist)} to {len(new_wordlist_)}')
    
    # 将处理好的存入文件
    write_new_w2v(new_wordlist_,'chinese_w2v_base.txt')
    print(f'[!] write the new w2v into chinese_w2v_base.txt')



#作图
if not os.path.exists('wordnet.pk1'):
    #加载 chinese_w2v_base来作图
    w2v = KeyedVectors.load_word2vec_format('chinese_w2v_base.txt',binary=False)
    print(f'[!] laod the [new] word2vec from chinese_w2v_base.txt')
    graph = nx.Graph()
    graph.add_nodes_from(w2v.index2word)
    for word in tqdm(w2v.index2word):
        neighbors = w2v.most_similar(word,topn=10)
        graph.add_weighted_edges_from([(word,n,1-w) for n,w in neighbors if 1-w < 0.6])
    with open('wordnet.pk1','wb') as f:
        pickle.dump(graph,f)
        print(f'[!] save the word net into wordnet.pkl')
else:#如果此时已经存在了图
    with open('wordnet.pk1','rb') as f:
        graph = pickle.load(f)


# In[ ]:




