#!/usr/bin/env python
# coding: utf-8

# In[33]:


import gensim
import networkx as nx
from LAC import LAC
import numpy as np
import torch
from api_for_Whoosh import WhooshChat
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from jieba import analyse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
import random
PAD = 0

class BERTRerievalenvAgent:
    def __init__(self,wordnet =None,talk_samples = 128,model = 'bert-base-chinese',index = 'index'):
        
        self.history = []
        self.done_reward,self.smooth_penalty,self.step_penalty = 100,20,5
        self.wordnet = wordnet
        self.talk_samples = talk_samples
        self.lac = LAC(mode = 'lac')
        self.model = AutoModelForSequenceClassification.from_pretrained(model,num_labels = 2)
        self.max_len = 10
        self.searcher = WhooshChat(index)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if torch.cuda.is_available():
            self.model.cuda()
    def reset(self):
        self.history = []
    def process_uttr(self,topics = None,msgs = None,max_len=10):
        # get candidate uttr
        candidates = self.searcher.search(msgs=msgs,topics=topics)
        uttr = [candidate["utter"] for candidate in candidates]
        print(uttr)
        # construct input_ids,token_type_ids,attn_mask
        input_ids = self.tokenizer.batch_encode_plus([msgs] + uttr)['input_ids']
        context_input_ids,response_input_ids = input_ids[0],input_ids[1:]
        context_token_type_ids = [0 for _ in context_input_ids]
        response_token_type_ids = [[1 for _ in i] for i in response_input_ids]
        
        # length limitation
        collection = [] #(拼接好的id，拼接好的tokenid)  每一个是一个候选的情况
        for r1,r2 in zip(response_input_ids,response_token_type_ids): #[id,1]
            # 没有看懂这里拼接的时候为什么要把 r1[0]和 r2[0]去除了 A:开头为[CEL]的号
            # ！！！这里是拼接不是求和
            p1 = context_input_ids + r1[1:]
            p2 = context_token_type_ids + r2[1:]
            # len(p1) == len(p2)
            if len(p1) > max_len:
                cut_size = len(p1) - max_len + 1
                # p[0]为[cel]的号
                p1 = [p1[0]] + p1[cut_size:]
                p2 = [p2[0]] + p2[cut_size:]
            collection.append((p1,p2))
        print(collection)
        input_ids = [torch.LongTensor(i[0]) for i in collection]
        token_type_ids = [torch.LongTensor(i[1]) for i in collection]
        print(input_ids)
        input_ids =pad_sequence(input_ids,batch_first = True ,padding_value= PAD) # batch_first ->(batch_Size,input_size,length)
        token_type_ids = pad_sequence(token_type_ids,batch_first = True,padding_value= PAD)
        # attn_mask的处理 1）得到不为0的位置坐标 2）构建与 input 一样大小的 0矩阵  3）由下标将 0 矩阵中的情况置为1
        attn_mask_index = input_ids.nonzero().tolist() #得到 input_ids里不为0的下标情况
        attn_mask_index_x,attn_mask_index_y = [i[0] for i in attn_mask_index] , [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(input_ids)
        attn_mask[attn_mask_index_x,attn_mask_index_y] = 1
        
        if torch.cuda.is_available():
            input_ids,token_type_ids,attn_mask = input_ids.cuda(),token_type_ids.cuda(),attn_mask.cuda()
        return uttr,input_ids,token_type_ids,attn_mask
    
    @torch.no_grad()
    def talk(self, msgs, topic=None):
        self.model.eval()
        utterances_, inpt_ids, token_type_ids, attn_mask = self.process_uttr(
            topic, msgs, max_len=self.max_len,
        )
        # prepare the data input
        output = self.model(inpt_ids, token_type_ids, attn_mask)    # [B, 2]
        #print(f'[!!] the value of output is {output[0]}')
        #print(f'[!]the output type {type(output[0])}')
        output = F.softmax(output[0], dim=-1)[:, 1]    # [B]
        item = torch.argmax(output).item()
        msg = utterances_[item]
        return msg
    @torch.no_grad()
    def get_res(self, data):
        '''return reward and next utterances for the BERTRetrievalEnvAgent'''
        msgs = [i['msg'] for i in data['msgs']]
        # NOTE: in order to make sure the user speak based on the given conversation, use the topic for coarse ranking
        # topic = self.extract_topic_words(msgs[-1])
        topic = None
        msgs = ' [SEP] '.join(msgs)
        res = self.talk(msgs, topic=topic)
        self.history.append(res)
        return res 


# In[37]:


#这个是机器人端方法之一，在self-play里没有用到
class BERTRetrievalKGGreedyAgent:
    def __init__(self,source,target,wordnet = None,index='index',model = 'bert-base-chinese'):
        self.topic_history = []
        self.wordnet = wordnet
        self.w2v = gensim.models.KeyedVectors.load_word2vec_format('/data/home/zzp/jupyter/data/chinese_w2v_base.txt')
        self.source = source
        self.target = target
        self.searcher = WhooshChat(index)
        self.current_node = []
        self.model = AutoModelForSequenceClassification.from_pretrained(model,num_labels = 2)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if torch.cuda.is_available():
            self.model.cuda()
    def reset(self,target,source):
        self.target = target
        self.source = source
        self.current_node = [source]
        self.topic_history = [source]
        self.history = []
        print(f'[!] the dialog should move from source: {source} to target:  {target}')  
        
    def search_candidates(self,msgs,nodes):
        '''
        f(n) = h(n)+g(n)
        '''
        # generate the candidate
        candidates = []
        for node in nodes:
            neighbors = []
            # ????在建图的时候不已经用 similarity来算边和边的权重了吗 ？？ 直接搜图不香吗
            # A：这里是在进行 A* 算法
            base_dis = self.w2v.similarity(node,self.target)
            for n in self.wordnet.neighbors(node):
                if self.w2v.similarity(n,self.target)>= base_dis:
                    continue
                rerieval_rest = self.searcher.search(msgs = msgs,topics = [node,n])
                if not rerieval_rest: # 啥都没有找到
                    continue
                try:
                    path = nx.shortest_path(self.wordnet,n,self.target) # ?？这里为什么直接最短路径
                except nx.NetworkXNoPath as e:
                    continue
                neighbors.append((node,n,path))
            candidates.extend(neighbors)
        print(f'[!]the candidates are{candidates}')
        candidates.sort(key = lambda i:len(i[2])) #要单独写一行，不然 就得到的是 .sort()的返回值None
        print(f'[!]after sorted{candidates}')
        return candidates #按路径长度降序输出排序
        # score the f(n) and sort 
        #pass
   
    def move_on_kg(self,msgs,current_node,size =1):
        '''current nodes are extracted from the human utterance(maybe multiple)'''
        
        candidates = self.search_candidates(msgs,current_node) #candidates是个列表
        candidates = candidates[:size]
        topics = []
        for candidate in candidates:
            topics.append(candidate[1]) # 0->node 1->下一节点 2->path 3->长度
        print(topics)
        return topics #这里的topics是下一层里潜在的，可能是cluster（如果size不为1）
    
    def process_utterances(self,topics = None,msgs = None,max_len=10):
        '''copy from bert_retieval_base'''
        # get candidate uttr
        candidates = self.searcher.search(msgs=msgs,topics=topics)
        uttr = [candidate["utter"] for candidate in candidates]
        
        # construct input_ids,token_type_ids,attn_mask
        input_ids = self.tokenizer.batch_encode_plus([msgs] + uttr)['input_ids']
        context_input_ids,response_input_ids = input_ids[0],input_ids[1:]
        context_token_type_ids = [0 for _ in context_input_ids]
        response_token_type_ids = [[1 for _ in i] for i in response_input_ids]
        
        # length limitation
        collection = [] #(拼接好的id，拼接好的tokenid)  每一个是一个候选的情况
        for r1,r2 in zip(response_input_ids,response_token_type_ids): #[id,1]
            # 没有看懂这里拼接的时候为什么要把 r1[0]和 r2[0]去除了 A:开头为[CEL]的号
            # ！！！这里是拼接不是求和
            p1 = context_input_ids + r1[1:]
            p2 = context_token_type_ids + r2[1:]
            # len(p1) == len(p2)
            if len(p1) > max_len:
                cut_size = len(p1) - max_len + 1
                # p[0]为[cel]的号
                p1 = [p1[0]] + p1[cut_size:]
                p2 = [p2[0]] + p2[cut_size:]
            collection.append((p1,p2))
        
        input_ids = [torch.LongTensor(i[0]) for i in collection]
        token_type_ids = [torch.LongTensor(i[1]) for i in collection]
        
        input_ids =pad_sequence(input_ids,batch_first = True ,padding_value= PAD) # batch_first ->(batch_Size,input_size,length)
        token_type_ids = pad_sequence(token_type_ids,batch_first = True,padding_value= PAD)
        # attn_mask的处理 1）得到不为0的位置坐标 2）构建与 input 一样大小的 0矩阵  3）由下标将 0 矩阵中的情况置为1
        attn_mask_index = input_ids.nonzero().tolist() #得到 input_ids里不为0的下标情况
        attn_mask_index_x,attn_mask_index_y = [i[0] for i in attn_mask_index] , [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(input_ids)
        attn_mask[attn_mask_index_x,attn_mask_index_y] = 1
        
        if torch.cuda.is_available():
            input_ids,token_type_ids,attn_mask = input_ids.cuda(),token_type_ids.cuda(),attn_mask.cuda()
        return uttr,input_ids,token_type_ids,attn_mask
    
    @torch.no_grad()
    def talk(self,msgs,topics):
        '''topic means the current topic node in the knowledge graph path'''
        #self.model.eval()   # !!!model未导入
        # 1) input the topic information for whoosh
        utterances,input_ids,token_type_ids,attn_mask = self.process_utterances(
            topics,msgs
        )
        # 2) nerual ranking with the topic information
        #!!! 下面的都没有导入
        output = self.model(input_ids,token_type_ids,attn_mask)
        output = F.softmax(output[0],dim =-1)[:1]
        output = torch.argsort(output,descending = True)
        item = torch.argmax(output).item()
        msg = utterances[item]
        return msg
    
    def obtain_keywords(self, utterance):
        '''select the keyword that most similar to the current_node as the keyword in the human response'''
        keywords = analyse.extract_tags(utterance)
        print(f'[!] the keywords in the sentence are {keywords}')
        nodes = [i for i in keywords if i in self.wordnet.nodes]
        assert len(nodes) != 0, f'[!] cannot find the keywords in the human utterances'
        keyword = random.choice(nodes) #随便选一个，但这个就有很大改进空间了！！！
        return [keyword]
    
    def get_res(self, data):
        '''
        data = {
            "msgs": [
                {
                    'fromUser': robot_id,
                    'msg': msg,
                    'timestamp': timestamp
                },
                ...
            ]
        }
        ''' 
        if len(data['msgs']) > 0:
            # 1) move
            response = data['msgs'][-1]['msg']
            keyword = self.obtain_keywords(response)
            print(f'the keywords {keyword}')
            print(f'the current_node {self.current_node}')
            if self.current_node:
                self.current_node = list(set(keyword+self.current_node))
            else:
                self.current_node = list(set(keyword))
            topic = self.move_on_kg(response,keyword)
            self.topic_history.append(topic)
            # 2) obtain the responses based on the next_node
            msgs = [i['msg'] for i in data['msgs']]
            msgs = ' [SEP] '.join(msgs)
            res = self.talk(msgs,topic)
        else:
            # ?????
            res = self.searcher.talk('', topic=self.current_node)
        self.history.append(res)
        return res
    
                
            
            
            


# In[43]:


# 我咋觉得self-play没有用这个呢 process_utt-----都没有
class BERTRetrievalClusterGreedyAgent:
    def __init__(self,source,target,wordnet = None,index='index',model = 'bert-base-chinese'):
        self.topic_history = []
        self.wordnet = wordnet
        self.w2v = gensim.models.KeyedVectors.load_word2vec_format('/data/home/zzp/jupyter/data/chinese_w2v_base.txt')
        self.source = source
        self.target = target
        self.searcher = WhooshChat(index)
        '''额外的'''
        self.number_candidate = 5
        self.cluster_width = 50
        self.cutter = LAC(mode = 'seg')
        self.g_scorer = BERTBiEncoder(share=True) #？？？没有找到这个函数
        self.current_node = []
        self.model = AutoModelForSequenceClassification.from_pretrained(model,num_labels = 2)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if torch.cuda.is_available():
            self.model.cuda()
    def reset(self):
        self.current_node.append(self.source)
        self.history = []
        self.topic_history = [self.source]
    def obtain_keywords(self,utterance):
        return [i for i in self.cutter.run(utterance) if i in self.w2v] #????这选keyword的方法是不是有点太草率了哦--离谱
    
    def get_candidates(self,start_node,size=5):
        # obtain not only the 1-hop neighboor
        path_lengths = nx.single_source_shortest_path_length(self.wordnet,start_node,cutoff=2)
        candidates - [node for node, length in path_lengths.items() if length <=2]
        dis = self.w2v.most_similar(start_node,self.target)
        # closer and have the path to the target will be consider as candidates
        nodes = []
        for candidate in candidates:
            if self.w2v.similarity(candidate,self.target) > dis:
                path = nx.dijkstra_path(self.wordnet,candidate,self.target)
                if path:
                    nodes.append((candidate,path))
        return nodes
    
    def get_g_function(self,msgs,node):
        # retrieval a bag of response
        # ????没有找到这个函数 process_utterances_biencoder
        utterances, context_inpt_ids, response_inpt_ids, attn_mask = self.process_utterances_biencoder([node], msgs, max_len=10)
        #？？？
        scores = self.g_scorer.predict(context_inpt_ids, response_inpt_ids, attn_mask).tolist()
        return np.mean
    
    def get_h_function(self,path):
        # search the path
        # calculate the path
        #????
        pass
    @torch.no_grad()
    def move_on_kg(self):
        candidates = self.get_candidates(current_node,size = 5)
        # ??????
        g_scores = [self.get_g_function(i[0]) for i in candidates]
        h_scores = [self.get_h_function(i[1]) for i in candidates]
        # resort
        scores = []
        # ?????
        for g,h,candidate in candidates:
            scores.append((g+h)/2 ,candidate)
        scores = sorted(scores,key=lambda x:x[0],reverse=True)
        self.current_node = scores[0][1]
        if self.current_node == self.target:
            pass
    
    '''topic means the current topic node in the knowledge graph path'''
    @torch.no_grad()
    def talk(self,msgs):
        
       
        #self.model.eval()   # ???!!!model未导入
        # 1) input the topic information for whoosh
        utterances,input_ids,token_type_ids,attn_mask = self.process_utterances(topics,msgs)
        
        # 2) nerual ranking with the topic information
        #!!! 下面的都没有导入
        output = self.model(input_ids,token_type_ids,attn_mask)
        output = F.softmax(output[0],dim =-1)[:1]
        output = torch.argsort(output,descending = True)
        # 3) post ranking with multiple current topic words
        for i in output:
            flag,chosen_word = False,None
            for word in self.current_node:
                if word in utterances[i.item()]:
                    item,flag = i,True
                    chosen_word = word
                    break
            if flag:
                break
        else:
            item = 0
        msg = utterance[item]
        return msg,chosen_wordi
    
    def get_res(self,data):
        '''
        data = {
            "msgs": [
                {
                    'fromUser': robot_id,
                    'msg': msg,
                    'timestamp': timestamp
                },
                ...
            ]
        }
        ''' 
        if len(data['msgs']) > 0:
            # 1) move
            response = data['msgs'][-1]['msg']
            keywords = self.obtain_keywords(response)
            if self.current_node:
                self.current_node = list(set(keywords + self.current_node))
            else:
                self.current_node = list(set(keywords))
            self.move_on_kg()
            # 2) obtain the response based on the next_node
            msgs = [i['msg'] for i in data['msgs']]
            msgs = '[SEP] '.join(msgs)
            res,chosen_word = self.talk(msgs)
            self.topic_history.append(self.current_node)
            self.current_node = chosen_word
        else:
            res = self.searcher.talk('', topic=self.current_node)
        self.history.append(res)
        return res
        


# In[ ]:




