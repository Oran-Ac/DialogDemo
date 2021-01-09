from header import*
# import random
    
def read_data_for_PostTraining(path, lang='zh',max_predictions_per_seq=25,masked_lm_prob=0.15):
    spliter = '' if lang == 'zh' else ' '
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            line = line.strip().split('\t')
            label, uttrs = int(line[0]), line[1:]
            
            tokens = []
            #tokens.append(' [CLS] ') ,[CLS]自动添加
            for i,uttr in enumerate(uttrs): #得到tokens的数组，方便决定哪一个被遮住
                token = uttr.split()
                tokens = tokens + token
                if i == len(uttrs)-2: #倒数两个
                    tokens.append(' [SEP] ')
                elif i == len(uttrs)-1: #倒数第一个
                    continue
                else:
                    tokens.append(' [EOT] ')
                
            num_to_predict = min(max_predictions_per_seq,max(1,int(round(len(tokens)*masked_lm_prob)))) #被遮住的数目
            cand_indexes = []
            for i in range(len(tokens)):
                if tokens[i] == ' [CLS] ' or tokens[i] == ' [SEP] ' or tokens[i] == ' [EOT] ':
                    continue
                else:
                    cand_indexes.append(i)
            random.shuffle(cand_indexes) #将索引随机排序
            masked_lms = [] #被遮住的 [(索引，单词)]
            covered_indexes = set()
            for index in cand_indexes:
                if len(masked_lms) >= num_to_predict:
                    break
                if i in covered_indexes:
                    continue
                covered_indexes.add(index)
                masked_token = None
                if random.random() < 0.8: # 80% replace
                    masked_token = ' [MASK] '
                    
                elif random.random() < 0.5: # 20% x 0.5 ,keep the original
                    masked_token = tokens[index]
                else:
                    masked_token = tokens[cand_indexes[random.randint(0,len(cand_indexes)-1)]]# 20% x 0.5 ,random choose from the sentence ！！！！！！【注意】：好像是有vocab_file的存在！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
                masked_lms.append({'index':index,'token':tokens[index]})
                tokens[index] = masked_token
            masked_lm_positions = []
            masked_lm_labels = []
            for lm in masked_lms:
                masked_lm_positions.append(lm['index'])
                masked_lm_labels.append(lm['token'])
            
            uttr = spliter.join(tokens)
            #("长句","分词后句","是否为下一句","遮住的原文","遮住的位置")
            dataset.append((uttr,tokens,label,masked_lm_labels,masked_lm_positions))
        return dataset
