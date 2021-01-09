# from torch.utils.data import Dataset
# from transformers import BertTokenizer
from utils import *
# import os
class PostTrainingData(Dataset):
    def __init__(self,path,mode='train',max_len=300,lang='zh'):
        self.mode = mode
        self.max_len = max_len
        if lang == 'zh':
            self.vocab = self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')
        else:
            self.vocab = BertTokenizer.from_pretrained('bert-base-uncased')
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_ruirbi.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        if mode == 'train':
            data = read_data_for_PostTraining(path,lang=lang)
        else:
            pass
        self.data = []
        if mode in ['train','dev']:
            for utter,_,isNextLabel,masked_labels,masked_position in tqdm(data):
                item = self.vocab.encode_plus(utter)
                ids = item['input_ids']
                ids = self._length_limit(ids)
                masked_label_ids = []
                for masked_label in masked_labels:
                    masked_label_ids.append(self.vocab.encoder(masked_label)[1:-1]) #去掉自动添加的首尾
                self.data.append({'ids': ids,'isNextLabel':isNextLabel,'masked_ids':masked_label_ids,'masked_position':masked_position})
        else:
            pass
    
    def _length_limit(self, ids):
        if len(ids) > self.max_len:
            ids = [ids[0]] + ids[-(self.max_len-1):]
        return ids
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        pass
    
    def save_pickle(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save dataset into {self.pp_path}')
        
    def generate_mask(self, ids):
        attn_mask_index = ids.nonzero().tolist()
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        return attn_mask
    
    def generate_labe(self,ids,masked_position,masked_ids):
        
        labels= torch.zeros_like(ids)
        for batch_num in range(len(labels)): #batch_size
            for j in range(len(labels[batch_num])): #seq_length
                labels[batch_num,j] = -100 #忽略的
                
        for batch_num in range(len(labels)): #batch_size
            for j,location in enumerate(masked_position) : #seq_length
                labels[batch_num,location] = masked_ids[batch_num,j] # 被替换位置实际的值
        return labels    #[B,S]
        
        
    
    '''
   {'ids': ids,'isNextLabel':isNextLabel,'masked_ids':masked_label_ids,'masked_position':masked_position}
   ids ,attn_mask ,labels  ,next_sentence_label
    '''
    def collate(self, batch): # mini-batch
        if self.mode == 'train':
            ids = [i[0] for i in batch]
            isNextLabel = [i[1] for i in batch]
            masked_ids = [i[2] for i in batch]
            masked_position = [i[3] for i in batch]
            ids = pad_sequence(ids,batch_first=True,padding_value = self.pad)
            attn_mask = self.generate_mask(ids)
            labels = self.generate_labe(ids,masked_position,masked_ids)
            next_sentence_label= isNextLabel #[B]
            return ids,attn_mask,labels,next_sentence_label
        
        else:
            pass
            
            
        