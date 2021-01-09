from model import *
from header import*
# import torch.nn as nn

class PostTrainingBert(nn.Module):
    def __init__(self,lang='zh'):
        super(BertEmbedding_Post,self).__init__()
        model_name = 'bert-base-chinese' if lang == 'zh' else 'bert-base-uncased'
        self.model = BertForPreTraining_for_PostTraining.from_pretrained(model_name)
    
    # 注意这里传入的 label 的形式 [B,]
    # ? attn_mask里 [mask]应该为 0 还是1
    def forward(self,ids,attn_mask,labels,next_sentence_label):
        outputs = self.model(ids = ids,attn_mask =attn_mask,labels = labels ,next_sentence_label= next_sentence_label)  
        mlm_loss, nsp_loss, prediction_scores, seq_relationship_score = bert_outputs[:4]

        
        return mlm_loss, nsp_loss
    



class PostTrainingAgent(RetrievalBaseAgent):
    # 这个init部分不知道对不对呜呜呜
    def __init__(self, multi_gpu, total_step, run_mode='train', local_rank=0, kb=True, lang='zh'):
        super(BERTBiEncoderAgent, self).__init__(kb=kb)
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        
        self.args = {
            'lr': 5e-5,
            'lr_': 5e-4,
            'grad_clip': 1.0,
            'multi_gpu': self.gpu_ids,
            'talk_samples': 256,
            'vocab_file': 'bert-base-chinese' if lang == 'zh' else 'bert-base-uncased',
            'pad': 0,
            'samples': 10,
            'model': 'bert-base-chinese' if lang == 'zh' else 'bert-base-uncased',
            'amp_level': 'O2',
            'local_rank': local_rank,
            'warmup_steps': int(0.1 * total_step),
            'total_step': total_step,
            'retrieval_model': model,
            'num_encoder_layers': 4,
            'dim_feedforward': 2048,
            'nhead': 8,
            'dropout': 0.1,
            'max_len': 512,
            'poly_m': 16,
            'lang': lang,
        }
        self.vocab = BertTokenizer.from_pretrained(self.args['vocab_file'])
        self.model = PostTrainingBert(lang=self.args['lang'])
        if torch.cuda.is_available():
            self.model.cuda()
        if run_mode == 'train':
            self.optimizer = transformers.Adamw(
                self.model.parameters(), 
                    lr=self.args['lr']
            )
            self.model, self.optimizer = amp.initialize(
                self.model, 
                self.optimizer,
                opt_level=self.args['amp_level'],
            )
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=self.args['warmup_steps'], 
                num_training_steps=total_step,
            )
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True,
            )
        self.show_parameters(self.args)
        
        def train_model(self,train_iter,mode = 'train',recoder = None,idx_ = 0):
            self.model.train()
            mlm_loss_total,nsp_loss_total,batch_num = 0
            pbar = tqdm(train_iter)
            for idx,batch in enumerate(pbar):
                self.optimizer.zero_grad()
                ids ,attn_mask,labels,next_sentence_label= batch
                
                losses = self.model(ids = ids,attn_mask =attn_mask,labels = labels ,next_sentence_label= next_sentence_label)
                mlm_loss,nsp_loss = losses
                mlm_loss = mlm_loss.mean()
                nsp_loss = nsp_loss.mean()
                mlm_loss_total += mlm_loss
                nsp_loss_total += nsp_loss
                loss = mlm_loss_total + nsp_loss_total
                
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
                self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss.item()
                batch_num += 1
                if recoder is not None:
                    recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
                    recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
                pbar.set_description(f'[!] loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}')
            if recoder is not None:
                recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)      
                
                    
            return round(total_loss / batch_num, 4)
        
        @torch.no_grad()
        def test_model(self,test_iter,recoder=None,idx_=0):
            pass
            



class RetrievalBaseAgent:

    def __init__(self, searcher=True, kb=True):
        if searcher:
            self.searcher = ESChat('retrieval_database', kb=kb)
        self.history = []    # save the history during the SMP-MCC test

    def show_parameters(self, args):
        print('========== Model ==========')
        print(self.model)
        print('========== Model ==========')
        print(f'========== Model Parameters ==========')
        for key, value in args.items():
            print(f'{key}: {value}')
        print(f'========== Model Parameters ==========')

    def save_model(self, path):
        try:
            state_dict = self.model.module.state_dict()
        except:
            state_dict = self.model.state_dict()
        torch.save(state_dict, path)
        print(f'[!] save model into {path}')
    
    def load_model(self, path):
        '''
        add the `module.` before the state_dict keys if the error are raised,
        which means that the DataParallel(self.model) are used to load the model
        '''
        state_dict = torch.load(path)
        try:
            self.model.load_state_dict(state_dict)
        except:
            current_module = True if 'module' in [i[0] for i in self.model.state_dict().items()][0] else False
            saved_module = True if 'module' in [i[0] for i in state_dict.items()][0] else False
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if saved_module and not current_module:
                    name = k[7:]
                    new_state_dict[name] = v
                elif not saved_module and current_module:
                    name = f"module.{k}"
                    new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
        print(f'[!] load model from {path}')

    def train_model(self, train_iter, mode='train'):
        raise NotImplementedError

    def test_model(self, test_iter, path):
        raise NotImplementedError
        
    def process_utterances_biencoder(self, topic, msgs, max_len=0):
        def _length_limit(ids):
            if len(ids) > max_len:
                ids = [ids[0]] + ids[-(max_len-1):]
            return ids
        utterances = self.searcher.search(msgs, samples=self.args['talk_samples'], topic=topic)
        utterances = [i['utterance'] for i in utterances]
        utterances = list(set(utterances) - set(self.history))
        inpt_ids = self.vocab.batch_encode_plus([msgs] + utterances)['input_ids']
        context_inpt_ids, response_inpt_ids = inpt_ids[0], inpt_ids[1:]
        context_inpt_ids = torch.LongTensor(_length_limit(context_inpt_ids))
        response_inpt_ids = [torch.LongTensor(_length_limit(i)) for i in response_inpt_ids]
        response_inpt_ids = pad_sequence(response_inpt_ids, batch_first=True, padding_value=self.args['pad'])
        attn_mask_index = response_inpt_ids.nonzero().tolist()
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(response_inpt_ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        
        if torch.cuda.is_available():
            context_inpt_ids, response_inpt_ids, attn_mask = context_inpt_ids.cuda(), response_inpt_ids.cuda(), attn_mask.cuda()
        return utterances, context_inpt_ids, response_inpt_ids, attn_mask

    def process_utterances(self, topic, msgs, max_len=0, context=True):
        '''Process the utterances searched by Elasticsearch; input_ids/token_type_ids/attn_mask'''
        if not context:
            msgs = ''
        utterances_ = self.searcher.search(msgs, samples=self.args['talk_samples'], topic=topic)
        utterances_ = [i['utterance'] for i in utterances_]
        # remove the utterances that in the self.history
        utterances_ = list(set(utterances_) - set(self.history))
        
        # construct inpt_ids, token_type_ids, attn_mask
        inpt_ids = self.vocab.batch_encode_plus([msgs] + utterances_)['input_ids']
        context_inpt_ids, responses_inpt_ids = inpt_ids[0], inpt_ids[1:]
        context_token_type_ids = [0] * len(context_inpt_ids)
        responses_token_type_ids = [[1] * len(i) for i in responses_inpt_ids]
        
        # length limitation
        collection = []
        for r1, r2 in zip(responses_inpt_ids, responses_token_type_ids):
            p1, p2 = context_inpt_ids + r1[1:], context_token_type_ids + r2[1:]
            if len(p1) > max_len:
                cut_size = len(p1) - max_len + 1
                p1, p2 = [p1[0]] + p1[cut_size:], [p2[0]] + p2[cut_size:]
            collection.append((p1, p2))
            
        inpt_ids = [torch.LongTensor(i[0]) for i in collection]
        token_type_ids = [torch.LongTensor(i[1]) for i in collection]
        
        inpt_ids = pad_sequence(inpt_ids, batch_first=True, padding_value=self.args['pad'])
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=self.args['pad'])
        attn_mask_index = inpt_ids.nonzero().tolist()
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(inpt_ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        
        if torch.cuda.is_available():
            inpt_ids, token_type_ids, attn_mask = inpt_ids.cuda(), token_type_ids.cuda(), attn_mask.cuda()
        return utterances_, inpt_ids, token_type_ids, attn_mask

    def talk(self, msgs, topic=None):
        '''
        topic: topic of the conversation
        msgs: a string of the conversation context
        '''
        raise NotImplementedError

    def get_res(self, data):
        '''
        data = {
            "kg_path": group_id,
            "topic": topic,
            "node": robot_id
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
        msgs = [i['msg'] for i in data['msgs']]
        msgs = ' [SEP] '.join(msgs)
        topic = data['topic'] if 'topic' in data else None
        res = self.talk(msgs, topic=topic)
        self.history.append(res)
        return res