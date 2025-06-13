from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel

import torch
from torch import nn
import torch.nn.functional as F

class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.distill = config.get('distill', False)
        
        # --- Lấy config cho VOP ---
        # Nếu không có section 'vop' trong config, prompt_config sẽ là None
        # Biến này sẽ được truyền vào các encoder ở các bước sau
        prompt_config = config.get('vop', None) 
        # ------------------------

        embed_dim = config['embed_dim']        
        vision_width = config['vision_width']  
        
        self.num_frames_per_video = config['num_frames_per_video'] 

        # Trong các bước tiếp theo, chúng ta sẽ thêm `prompt_config=prompt_config` vào đây
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))    

        bert_config = BertConfig.from_json_file(config['bert_config'])
        # Trong các bước tiếp theo, chúng ta sẽ thêm `prompt_config=prompt_config` vào đây
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)      

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)   

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])   
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']  
        self.itm_head = nn.Linear(text_width, 2) 
        
        # Momentum models không cần prompt và không cần thay đổi
        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)           
        self.text_proj_m = nn.Linear(text_width, embed_dim)   

        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]
        self.copy_params()

        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("idx_queue", torch.full((1,self.queue_size),-100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        
        # --- ĐÓNG BĂNG BACKBONE NẾU PROMPT TUNING ĐƯỢC KÍCH HOẠT ---
        if prompt_config:
            print("--- Prompt Tuning is enabled. Freezing backbone encoders. ---")
            
            # Đóng băng toàn bộ visual encoder trước
            # Ở các bước sau, chúng ta sẽ thêm logic để không đóng băng các prompt
            for param in self.visual_encoder.parameters():
                param.requires_grad = False

            # Đóng băng toàn bộ text encoder trước
            # Ở các bước sau, chúng ta sẽ thêm logic để không đóng băng các prompt
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        # ----------------------------------------------------


    def forward(self, video, text, alpha, idx): 
        B, num_frames_video, C, H, W = video.shape 
        
        video_flat = video.view(B * num_frames_video, C, H, W)
        
        image_embeds_flat = self.visual_encoder(video_flat) 
        image_feat_flat = F.normalize(self.vision_proj(image_embeds_flat[:,0,:]),dim=-1) 

        image_feat_per_frame = image_feat_flat.view(B, num_frames_video, -1)
        image_feat = image_feat_per_frame.mean(dim=1) 
        
        image_embeds_fusion = image_embeds_flat.view(B, num_frames_video, image_embeds_flat.size(1), -1).mean(dim=1)
        
        image_atts = torch.ones(image_embeds_fusion.size()[:-1],dtype=torch.long).to(video.device)
            
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                 
        
        # Xử lý loss ITA tùy thuộc vào distill có được bật hay không
        if self.distill:
            with torch.no_grad():
                self._momentum_update()
                image_embeds_m_flat = self.visual_encoder_m(video_flat) 
                image_feat_m_flat = F.normalize(self.vision_proj_m(image_embeds_m_flat[:,0,:]),dim=-1)  
                image_feat_m = image_feat_m_flat.view(B, num_frames_video, -1).mean(dim=1)
                
                text_output_m = self.text_encoder_m(text.input_ids, attention_mask = text.attention_mask, return_dict = True, mode = 'text')    
                text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 

                # Lấy queue từ self
                image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)                                         
                text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)
                
                sim_i2t_m = image_feat_m @ text_feat_all / self.temp 
                sim_t2i_m = text_feat_m @ image_feat_all / self.temp   

                idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()],dim=1)  
                pos_idx = torch.eq(idx, idx_all).float()       
                sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)   
                
                sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
            
            sim_i2t = image_feat @ text_feat_all / self.temp
            sim_t2i = text_feat @ image_feat_all / self.temp
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 
            
            # Dequeue and enqueue chỉ được gọi khi distillation được bật
            self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)
        
        else: # Nếu không distill, tính loss trực tiếp với in-batch negatives
            sim_i2t = image_feat @ text_feat.t() / self.temp
            sim_t2i = text_feat @ image_feat.t() / self.temp
            
            targets = torch.arange(B).to(video.device)
            loss_i2t = F.cross_entropy(sim_i2t, targets)
            loss_t2i = F.cross_entropy(sim_t2i, targets)

        loss_ita = (loss_i2t + loss_t2i) / 2

        ###=================================###
        # ITM loss (phần này không thay đổi)
        output_pos = self.text_encoder(encoder_embeds = text_embeds, 
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = image_embeds_fusion, 
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )            
        with torch.no_grad():
            bs = video.size(0)      
            
            # Khi không distill, sim_i2t/sim_t2i được tính với in-batch, nên kích thước khác
            if self.distill:
                sim_i2t_for_neg = sim_i2t[:, :bs]
                sim_t2i_for_neg = sim_t2i[:, :bs]
            else:
                sim_i2t_for_neg = sim_i2t
                sim_t2i_for_neg = sim_t2i

            weights_i2t = F.softmax(sim_i2t_for_neg + 1e-4, dim=1)
            weights_t2i = F.softmax(sim_t2i_for_neg + 1e-4, dim=1)

            mask = torch.eq(idx, idx.T)
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i.masked_fill_(mask, 0) 

        video_embeds_neg = []    
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            video_embeds_neg.append(image_embeds_fusion[neg_idx]) 
        video_embeds_neg = torch.stack(video_embeds_neg,dim=0)   

        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg,dim=0)   
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg],dim=0)     
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0)     

        video_embeds_all = torch.cat([video_embeds_neg,image_embeds_fusion],dim=0)
        video_atts_all = torch.cat([image_atts,image_atts],dim=0)

        output_neg = self.text_encoder(encoder_embeds = text_embeds_all, 
                                        attention_mask = text_atts_all,
                                        encoder_hidden_states = video_embeds_all, 
                                        encoder_attention_mask = video_atts_all,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )                         

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.itm_head(vl_embeddings)            

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(video.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)     

        return loss_ita, loss_itm 
 
    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False    

    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idx):
        image_feats = concat_all_gather(image_feat) 
        text_feats = concat_all_gather(text_feat)
        idxs = concat_all_gather(idx)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  

        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size 

        self.queue_ptr[0] = ptr  
        

@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output        