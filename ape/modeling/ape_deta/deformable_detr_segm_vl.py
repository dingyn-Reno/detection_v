import copy
import math
import os
import time
from typing import Dict, List, Optional, Tuple
from PIL import Image
import pickle
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import pdb
import fvcore.nn.weight_init as weight_init
from ape.modeling.text import utils as text_utils
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import Conv2d, ShapeSpec, get_norm, move_device_like
from detectron2.modeling import GeneralizedRCNN
from detectron2.modeling.meta_arch.panoptic_fpn import combine_semantic_and_instance_outputs
from detectron2.modeling.postprocessing import detector_postprocess, sem_seg_postprocess
from detectron2.structures import BitMasks, Boxes, ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom
from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import inverse_sigmoid
from torchvision.ops.boxes import batched_nms
from clip import clip
import re
from ape.data.detection_utils import get_fed_loss_cls_weights

from .deformable_detr import DeformableDETR
from .fast_rcnn import fast_rcnn_inference
from .segmentation import MaskHeadSmallConv, MHAttentionMap
import random
from torch.nn.utils.rnn import pad_sequence

class CrossAttention(nn.Module):
    def __init__(self,in_dim=256,out_dim=1024,device='cuda',pre_select=False,query_path='models/city_large_query_50_sel_tiny.pth'):
        super().__init__()
        self.to_q=nn.Linear(out_dim,out_dim)
        self.to_kv=nn.Linear(in_dim,out_dim)
        self.norm = nn.LayerNorm(out_dim)
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_kv.weight)
        if pre_select == True:
            self.pre_to_q=nn.Linear(in_dim,in_dim)
            self.pre_to_kv=nn.Linear(in_dim,in_dim)
        # pdb.set_trace()
        try:
            query=torch.load(query_path)
            arr=[]
            for i in range(1,len(query)+1):
                arr.append(torch.mean(query[i],dim=0).cpu())
            # arr.append(torch.zeros(1,256))
            self.vision_query=torch.cat(arr,dim=0).to(device)
        except:
            self.vision_query=None
            
    def attn(self,q,k,v):
        q=self.pre_to_q(q)
        k=self.pre_to_kv(k)
        v=self.pre_to_kv(v)
        qk=torch.einsum('cf,df->cd',q,k)/torch.sqrt(torch.tensor(256, dtype=torch.float))
        qk=torch.softmax(qk,dim=-1)
        ret=torch.einsum('cd,df->cf',qk,v)
        return ret
    def forward(self,query,pre_select=None):
        if self.vision_query is None:
            return 0.0
        if pre_select!=None:
            pre_select=torch.mean(pre_select,dim=0)
            vision_query=self.attn(self.vision_query,pre_select,pre_select)
        else:
            vision_query=self.vision_query
        # pdb.set_trace()
        query=self.norm(query)
        query=self.to_q(query)
        key=self.to_kv(vision_query)
        value=self.to_kv(vision_query)
        # pdb.set_trace()
        qk=torch.einsum('bcf,df->bcd',query,key)/torch.sqrt(torch.tensor(1024, dtype=torch.float))
        qk=torch.softmax(qk,dim=-1)
        ret=torch.einsum('bcd,df->bcf',qk,value)
        return ret

def get_index_with_padding_batch(a, padding_value=None):
    '''
    Given an attention mask, which only contains 0 and 1, return a tensor that contains the index of non-zero elements. Pad each row of output tensor with given padding_value to the same length.
    Inputs:
        a - (B, M, N)
    Outputs:
        torch.tensor - (B, M, K) , K is the max length of non-zero elements in the N-dim of a.
    Note!!!
        padding_value == N, namely, concat a zero vector at the end of vision query as a candidate padding token.
    '''
    if padding_value is None:
        padding_value = a.shape[-1]
    else:
        assert padding_value == a.shape[-1]

    # Get the indices of non-zero elements, then insert the indices into a new tensor with all padding value.
    non_zero = (a != 0)
    max_length = non_zero.sum(-1).max()
    indices = torch.where(non_zero, torch.arange(a.shape[-1], dtype=torch.long, device=a.device), torch.tensor(padding_value, dtype=torch.long, device=a.device))
    
    # make valid indices at the begining of the tensor, and then split them out.
    padded_indices = indices.topk(k=max_length, dim=-1, largest=False).values
    return padded_indices[:, :, :max_length]

def easy_gather(x, indices):
    # x: B,N,C; indices: B,N
    B, N, C = x.shape
    N_new = indices.shape[1]
    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
    indices = indices + offset
    out = x.flatten(0,1)[indices.view(-1)].view(B, N_new, C)
    return out

class MetaAdapter(nn.Module):
    def __init__(self, dim=1024, num_heads=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.alpha_proj = nn.Linear(dim, 1, bias=True)

        self.up_vision_query = nn.Linear(256,dim)
        self._reset_parameters()
    
    @classmethod
    def _construct_sparse_inputs(cls, x, vision, attention_mask):
        B, V, C = vision.shape # batch, vision, dim
        vision=torch.cat([vision, vision.new_zeros(B, 1, C)], dim=1) # B, N+1, C
        padding_index=V
        index = get_index_with_padding_batch(attention_mask.transpose(2,1), padding_value=padding_index)
        B, T, S = index.shape # batch, text, num_querys
        vision=easy_gather(vision, index.flatten(1,2)).reshape(B, T, S, C)
        x = x[:,:,None,...]
        new_mask=(index[:,:,None,...] != padding_index) # batch, vision, text
        new_mask=new_mask.transpose(-2,-1) # batch, vision, text
        return x, vision, new_mask

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.alpha_proj.weight)
        nn.init.constant_(self.alpha_proj.bias, 1)

    def forward(self, x,vision, attention_mask=None):
        # pdb.set_trace()
        
        x, vision, attention_mask = self._construct_sparse_inputs(x, vision, attention_mask)
        res=x
        q=x
        vision=self.up_vision_query(vision)
        B, K,Q, C = vision.shape
        B, K,T,C = x.shape
        k=self.k_proj(vision)
        v=self.v_proj(vision)
        attention_mask_qk=attention_mask.float()*(-100)
        qk=torch.einsum('bkqc,bktc -> bkqt',k,q)/torch.sqrt(torch.tensor(self.dim, dtype=torch.float))+attention_mask_qk
        b,k,q,t=qk.shape
        qk=qk.reshape(b,k*q*t).softmax(-1)
        qk=qk.reshape(b,k,q,t)
        attn=torch.einsum('bkqt,bkqc -> bktc',qk,v)

        return (attn+res).squeeze(-2)

class MQDET(nn.Module):
    def __init__(self,bank_path='',device='cuda') -> None:
        super().__init__()
        self.device=device
        self.query_bank = torch.load(bank_path, map_location=self.device)
        self.LEARNABLE_BANK=False
        self.num_query_per_class=5
        self.pure_text_rate=0.
    def forward(self, batched_label_list, batched_location_map, batched_pos_labels = None):
        '''
        Return query features, attention mask

        batched_label_list: [[list]] - batch_size, num_labels
        batched_location_map: [torch.tensor] one-hot -  batch_size, (num_labels, num_text_tokens)
        '''
        if self.query_bank is None:
            return None, None, None
        # pdb.set_trace()
        batched_queries = []
        batched_queries_attn_mask = []
        batched_has_vision_query = []
        for k, (label_list, location_map) in enumerate(zip(batched_label_list, batched_location_map)):
            query_per_image = []
            mask_per_image = []
            has_vision_query = []
            for label, loc_map in zip(label_list, location_map):
                loc_map = loc_map.to(self.device)
                if self.LEARNABLE_BANK:
                    candidate_queries=self.query_bank[str(label)]
                else:
                    candidate_queries=self.query_bank[label]
                num_total_queries=len(candidate_queries)
                loc_map = loc_map [None, ...] # 1, num_text_tokens

                # num_query_per_class = self.num_query_per_class
                num_query_per_class = np.random.choice(range(1, self.num_query_per_class+1)) 
                num_queries = min(num_total_queries, num_query_per_class)

                if (random.random() < self.pure_text_rate) and self.training:
                    # data augmentation: random select some labels for only text inputs, without vision query
                    num_queries = 0

                idx= np.random.choice(num_total_queries, num_queries, replace=False).tolist()
                if isinstance(candidate_queries, list):
                    assert len(idx) == 0
                else:
                    queries = candidate_queries[idx]
                    num_scale=queries.shape[1]
                    queries=queries.flatten(0,1)
                    queries_attn_mask = loc_map.expand(num_queries*num_scale, -1)
                    query_per_image.append(queries)
                    mask_per_image.append(queries_attn_mask)

                if batched_pos_labels is None:
                    pos_flag = True
                else:
                    pos_flag = (label in batched_pos_labels[k])

                if pos_flag:
                    has_vision_query.append(1 if num_queries > 0 else 0)

            query_per_image=torch.cat(query_per_image)
            mask_per_image=torch.cat(mask_per_image)

            batched_queries.append(query_per_image)
            batched_queries_attn_mask.append(mask_per_image)
            batched_has_vision_query.append(has_vision_query)

        
        batched_queries=pad_sequence(batched_queries, batch_first=True) # TODO: more efficiet implement
        batched_queries_attn_mask=pad_sequence(batched_queries_attn_mask, batch_first=True)
        
        # The batched_location_map averages the scores, for example, 'apple pie' has two tokenized tokens, thus the location map is (0.5, 0.5) rather than (1, 1). 
        # So we reformulate the batched_queries_attn_mask to 0 or 1.
        batched_queries_attn_mask[batched_queries_attn_mask!=0] = 1

        return batched_queries, batched_queries_attn_mask, batched_has_vision_query

class MapRelationWords(nn.Module):
    def __init__(self,cates_num=52,device='cpu'):
        super().__init__()
        self.relation_weights=nn.Parameter(torch.zeros((cates_num,1))).to(device)
    def forward(self,x):
        ret=x*self.relation_weights
        return ret

class Relation_adapter(nn.Module):
    def __init__(self,ra_map_path):
        super().__init__()
        assert os.path.exists(ra_map_path)==True,"{} does not exist!".format(ra_map_path)
        with open(ra_map_path,'rb') as f:
            self.js=pickle.load(f)
        pass
    def forward(self):
        pass

class Cafo_cache(nn.Module):
    def __init__(self,clip_adapter_shape=(256,256),dino_adapter_shape=(384,256),init_alpha=0.6,init_beta=1,cls_num=13):
        super().__init__()
        self.clip_adapter=nn.Linear(clip_adapter_shape[0],clip_adapter_shape[1])
        self.dino_adapter=nn.Linear(dino_adapter_shape[0],dino_adapter_shape[1])
        self.cache_adapter=nn.Linear(clip_adapter_shape[1],cls_num)
        self.clip_logit_adapter=nn.Linear(clip_adapter_shape[1],cls_num)
        self.dino_logit_adapter=nn.Linear(dino_adapter_shape[1],cls_num)
        self.dinov2_vits14 = torch.hub.load('/home/vis/dingyuning03/baidu/personal-code/dingyuning_APE/APE/dinov2/facebookresearch_dinov2_main', 'dinov2_vits14',source='local').cuda()
        for p in self.dinov2_vits14.parameters():
            p.requires_grad = False
        self.dinov2_vits14.eval()
        self.alpha=init_alpha
        self.beta=init_beta
        self.cls=cls_num
    def logits_fuse(self,zero_logtis, logits, normalize='mean'):
        # normalize logits
        softmax_fun = nn.Softmax(dim=1)
        if normalize == 'softmax':
            zero_logtis = softmax_fun(zero_logtis)
        elif normalize =='linear':
            zero_logtis /= torch.norm(zero_logtis, p=2, dim=1, keepdim=True)
        elif normalize == 'mean':
            logits_std = torch.std(zero_logtis, dim=1, keepdim=True)
            logits_mean = torch.mean(zero_logtis, dim=1, keepdim=True)
            zero_logtis = (zero_logtis - logits_mean) / logits_std
        else:
            raise("error normalize!")
        similarity_matrix = []
        normalize_logits = []
        for logit in logits:
            if normalize == 'softmax':
                current_normalize_logits = softmax_fun(logit)
            elif normalize =='linear':
                current_normalize_logits = logit / torch.norm(logit, p=2, dim=1, keepdim=True)
            elif normalize == 'mean':
                logits_std = torch.std(logit, dim=1, keepdim=True)
                logits_mean = torch.mean(logit, dim=1, keepdim=True)
                current_normalize_logits = (logit - logits_mean) / logits_std
            else:
                raise("error normalize!")
            current_similarity = current_normalize_logits * zero_logtis
            current_similarity = torch.sum(current_similarity, dim=1, keepdim=True)
            similarity_matrix.append(current_similarity)
            normalize_logits.append(current_normalize_logits)
        similarity_matrix = torch.stack(similarity_matrix, dim=-2)
        similarity_matrix = softmax_fun(similarity_matrix)
        normalize_logits = torch.stack(normalize_logits, dim=-2)
        result_logits = torch.sum(normalize_logits * similarity_matrix, dim=1)

        return result_logits
        
    def forward(self,clip_feature,image_tensor):
        clip_feature=torch.mean(clip_feature,dim=-1)
        clip_feature=torch.mean(clip_feature,dim=-1) #b,256
        clip_feature=self.clip_adapter(clip_feature)
        image_tensor=image_tensor[:,:,0:1022,0:1022]
        with torch.no_grad():
            features_dict = self.dinov2_vits14.forward_features(image_tensor)
        dino_feature=features_dict['x_norm_clstoken']
        dino_feature=self.dino_adapter(dino_feature)
        clip_logits_origin=((-1) * (self.beta - self.beta * clip_feature)).exp()
        dino_logits=((-1) * (self.beta - self.beta * dino_feature)).exp()
        clip_logits=self.clip_logit_adapter(clip_logits_origin)
        dino_logits=self.dino_logit_adapter(dino_logits)
        clip_cache_logits=self.cache_adapter(clip_logits_origin)
        cache_logits=self.logits_fuse(clip_cache_logits,[clip_logits, dino_logits])
        
        return cache_logits


class DeformableDETRSegmVL(DeformableDETR):
    """Implements the Deformable DETR model.

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2010.04159>`_ .

    Args:
        backbone (nn.Module): the backbone module.
        position_embedding (nn.Module): the position embedding module.
        neck (nn.Module): the neck module.
        transformer (nn.Module): the transformer module.
        embed_dim (int): the dimension of the embedding.
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): whether to use auxiliary loss. Default: True.
        with_box_refine (bool): whether to use box refinement. Default: False.
        as_two_stage (bool): whether to use two-stage. Default: False.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 100.

    """

    def __init__(
        self,
        instance_on: bool = True,
        semantic_on: bool = False,
        panoptic_on: bool = False,
        freeze_detr=False,
        freeze_backbone=False,
        input_shapes=[],
        mask_in_features=[],
        mask_encode_level=0,
        stuff_dataset_learn_thing: bool = True,
        stuff_prob_thing: float = -1.0,
        name_prompt_fusion_type: str = "none",
        name_prompt_fusion_text: bool = None,
        test_mask_on: bool = True,
        semantic_post_nms: bool = True,
        panoptic_post_nms: bool = True,
        aux_mask: bool = False,
        panoptic_configs: dict = {
            "prob": 0.1,
            "pano_temp": 0.06,
            "transform_eval": True,
            "object_mask_threshold": 0.01,
            "overlap_threshold": 0.4,
        },
        gcp: dict={
            "gcp_on": False,
            "gcp_weight": 0.5,
            "query_path": ''
        },
        cafo: dict={
            "enabled": False,
            "clip_adapter_shape": (256,256),
            "dino_adapter_shape": (384,256),
            "init_alpha": 0.6,     
            "init_beta": 1.0,
            "cls_num": 13
        },
        support_words: dict={
            "enabled": False,
            "words_dict": None,
            "words_num": 5,
            "support_mode": 'adapt',
            "weight": 0.5
        },
        vision_prompts: dict={
            "enabled": False,
            "weight_path": None,
        },
        cls_nums=58,
        mode='train',
        prompts_mode='mini',
        prompts_path=None,
        prompts_update_rate=1.0,
        vision_prompts_on=False,
        vl_mode='adapter',
        **kwargs,
    ):
        super().__init__(**kwargs)
        device='cuda'
        self.instance_on = instance_on
        self.semantic_on = semantic_on
        self.panoptic_on = panoptic_on
        self.gcp=gcp
        self.cafo=cafo
        self.vl_mode=vl_mode
        self.mode=mode
        self.prompts_mode=prompts_mode
        self.prompts_path=prompts_path
        self.prompts_update_rate=prompts_update_rate
        self.vision_prompts_on=vision_prompts_on
        self.update_vision_prompts_flags=1
        if vision_prompts_on:
            self.prompts_backbone, self.preprocess = clip.load("ViT-B/32", device=device)
        
            self.preprocess=self.redefine_preprocess
            for p in self.prompts_backbone.parameters():
                p.requires_grad = False
            self.cls_nums=cls_nums
            self.vision_prompts=nn.Parameter(torch.randn((1,self.cls_nums,512))).to(device)
            # self.learnable_prompts=nn.Parameter(torch.randn((1,self.cls_nums,512)),requires_grad=True).to(device)
            self.prompt_liner=nn.Linear(512,1024)
            for k,v in self.named_parameters():
                print('{}: {}'.format(k, v.requires_grad))

        if self.cafo["enabled"]==True:
            self.cafo_cache=Cafo_cache(
                clip_adapter_shape=self.cafo["clip_adapter_shape"],
                dino_adapter_shape=self.cafo["dino_adapter_shape"],
                init_alpha=self.cafo["init_alpha"],
                init_beta=self.cafo["init_beta"]
            )
        self.support_words=support_words
        if self.support_words['enabled']==True:
            if self.support_words['words_dict']!=None:
                try:
                    with open(self.support_words['words_dict'],'rb') as f:
                        self.text_support_data=pickle.load(f)['text']
                except:
                    self.text_support_data=[]
                self.text_support_adapt=torch.nn.Linear(2048,1024)
        if freeze_detr:
            for p in self.parameters():
                p.requires_grad_(False)
  
        
        self.input_shapes = input_shapes
        self.mask_in_features = mask_in_features
        self.mask_encode_level = mask_encode_level
        if gcp['gcp_on']:
            if gcp['query_path']!='':
                self.cross_attn=CrossAttention(gcp['query_path'])
            else:
                self.cross_attn=CrossAttention()
        if vision_prompts['enabled']:
            with open(vision_prompts['weight_path'],'rb') as f:
                self.vision_prompts=torch.load(f)
        hidden_dim = self.transformer.embed_dim
        norm = "GN"
        use_bias = False

        assert len(self.mask_in_features) == 1
        in_channels = [self.input_shapes[feat_name].channels for feat_name in self.mask_in_features]
        in_channel = in_channels[0]

        self.lateral_conv = Conv2d(
            in_channel,
            hidden_dim,
            kernel_size=1,
            stride=1,
            bias=use_bias,
            padding=0,
            norm=get_norm(norm, hidden_dim),
        )
        self.output_conv = Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            stride=1,
            bias=use_bias,
            padding=1,
            norm=get_norm(norm, hidden_dim),
            activation=F.relu,
        )
        self.mask_conv = Conv2d(
            hidden_dim, hidden_dim, kernel_size=1, stride=1, bias=use_bias, padding=0
        )

        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.aux_mask = aux_mask
        if self.aux_mask:
            self.mask_embed = nn.ModuleList(
                [copy.deepcopy(self.mask_embed) for i in range(len(self.class_embed) - 1)]
            )

        weight_init.c2_xavier_fill(self.lateral_conv)
        weight_init.c2_xavier_fill(self.output_conv)
        weight_init.c2_xavier_fill(self.mask_conv)

        self.stuff_dataset_learn_thing = stuff_dataset_learn_thing
        self.stuff_prob_thing = stuff_prob_thing
        self.test_mask_on = test_mask_on
        self.semantic_post_nms = semantic_post_nms
        self.panoptic_post_nms = panoptic_post_nms
        self.panoptic_configs = panoptic_configs

        self.name_prompt_fusion_type = name_prompt_fusion_type
        self.name_prompt_fusion_text = name_prompt_fusion_text
        if name_prompt_fusion_type == "learnable":
            self.name_prompt_fusion_feature = nn.Parameter(
                torch.Tensor(1, 1, self.embed_dim_language)
            )
            nn.init.normal_(self.name_prompt_fusion_feature)
        elif name_prompt_fusion_type == "zero":
            self.name_prompt_fusion_feature = nn.Parameter(
                torch.zeros(1, 1, self.embed_dim_language), requires_grad=False
            )
        else:
            self.name_prompt_fusion_feature = None
        
    
    def freeze(self):
        pass

    def redefine_preprocess(self,input_tensor):

        input_tensor = input_tensor.permute(2, 0, 1) # Change image from HWC to CHW format
        preprocess = transforms.Compose([
            transforms.ToPILImage('RGB'), # Convert tensor to PIL Image
            transforms.Resize(224, interpolation=Image.BICUBIC), # Resize image
            transforms.CenterCrop((224, 224)), # Center crop image
            transforms.ToTensor(), # Convert image to tensor
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                             std=(0.26862954, 0.26130258, 0.27577711)) # Normalize image
        ])
        return preprocess(input_tensor)
    
    def generate_with_probability(self):
        import random
        random_num=random.random()
        if random_num<=self.prompts_update_rate:
            return True
        else:
            return False
    def update_vision_prompts(self,batched_inputs,device='cuda'):
        if self.mode=='infer' and self.prompts_mode=='mini' and self.prompts_path!=None:
            prompt_names = os.listdir(self.prompts_path)
            pattern = re.compile(r"tensor\((\d+)\)\.png")
            for name in prompt_names:
                category = int(pattern.search(name).group(1))
                image = Image.open(os.path.join(self.prompts_path, name)).convert("RGB")
                image = np.array(image)[:, :, ::-1].copy()              
                data=torch.from_numpy(image)
                # pdb.set_trace()
                self.prompts_backbone.eval()    
                roi=self.preprocess(data).to(device).unsqueeze(0)
                roi_features = self.prompts_backbone.encode_image(roi)
                self.vision_prompts[0,category]=roi_features[0]
            return
        if self.mode=='infer':
            return
        # pdb.set_trace()
        if 'instances' not in batched_inputs[0].keys():
            return
        instance=batched_inputs[0]['instances']
        
        for i in range(0,len(instance)):
            try:
                if self.generate_with_probability()==False:
                    continue
                bbox=instance[i].gt_boxes.tensor
                category=instance[i].gt_classes[0]
                img=batched_inputs[0]['image']
                # pdb.set_trace()
                x1,y1,x2,y2=int(bbox[0,0]),int(bbox[0,1]),int(bbox[0,2]),int(bbox[0,3])
                data=img[:,y1:y2,x1:x2]
                data=data.permute(1,2,0)
                cv2.imwrite('output_imgs/{}.png'.format(str(category)),data.cpu().numpy())
                # pdb.set_trace()
                self.prompts_backbone.eval()
                roi=self.preprocess(data).to(device).unsqueeze(0)
                roi_features = self.prompts_backbone.encode_image(roi)
                self.vision_prompts[0,category]=roi_features[0]
            except:
                continue


    def forward(self, batched_inputs, do_postprocess=True,support_dict=None):
        # pdb.set_trace()
        if support_dict!=None:
        # pdb.set_trace()
            self.origin_text_support_data=self.text_support_data.copy()
            self.origin_vision_query=self.cross_attn.vision_query.detach()
            if support_dict!=None:
                if self.support_words['enabled']==True:
                    self.text_support_data=support_dict['text']
                if self.gcp['gcp_on']==True:
                    # self.cross_attn.
                    query=support_dict['image']
                    arr=[]
                    for i in range(1,len(query)+1):
                        arr.append(torch.mean(query[i],dim=0).cpu())
                    self.cross_attn.vision_query=torch.cat(arr,dim=0).to('cuda').float()
            else:
                self.cross_attn.vision_query=self.origin_vision_query
                self.text_support_data=self.origin_text_support_data

        if self.training:
            if "dataset_id" in batched_inputs[0]:
                dataset_ids = [x["dataset_id"] for x in batched_inputs]
                assert len(set(dataset_ids)) == 1, dataset_ids
                dataset_id = dataset_ids[0]
            else:
                dataset_id = 0
        else:
            dataset_id = self.eval_dataset_id
        

        if self.vision_prompts_on:
            if self.mode == 'infer' and self.update_vision_prompts_flags:
                self.update_vision_prompts(batched_inputs)
                self.update_vision_prompts_flags=0
                print("visual pormpt loaded")
            else:
                self.update_vision_prompts(batched_inputs)
               
            vision_prompts=self.vision_prompts
            vision_prompts=self.prompt_liner(vision_prompts) #1,cls,1024
            features_v=vision_prompts
            features_v_fusion=torch.mean(features_v,dim=1,keepdim=True)
        # pdb.set_trace()
        if dataset_id >= 0:
            prompt = self.dataset_prompts[dataset_id]
        elif "prompt" in batched_inputs[0]:
            prompt = batched_inputs[0]["prompt"]
        else:
            prompt = "name"

        if prompt == "expression":
            for x in batched_inputs:
                if isinstance(x["expressions"], List):
                    pass
                else:
                    x["expressions"] = [x["expressions"]]
                assert all([len(xx) > 0 for xx in x["expressions"]])
                assert all([isinstance(xx, str) for xx in x["expressions"]])
                self.test_topk_per_image = 1
        else:
            self.test_topk_per_image = self.select_box_nums_for_evaluation
        if self.select_box_nums_for_evaluation_list is not None:
            self.test_topk_per_image = self.select_box_nums_for_evaluation_list[dataset_id]

        if self.training and prompt == "phrase":
            gt_num = torch.tensor([len(input["instances"]) for input in batched_inputs]).to(
                self.device
            )
            gt_classes = torch.arange(gt_num.sum()).to(self.device)
            gt_cumsum = torch.cumsum(gt_num, dim=0).to(self.device)
            for i, input in enumerate(batched_inputs):
                if i == 0:
                    input["instances"].gt_classes = gt_classes[: gt_cumsum[i]]
                else:
                    input["instances"].gt_classes = gt_classes[gt_cumsum[i - 1] : gt_cumsum[i]]
        if self.training and prompt == "expression":
            gt_num = torch.tensor([len(input["instances"]) for input in batched_inputs]).to(
                self.device
            )
            gt_classes = torch.arange(gt_num.sum()).to(self.device)
            gt_cumsum = torch.cumsum(gt_num, dim=0).to(self.device)
            for i, input in enumerate(batched_inputs):
                if i == 0:
                    input["instances"].gt_classes = gt_classes[: gt_cumsum[i]]
                else:
                    input["instances"].gt_classes = gt_classes[gt_cumsum[i - 1] : gt_cumsum[i]]

                if not self.expression_cumulative_gt_class:
                    input["instances"].gt_classes *= 0

        if prompt == "text":
            
            texts = [x["text_prompt"] for x in batched_inputs]
            text_promp_text_list = [x.strip() for x in ",".join(texts).split(",")]
            text_promp_text_list = [x for x in text_promp_text_list if len(x) > 0]

            if any([True if x.count(" ") >= 1 else False for x in text_promp_text_list]):
                prompt = "phrase"
            else:
                prompt = "name"
        else:
            text_promp_text_list = None
        # pdb.set_trace()
        mode='train'
        if prompt == "name":
            if text_promp_text_list:
                text_list = text_promp_text_list
                cache = False
            elif dataset_id >= 0:
                text_list = get_text_list(
                    self.metadata_list[dataset_id], self.dataset_entities[dataset_id]
                )
                cache = True
            elif mode=='infer':
                texts = [x["text_prompt"] for x in batched_inputs]
                text_promp_text_list = [x.strip() for x in ",".join(texts).split(",")]
                text_promp_text_list = [x for x in text_promp_text_list if len(x) > 0]
                text_list=text_promp_text_list
                cache = True
                # pdb.set_trace()
            else:
                text_list = []
                for metadata, dataset_entity in zip(self.metadata_list, self.dataset_entities):
                    text_list += get_text_list(metadata, dataset_entity)
                text_list = text_list[:1203+365+601]
                text_list = text_list[:1203]
                cache = True
                
                # from detectron2.data.catalog import MetadataCatalog
                # metadata = MetadataCatalog.get("coco_2017_train_panoptic_separated")
                # text_list = get_text_list(metadata, "thing+stuff")
            
            if self.support_words['support_mode']=='concat':
                for i in range(0,len(text_list)):
                    text_list[i]=text_list[i]+' with '+self.text_support_data[i]

            # pdb.set_trace()
            outputs_l = self.model_language.forward_text(text_list, cache=cache)

            if "last_hidden_state_eot" in outputs_l:
                features_l = outputs_l["last_hidden_state_eot"]
                
            else:
                features_l = text_utils.reduce_language_feature(
                    outputs_l["last_hidden_state"],
                    outputs_l["attention_mask"],
                    reduce_type=self.text_feature_reduce_type,
                )
            if self.support_words['enabled']==True and self.support_words['support_mode']=='adapt' and len(self.text_support_data)==len(text_list):
                    outputs_support_l = self.model_language.forward_text(self.text_support_data, cache=cache)
                    output_l_combine=torch.cat([features_l,outputs_support_l['last_hidden_state_eot']],dim=-1)
                    features_l=features_l+self.support_words['weight']*self.text_support_adapt(output_l_combine)

            attention_mask_l = None

            if (
                dataset_id >= 0
                and self.dataset_entities[dataset_id] == "stuff"
                and self.metadata_list[dataset_id].get("stuff_classes")[0] == "things"
                and not self.stuff_dataset_learn_thing
            ):
                features_l[0, :] *= 0
                if self.training:
                    for i, input in enumerate(batched_inputs):
                        input["instances"] = input["instances"][input["instances"].gt_classes > 0]

            if self.text_feature_batch_repeat or True:
                features_l = features_l.unsqueeze(0).repeat(len(batched_inputs), 1, 1)
            else:
                features_l = features_l.unsqueeze(1)

        elif prompt == "phrase" or prompt == "expression":
            # pdb.set_trace()
            if text_promp_text_list:
                text_list = text_promp_text_list
            elif prompt == "phrase":
                text_list = [phrase for x in batched_inputs for phrase in x["instances"].phrases]
            elif prompt == "expression":
                text_list = [xx for x in batched_inputs for xx in x["expressions"]]

            outputs_l = self.model_language.forward_text(text_list)

            if self.text_feature_reduce_before_fusion:
                if "last_hidden_state_eot" in outputs_l:
                    features_l = outputs_l["last_hidden_state_eot"]
                else:
                    features_l = text_utils.reduce_language_feature(
                        outputs_l["last_hidden_state"],
                        outputs_l["attention_mask"],
                        reduce_type=self.text_feature_reduce_type,
                    )
                attention_mask_l = None

                if (
                    self.text_feature_bank
                    and not self.text_feature_bank_reset
                    and dataset_id >= 0
                    and dataset_id < len(self.metadata_list)
                ):
                    features_l = torch.cat(
                        [features_l, self.features_phrase_bank[dataset_id]], dim=0
                    )
                    features_l = features_l[
                        : max(len(text_list), self.criterion[dataset_id].num_classes)
                    ]
                    self.features_phrase_bank[
                        dataset_id, : self.criterion[dataset_id].num_classes
                    ] = features_l[: self.criterion[dataset_id].num_classes]
                elif self.text_feature_bank and self.text_feature_bank_reset:
                    features_l = torch.cat(
                        [features_l, self.features_phrase_bank[dataset_id] * 0], dim=0
                    )
                    features_l = features_l[
                        : max(len(text_list), self.criterion[dataset_id].num_classes)
                    ]

                if self.text_feature_batch_repeat:
                    features_l = features_l.unsqueeze(0).repeat(len(batched_inputs), 1, 1)
                else:
                    features_l = features_l.unsqueeze(1)
            else:
                features_l = outputs_l["last_hidden_state"]
                attention_mask_l = outputs_l["attention_mask"]
        start_time = time.perf_counter()
        
        self.backbone_time = time.perf_counter() - start_time
        images = self.preprocess_image(batched_inputs)

        batch_size, _, H, W = images.tensor.shape
        img_masks = images.tensor.new_ones(batch_size, H, W)
        for image_id, image_size in enumerate(images.image_sizes):
            img_masks[image_id, : image_size[0], : image_size[1]] = 0
        self.preprocess_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        # pdb.set_trace()
        features = self.backbone(images.tensor)  # output feature dict

        if self.gcp['gcp_on']:
            features_l=features_l+self.cross_attn(features_l)*self.gcp['gcp_weight']
        
        if prompt == "name":
            if (
                self.name_prompt_fusion_text is not None
                and self.name_prompt_fusion_text[dataset_id]
            ):
                features_l_fusion = features_l
            else:
                if self.name_prompt_fusion_feature is not None:
                    features_l_fusion = self.name_prompt_fusion_feature.repeat(
                        len(batched_inputs), 1, 1
                    )
                else:
                    features_l_fusion = None
            attention_mask_l_fusion = None
        elif prompt == "phrase" or prompt == "expression":
            features_l_fusion = features_l
            attention_mask_l_fusion = attention_mask_l
            if self.name_prompt_fusion_feature is not None:
                features_l_fusion += 0.0 * self.name_prompt_fusion_feature

        

        if self.neck is not None:
            multi_level_feats = self.neck({f: features[f] for f in self.neck.in_features})
        else:
            multi_level_feats = [feat for feat_name, feat in features.items()]
        multi_level_masks = []
        multi_level_position_embeddings = []
        spatial_shapes = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_position_embeddings.append(
                self.position_embedding(multi_level_masks[-1]).to(images.tensor.dtype)
            )

            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
        # pdb.set_trace() 
        if not self.training and "mask_prompt" in batched_inputs[0]:
            masks_prompt = [self._move_to_current_device(x["mask_prompt"]) for x in batched_inputs]
            masks_prompt = [x.to(self.pixel_mean.dtype) for x in masks_prompt]
            masks_prompt = ImageList.from_tensors(
                masks_prompt,
                self.backbone.size_divisibility,
                padding_constraints=self.backbone.padding_constraints,
            )
            masks_prompt = masks_prompt.tensor
            if masks_prompt.sum() == 0:
                masks_prompt[...] = 255

            multi_level_masks_prompt = []
            for feat in multi_level_feats:
                multi_level_masks_prompt.append(
                    F.interpolate(masks_prompt[None], size=feat.shape[-2:], mode="bilinear")
                    .to(torch.bool)
                    .squeeze(0)
                )
            for mask_prompt in multi_level_masks_prompt:
                print("mask_prompt", mask_prompt.sum(), mask_prompt.size())
        else:
            multi_level_masks_prompt = None

        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight

        start_time = time.perf_counter()
        # pdb.set_trace()
        (
            inter_states,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
            anchors,
            memory,
            features_l_fusion,
        ) = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            query_embeds,
            features_l_fusion,
            attention_mask_l_fusion,
            multi_level_masks_prompt,
        )

        if self.vision_prompts_on:
            (
                inter_states2,
                init_reference2,
                inter_references2,
                enc_outputs_class2,
                enc_outputs_coord_unact2,
                anchors2,
                memory2,
                features_v_fusion,
            ) = self.transformer(
                multi_level_feats,
                multi_level_masks,
                multi_level_position_embeddings,
                query_embeds,
                features_v_fusion,
                None,
                None,
            )



        self.transformer_time = time.perf_counter() - start_time

        

        mask_features = self.maskdino_mask_features(memory, features, multi_level_masks)
       
        # pdb.set_trace()
        if prompt == "name":
            features_l = 1.0 * features_l + 0.0 * features_l_fusion
        elif prompt == "phrase" or prompt == "expression":
            features_l = 0.0 * features_l + 1.0 * features_l_fusion

            if not self.text_feature_reduce_before_fusion:
                features_l = text_utils.reduce_language_feature(
                    features_l, attention_mask_l, reduce_type=self.text_feature_reduce_type
                )
                attention_mask_l = None

                if self.text_feature_bank:
                    features_l = torch.cat(
                        [features_l, self.features_phrase_bank[dataset_id]], dim=0
                    )
                    features_l = features_l[: self.criterion[dataset_id].num_classes]
                    self.features_phrase_bank[
                        dataset_id, : self.criterion[dataset_id].num_classes
                    ] = features_l
                elif self.text_feature_bank and not self.training:
                    features_l = torch.cat(
                        (
                            features_l,
                            torch.zeros(
                                (self.criterion[dataset_id].num_classes - 1, features_l.size(1)),
                                dtype=features_l.dtype,
                                device=self.device,
                            ),
                        ),
                        dim=0,
                    )

                if self.text_feature_batch_repeat:
                    features_l = features_l.unsqueeze(0).repeat(len(batched_inputs), 1, 1)
                else:
                    features_l = features_l.unsqueeze(1)
        if self.vision_prompts_on and self.vl_mode=='adapter':
            features_l=features_l+features_v
        elif self.vision_prompts_on and self.vl_mode=='ensemble':
            # pdb.set_trace()
            outputs_classes_v = []
            outputs_coords_v = []
            outputs_masks_v = []
            for lvl in range(inter_states2.shape[0]):
                # pdb.set_trace()
                if lvl == 0:
                    reference = init_reference2
                else:
                    reference = inter_references2[lvl - 1]
                reference = inverse_sigmoid(reference)
                if prompt == "name":
                    outputs_class_v = self.class_embed[lvl](inter_states2[lvl], features_v)
                elif prompt == "phrase" or prompt == "expression":
                    outputs_class_v = self.class_embed[lvl](inter_states2[lvl], features_v)
                else:
                    outputs_class_v = self.class_embed[lvl](inter_states2[lvl])
                b,bbox,c=outputs_class_v.shape
                if self.cafo["enabled"]==True:
                    cache_det_logits=cache_logits.unsqueeze(1).repeat(1,bbox,1)
                    outputs_class_v=outputs_class_v+cache_det_logits
                tmp = self.bbox_embed[lvl](inter_states2[lvl])
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference
                outputs_coord_v = tmp.sigmoid()
                outputs_classes_v.append(outputs_class_v)
                outputs_coords_v.append(outputs_coord_v)

                if self.aux_mask:
                    mask_embeds = self.mask_embed[lvl](inter_states2[lvl])
                else:
                    mask_embeds = self.mask_embed(inter_states2[lvl])
                outputs_mask_v = torch.einsum("bqc,bchw->bqhw", mask_embeds, mask_features)
                outputs_masks_v.append(outputs_mask_v)
            
            outputs_class_v = torch.stack(outputs_classes_v)
            outputs_coord_v = torch.stack(outputs_coords_v)
            # pdb.set_trace()
            outputs_mask_v = outputs_masks_v
            outputs_mask_v[-1] += 0.0 * sum(outputs_mask_v)

        outputs_classes = []
        outputs_coords = []
        outputs_masks = []
        
        if self.cafo["enabled"]==True:
            cache_logits=self.cafo_cache(features['p6'],images.tensor)
        for lvl in range(inter_states.shape[0]):
            # pdb.set_trace()
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            if prompt == "name":
                outputs_class = self.class_embed[lvl](inter_states[lvl], features_l)
            elif prompt == "phrase" or prompt == "expression":
                outputs_class = self.class_embed[lvl](inter_states[lvl], features_l)
            else:
                outputs_class = self.class_embed[lvl](inter_states[lvl])
            b,bbox,c=outputs_class.shape
            if self.cafo["enabled"]==True:
                cache_det_logits=cache_logits.unsqueeze(1).repeat(1,bbox,1)
                outputs_class=outputs_class+cache_det_logits
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

            if self.aux_mask:
                mask_embeds = self.mask_embed[lvl](inter_states[lvl])
            else:
                mask_embeds = self.mask_embed(inter_states[lvl])
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embeds, mask_features)
            outputs_masks.append(outputs_mask)
        
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        # pdb.set_trace()
        outputs_mask = outputs_masks
        outputs_mask[-1] += 0.0 * sum(outputs_mask)

        if self.vision_prompts_on and self.vl_mode=='ensemble':
            # pdb.set_trace()
            outputs_class=torch.cat([outputs_class,outputs_class_v],dim=2)
            outputs_coord=torch.cat([outputs_coord,outputs_coord_v],dim=2)
            for m_idx in range(0,len(outputs_mask)):
                outputs_mask[m_idx]=torch.cat([outputs_mask[m_idx],outputs_masks_v[m_idx]],dim=1)
        
        output = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_masks": outputs_mask[-1],
            "init_reference": init_reference,
        }
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(
                outputs_class,
                outputs_coord,
                outputs_mask,
            )

        if self.as_two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            output["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
                "anchors": anchors,
                "spatial_shapes": spatial_shapes,
                "image_tensor_size": images.tensor.size()[2:],
            }

        if (
            self.vis_period > 0
            and self.training
            and get_event_storage().iter % self.vis_period == self.vis_period - 1
        ):
            self.visualize_training(batched_inputs, output, images, dataset_id)
            self.visualize_training_enc_output(batched_inputs, output, images, dataset_id)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)

            loss_dict = self.criterion[dataset_id](output, targets)

            weight_dict = self.criterion[dataset_id].weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:

            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            mask_pred = output["pred_masks"]

            start_time = time.perf_counter()

            iter_func = retry_if_cuda_oom(F.interpolate)
            mask_pred = iter_func(
                mask_pred, size=images.tensor.size()[2:], mode="bilinear", align_corners=False
            )

            merged_results = [{} for _ in range(box_cls.size(0))]
            if self.instance_on and not (
                self.eval_dataset_entity and "thing" not in self.eval_dataset_entity
            ):
                if dataset_id >= 0 and dataset_id < len(self.metadata_list):
                    if is_thing_stuff_overlap(self.metadata_list[dataset_id]):
                        thing_id = self.metadata_list[
                            dataset_id
                        ].thing_dataset_id_to_contiguous_id.values()
                        thing_id = torch.Tensor(list(thing_id)).to(torch.long).to(self.device)

                        detector_box_cls = torch.zeros_like(box_cls)
                        detector_box_cls += float("-inf")
                        detector_box_cls[..., thing_id] = box_cls[..., thing_id]
                    else:
                        num_thing_classes = len(self.metadata_list[dataset_id].thing_classes)
                        detector_box_cls = box_cls[..., :num_thing_classes]
                else:
                    detector_box_cls = box_cls

                use_sigmoid = True
                detector_results, filter_inds = self.inference(
                    detector_box_cls, box_pred, images.image_sizes, use_sigmoid=use_sigmoid
                )

                if self.test_mask_on:
                    detector_mask_preds = [
                        x[filter_ind] for x, filter_ind in zip(mask_pred, filter_inds)
                    ]

                    for result, box_mask in zip(detector_results, detector_mask_preds):
                        box_mask = box_mask.sigmoid() > 0.5
                        box_mask = BitMasks(box_mask).crop_and_resize(
                            result.pred_boxes.tensor.to(box_mask.device), 128
                        )
                        result.pred_masks = (
                            box_mask.to(result.pred_boxes.tensor.device)
                            .unsqueeze(1)
                            .to(dtype=torch.float32)
                        )

                if do_postprocess:
                    assert (
                        not torch.jit.is_scripting()
                    ), "Scripting is not supported for postprocess."
                    detector_results = DeformableDETRSegmVL._postprocess_instance(
                        detector_results, batched_inputs, images.image_sizes
                    )
                    for merged_result, detector_result in zip(merged_results, detector_results):
                        merged_result.update(detector_result)

            else:
                detector_results = None

            if self.semantic_on and not (
                self.eval_dataset_entity and "stuff" not in self.eval_dataset_entity
            ):

                semantic_mask_pred = mask_pred.clone()
                semantic_box_cls = get_stuff_score(
                    box_cls, self.metadata_list[dataset_id], self.dataset_entities[dataset_id]
                )

                if self.semantic_post_nms:
                    _, filter_inds = self.inference(semantic_box_cls, box_pred, images.image_sizes)
                    semantic_box_cls = torch.stack(
                        [x[filter_ind] for x, filter_ind in zip(semantic_box_cls, filter_inds)],
                        dim=0,
                    )
                    semantic_mask_pred = torch.stack(
                        [x[filter_ind] for x, filter_ind in zip(semantic_mask_pred, filter_inds)],
                        dim=0,
                    )

                if do_postprocess:
                    assert (
                        not torch.jit.is_scripting()
                    ), "Scripting is not supported for postprocess."
                    semantic_results = DeformableDETRSegmVL._postprocess_semantic(
                        semantic_box_cls, semantic_mask_pred, batched_inputs, images
                    )
                    if (
                        dataset_id >= 0
                        and self.dataset_entities[dataset_id] == "stuff"
                        and self.metadata_list[dataset_id].get("stuff_classes")[0] == "things"
                        and self.stuff_prob_thing > 0
                    ):
                        for semantic_result in semantic_results:
                            semantic_result["sem_seg"][0, ...] = math.log(
                                self.stuff_prob_thing / (1 - self.stuff_prob_thing)
                            )
                    for merged_result, semantic_result in zip(merged_results, semantic_results):
                        merged_result.update(semantic_result)

            else:
                semantic_results = None

            if self.panoptic_on and not (
                self.eval_dataset_entity and "thing+stuff" not in self.eval_dataset_entity
            ):
                assert dataset_id >= 0 and dataset_id < len(self.metadata_list)
                if do_postprocess:
                    assert (
                        not torch.jit.is_scripting()
                    ), "Scripting is not supported for postprocess."
                    if True:
                        if self.panoptic_post_nms:
                            _, filter_inds = self.inference(box_cls, box_pred, images.image_sizes)
                            panoptic_mask_pred = [
                                x[filter_ind] for x, filter_ind in zip(mask_pred, filter_inds)
                            ]
                            panoptic_box_cls = [
                                x[filter_ind] for x, filter_ind in zip(box_cls, filter_inds)
                            ]

                        panoptic_results = DeformableDETRSegmVL._postprocess_panoptic(
                            panoptic_box_cls,
                            panoptic_mask_pred,
                            batched_inputs,
                            images,
                            self.metadata_list[dataset_id],
                            self.panoptic_configs,
                        )
                    else:
                        panoptic_results = []
                        self.combine_overlap_thresh = 0.5
                        self.combine_stuff_area_thresh = 4096
                        self.combine_instances_score_thresh = 0.5
                        for detector_result, semantic_result in zip(
                            detector_results, semantic_results
                        ):
                            detector_r = detector_result["instances"]
                            sem_seg_r = semantic_result["sem_seg"]
                            panoptic_r = combine_semantic_and_instance_outputs(
                                detector_r,
                                sem_seg_r.argmax(dim=0),
                                self.combine_overlap_thresh,
                                self.combine_stuff_area_thresh,
                                self.combine_instances_score_thresh,
                            )
                            panoptic_results.append({"panoptic_seg": panoptic_r})
                    for merged_result, panoptic_result in zip(merged_results, panoptic_results):
                        merged_result.update(panoptic_result)

            else:
                panoptic_results = None

            self.postprocess_time = time.perf_counter() - start_time

            if do_postprocess:
                return merged_results

            return detector_results, semantic_results, panoptic_results

    def maskdino_mask_features(self, encode_feats, multi_level_feats, multi_level_masks):
        start_idx = sum(
            [mask.shape[1] * mask.shape[2] for mask in multi_level_masks[: self.mask_encode_level]]
        )
        end_idx = sum(
            [
                mask.shape[1] * mask.shape[2]
                for mask in multi_level_masks[: self.mask_encode_level + 1]
            ]
        )
        b, h, w = multi_level_masks[self.mask_encode_level].size()

        encode_feats = encode_feats[:, start_idx:end_idx, :]
        encode_feats = encode_feats.permute(0, 2, 1).reshape(b, -1, h, w)

        x = [multi_level_feats[f] for f in self.mask_in_features]
        x = x[0]
        x = self.lateral_conv(x)
        x = x + F.interpolate(encode_feats, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.output_conv(x)
        mask_features = self.mask_conv(x)

        return mask_features

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_mask):
        return [
            {"pred_logits": a, "pred_boxes": b, "pred_masks": c}
            for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_mask[:-1])
        ]

    def inference(self, box_cls, box_pred, image_sizes, use_sigmoid=True):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """

        if use_sigmoid:
            scores = torch.cat(
                (
                    box_cls.sigmoid(),
                    torch.zeros((box_cls.size(0), box_cls.size(1), 1), device=self.device),
                ),
                dim=2,
            )
        else:
            scores = torch.cat(
                (
                    box_cls,
                    torch.zeros((box_cls.size(0), box_cls.size(1), 1), device=self.device),
                ),
                dim=2,
            )

        boxes = box_cxcywh_to_xyxy(box_pred)

        img_h = torch.tensor([image_size[0] for image_size in image_sizes], device=self.device)
        img_w = torch.tensor([image_size[1] for image_size in image_sizes], device=self.device)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        boxes = boxes.unbind(0)
        scores = scores.unbind(0)
        image_shapes = image_sizes

        results, filter_inds = fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

        return results, filter_inds

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)

            if not targets_per_image.has("gt_masks"):
                gt_masks = torch.zeros((0, h, w), dtype=torch.bool)
            else:
                gt_masks = targets_per_image.gt_masks

            if not isinstance(gt_masks, torch.Tensor):
                if isinstance(gt_masks, BitMasks):
                    gt_masks = gt_masks.tensor
                else:
                    gt_masks = BitMasks.from_polygon_masks(gt_masks, h, w).tensor

            gt_masks = self._move_to_current_device(gt_masks)
            gt_masks = ImageList.from_tensors(
                [gt_masks],
                self.backbone.size_divisibility,
                padding_constraints=self.backbone.padding_constraints,
            ).tensor.squeeze(0)

            new_targets.append({"labels": gt_classes, "boxes": gt_boxes, "masks": gt_masks})

            if targets_per_image.has("is_thing"):
                new_targets[-1]["is_thing"] = targets_per_image.is_thing

        return new_targets

    def preprocess_image(self, batched_inputs):
        # pdb.set_trace()
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [x.to(self.pixel_mean.dtype) for x in images]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    @staticmethod
    def _postprocess_instance(
        instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes
    ):
        """
        Rescale the output instances to the target size.
        """
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r.to("cpu")})
        return processed_results

    @staticmethod
    def _postprocess_semantic(
        mask_clses,
        mask_preds,
        batched_inputs: List[Dict[str, torch.Tensor]],
        images,
        pano_temp=0.06,
        transform_eval=True,
    ):
        processed_results = []
        for mask_cls, mask_pred, input_per_image, image_size in zip(
            mask_clses, mask_preds, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])

            T = pano_temp
            mask_cls = mask_cls.sigmoid()

            if transform_eval:
                mask_cls = F.softmax(mask_cls / T, dim=-1)  # already sigmoid
            mask_pred = mask_pred.sigmoid()
            if mask_cls.size(1) > 1000:
                mask_cls = mask_cls.cpu()
                mask_pred = mask_pred.cpu()
            result = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)

            if True and False:
                num_thing_classes = len(
                    metadata.get(
                        "thing_classes",
                        [
                            "things",
                        ],
                    )
                )

                result_0 = result[:num_thing_classes, ...]
                result_1 = result[num_thing_classes:, ...]
                result_0 = result_0.mean(dim=0, keepdim=True)
                result = torch.cat([result_0, result_1], dim=0)

            r = sem_seg_postprocess(result, image_size, height, width)
            processed_results.append({"sem_seg": r})
        return processed_results

    @staticmethod
    def _postprocess_panoptic(
        mask_clses,
        mask_preds,
        batched_inputs: List[Dict[str, torch.Tensor]],
        images,
        metadata,
        panoptic_configs,
    ):
        prob = panoptic_configs["prob"]
        pano_temp = panoptic_configs["pano_temp"]
        transform_eval = panoptic_configs["transform_eval"]
        object_mask_threshold = panoptic_configs["object_mask_threshold"]
        overlap_threshold = panoptic_configs["overlap_threshold"]

        processed_results = []
        for mask_cls, mask_pred, input_per_image, image_size in zip(
            mask_clses, mask_preds, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])

            mask_pred = sem_seg_postprocess(mask_pred, image_size, height, width)

            T = pano_temp
            scores, labels = mask_cls.sigmoid().max(-1)
            mask_pred = mask_pred.sigmoid()
            keep = scores > object_mask_threshold
            if transform_eval:
                scores, labels = F.softmax(mask_cls.sigmoid() / T, dim=-1).max(-1)
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[keep]
            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

            panoptic_seg = torch.zeros((height, width), dtype=torch.int32, device=cur_masks.device)
            segments_info = []

            current_segment_id = 0

            if cur_masks.size(0) > 0:

                cur_mask_ids = cur_prob_masks.argmax(0)

            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= prob).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= prob)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < overlap_threshold:
                        continue

                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    if not isthing and metadata.get("stuff_classes")[0] == "things":
                        pred_class = int(pred_class) - len(metadata.thing_classes) + 1

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            processed_results.append({"panoptic_seg": (panoptic_seg, segments_info)})
        return processed_results

    @torch.no_grad()
    def visualize_training(
        self, batched_inputs, output, images, dataset_id, suffix="", do_nms=True
    ):
        if self.output_dir is None:
            return
        if self.training:
            storage = get_event_storage()
            os.makedirs(self.output_dir + "/training", exist_ok=True)
        else:
            os.makedirs(self.output_dir + "/inference", exist_ok=True)

        pred_logits = output["pred_logits"]
        pred_boxes = output["pred_boxes"]
        pred_masks = output["pred_masks"]

        thing_classes = self.metadata_list[dataset_id].get("thing_classes", [])
        stuff_classes = self.metadata_list[dataset_id].get("stuff_classes", [])
        if len(thing_classes) > 0 and len(stuff_classes) > 0 and stuff_classes[0] == "things":
            stuff_classes = stuff_classes[1:]
        if is_thing_stuff_overlap(self.metadata_list[dataset_id]):
            class_names = (
                thing_classes if len(thing_classes) > len(stuff_classes) else stuff_classes
            )
        else:
            class_names = thing_classes + stuff_classes

        if "instances" in batched_inputs[0] and batched_inputs[0]["instances"].has("phrases"):
            class_names = [phrase for x in batched_inputs for phrase in x["instances"].phrases] + [
                "unknown"
            ] * 1000
        if "expressions" in batched_inputs[0] and self.expression_cumulative_gt_class:
            class_names = [xx for x in batched_inputs for xx in x["expressions"]] + [
                "unknown"
            ] * 1000

        num_thing_classes = len(class_names)
        pred_logits = pred_logits[..., :num_thing_classes]

        if pred_masks is not None:
            pred_masks = [
                F.interpolate(
                    pred_mask.float().cpu().unsqueeze(0),
                    size=images.tensor.size()[2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                if pred_mask.size(0) > 0
                else pred_mask
                for pred_mask in pred_masks
            ]
        else:
            pred_masks = [
                torch.zeros(pred_box.size(0), image_size[0], image_size[1])
                for pred_box, image_size in zip(pred_boxes, images.image_sizes)
            ]

        if do_nms:
            results, filter_inds = self.inference(pred_logits, pred_boxes, images.image_sizes)
            pred_masks = [
                pred_mask[filter_ind.cpu()]
                for pred_mask, filter_ind in zip(pred_masks, filter_inds)
            ]
            for result, pred_mask in zip(results, pred_masks):
                result.pred_masks = pred_mask.sigmoid() > 0.5
        else:
            results = []
            for pred_logit, pred_box, pred_mask, image_size in zip(
                pred_logits, pred_boxes, pred_masks, images.image_sizes
            ):
                result = Instances(image_size)
                result.pred_boxes = Boxes(pred_box)
                result.scores = pred_logit[:, 0]
                result.pred_classes = torch.zeros(
                    len(pred_box), dtype=torch.int64, device=pred_logit.device
                )
                result.pred_masks = pred_mask.sigmoid() > 0.5

                results.append(result)

        from detectron2.utils.visualizer import Visualizer

        for input, result in zip(batched_inputs, results):

            if "expressions" in batched_inputs[0] and not self.expression_cumulative_gt_class:
                class_names = [xx for xx in input["expressions"]] + ["unknown"] * 1000

            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)

            if "instances" in input:
                labels = [
                    "{}".format(class_names[gt_class]) for gt_class in input["instances"].gt_classes
                ]
                v_gt = v_gt.overlay_instances(
                    boxes=input["instances"].gt_boxes,
                    masks=input["instances"].gt_masks
                    if input["instances"].has("gt_masks")
                    else None,
                    labels=labels,
                )
            else:
                v_gt = v_gt.output
            anno_img = v_gt.get_image()

            labels = [
                "{}_{:.0f}%".format(class_names[pred_class], score * 100)
                for pred_class, score in zip(result.pred_classes.cpu(), result.scores.cpu())
            ]
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=result.pred_boxes.tensor.clone().detach().cpu().numpy(),
                labels=labels,
                masks=result.pred_masks[:, : img.shape[0], : img.shape[1]]
                .clone()
                .detach()
                .cpu()
                .numpy()
                if result.has("pred_masks")
                else None,
            )
            pred_img = v_pred.get_image()

            vis_img = np.concatenate((anno_img, pred_img), axis=1)

            if result.has("pred_texts"):
                labels = [
                    "{}".format(text) for text, score in zip(result.pred_texts, result.scores.cpu())
                ]
                v_pred = Visualizer(img, None)
                v_pred = v_pred.overlay_instances(
                    boxes=result.pred_boxes.tensor.clone().detach().cpu().numpy(),
                    labels=labels,
                    masks=result.pred_masks.clone().detach().cpu().numpy(),
                )
                pred_img = v_pred.get_image()
                vis_img = np.concatenate((vis_img, pred_img), axis=1)

            basename = os.path.basename(input["file_name"])
            if self.training:
                cv2.imwrite(
                    os.path.join(
                        self.output_dir, "training", str(storage.iter) + suffix + "_" + basename
                    ),
                    vis_img[:, :, ::-1],
                )
            else:
                cv2.imwrite(
                    os.path.join(self.output_dir, "inference", suffix + basename),
                    vis_img[:, :, ::-1],
                )

    @torch.no_grad()
    def visualize_training_enc_output(self, batched_inputs, output, images, dataset_id, suffix=""):
        if self.output_dir is None:
            return
        if self.training:
            storage = get_event_storage()
            os.makedirs(self.output_dir + "/training", exist_ok=True)
        else:
            os.makedirs(self.output_dir + "/inference", exist_ok=True)

        pred_logits = output["enc_outputs"]["pred_logits"]
        pred_boxes = output["enc_outputs"]["pred_boxes"]

        results, filter_inds = self.inference(pred_logits, pred_boxes, images.image_sizes)

        from detectron2.utils.visualizer import Visualizer

        for input, result in zip(batched_inputs, results):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            if "instances" in input:
                v_gt = v_gt.overlay_instances(
                    boxes=input["instances"].gt_boxes,
                )
            else:
                v_gt = v_gt.output
            anno_img = v_gt.get_image()

            labels = [
                "{}_{:.0f}%".format(pred_class, score * 100)
                for pred_class, score in zip(result.pred_classes.cpu(), result.scores.cpu())
            ]
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=result.pred_boxes.tensor.clone().detach().cpu().numpy(),
                labels=labels,
            )
            pred_img = v_pred.get_image()

            vis_img = np.concatenate((anno_img, pred_img), axis=1)

            basename = os.path.basename(input["file_name"])
            if self.training:
                cv2.imwrite(
                    os.path.join(
                        self.output_dir,
                        "training",
                        str(storage.iter) + suffix + "_enc_output_" + basename,
                    ),
                    vis_img[:, :, ::-1],
                )
            else:
                cv2.imwrite(
                    os.path.join(self.output_dir, "inference", suffix + "enc_output_" + basename),
                    vis_img[:, :, ::-1],
                )

    def set_model_language(self, model_language):
        self.model_language = model_language


def is_thing_stuff_overlap(metadata):
    thing_classes = metadata.get("thing_classes", [])
    stuff_classes = metadata.get("stuff_classes", [])
    if len(thing_classes) == 0 or len(stuff_classes) == 0:
        return False

    if set(thing_classes).issubset(set(stuff_classes)) or set(stuff_classes).issubset(
        set(thing_classes)
    ):
        return True
    else:
        return False


def get_text_list(metadata, dataset_entity):
    thing_classes = metadata.get("thing_classes", [])
    stuff_classes = metadata.get("stuff_classes", [])

    if dataset_entity == "thing+stuff" and stuff_classes[0] == "things":
        text_list = list(thing_classes) + list(stuff_classes[1:])

    elif dataset_entity == "thing+stuff" and is_thing_stuff_overlap(metadata):
        text_list = thing_classes if len(thing_classes) > len(stuff_classes) else stuff_classes

    elif dataset_entity == "thing+stuff":
        text_list = list(thing_classes) + list(stuff_classes)

    elif dataset_entity == "stuff":
        text_list = list(stuff_classes)

    elif dataset_entity == "thing":
        text_list = list(thing_classes)

    return text_list


def get_stuff_score(box_cls, metadata, dataset_entity):
    thing_classes = metadata.get("thing_classes", [])
    stuff_classes = metadata.get("stuff_classes", [])

    semantic_box_cls = box_cls.clone()

    if dataset_entity == "thing+stuff" and stuff_classes[0] == "things":
        num_thing_classes = len(thing_classes)

        semantic_box_cls_0 = box_cls[..., :num_thing_classes]
        semantic_box_cls_1 = box_cls[..., num_thing_classes:]
        semantic_box_cls_0, _ = semantic_box_cls_0.min(dim=2, keepdim=True)
        semantic_box_cls = torch.cat([semantic_box_cls_0, semantic_box_cls_1], dim=2)

    if dataset_entity == "thing+stuff" and is_thing_stuff_overlap(metadata):
        semantic_box_cls = box_cls.clone()

    if dataset_entity == "stuff":
        semantic_box_cls = box_cls.clone()

    return semantic_box_cls
