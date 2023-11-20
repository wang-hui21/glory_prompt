import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, GatedGraphConv

from models.base.layers import *
from models.component.candidate_encoder import *
from models.component.click_encoder import ClickEncoder
from models.component.entity_encoder import EntityEncoder, GlobalEntityEncoder
from models.component.nce_loss import NCELoss
from models.component.news_encoder import *
from models.component.user_encoder import *
from transformers import BertTokenizer, BertConfig, AutoTokenizer
from new.myBertForMaskedLM import CustomBertForMaskedLM


class GLORY(nn.Module):
    def __init__(self, cfg, glove_emb=None, entity_emb=None, answer_ids=None):
        super().__init__()

        self.cfg = cfg
        self.use_entity = cfg.model.use_entity

        self.news_dim = cfg.model.head_num * cfg.model.head_dim
        self.entity_dim = cfg.model.entity_emb_dim

        config = BertConfig.from_pretrained(cfg.token.bertmodel)
        self.BERT = CustomBertForMaskedLM(config, cfg)
        # self.BERT.resize_token_embeddings(cfg.token.vocab_size)

        for param in self.BERT.parameters():
            param.requires_grad = True

        self.answer_ids = answer_ids
        self.mask_token_id = 103
        self.loss_func = nn.CrossEntropyLoss()
        # -------------------------- Model --------------------------
        # News Encoder

        self.local_news_encoder = NewsEncoder(cfg, glove_emb)

        # GCN
        self.global_news_encoder = Sequential('x, index', [
            (GatedGraphConv(self.news_dim, num_layers=3, aggr='add'), 'x, index -> x'),
        ])
        # Entity
        if self.use_entity:
            pretrain = torch.from_numpy(entity_emb).float()
            self.entity_embedding_layer = nn.Embedding.from_pretrained(pretrain, freeze=False, padding_idx=0)

            self.local_entity_encoder = Sequential('x, mask', [
                (self.entity_embedding_layer, 'x -> x'),
                (EntityEncoder(cfg), 'x, mask -> x'),
            ])

            self.global_entity_encoder = Sequential('x, mask', [
                (self.entity_embedding_layer, 'x -> x'),
                (GlobalEntityEncoder(cfg), 'x, mask -> x'),
            ])
        # Click Encoder
        self.click_encoder = ClickEncoder(cfg)

        # User Encoder
        self.user_encoder = UserEncoder(cfg)

        # Candidate Encoder
        self.candidate_encoder = CandidateEncoder(cfg)

        # click prediction
        # self.click_predictor = DotProduct()
        # self.loss_fn = NCELoss()

    def forward(self, subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask, sentence, tokenizer,
                label=None):
        # -------------------------------------- clicked ----------------------------------
        mask = mapping_idx != -1
        mapping_idx[mapping_idx == -1] = 0

        batch_size, num_clicked, token_dim = mapping_idx.shape[0], mapping_idx.shape[1], candidate_news.shape[-1]
        clicked_entity = subgraph.x[mapping_idx, -8:-3]

        # News Encoder + GCN
        x_flatten = subgraph.x.view(1, -1, token_dim)
        x_encoded = self.local_news_encoder(x_flatten).view(-1, self.news_dim)

        graph_emb = self.global_news_encoder(x_encoded, subgraph.edge_index)

        clicked_origin_emb = x_encoded[mapping_idx, :].masked_fill(~mask.unsqueeze(-1), 0).view(batch_size, num_clicked,
                                                                                                self.news_dim)
        clicked_graph_emb = graph_emb[mapping_idx, :].masked_fill(~mask.unsqueeze(-1), 0).view(batch_size, num_clicked,
                                                                                               self.news_dim)

        # Attention pooling
        if self.use_entity:
            clicked_entity = self.local_entity_encoder(clicked_entity, None)
        else:
            clicked_entity = None

        clicked_total_emb = self.click_encoder(clicked_origin_emb, clicked_graph_emb, clicked_entity)
        user_emb = self.user_encoder(clicked_total_emb, mask)

        # ----------------------------------------- Candidate------------------------------------
        cand_title_emb = self.local_news_encoder(candidate_news)  # [8, 5, 400]  按顺序存储候选新闻embedding
        if self.use_entity:
            origin_entity, neighbor_entity = candidate_entity.split(
                [self.cfg.model.entity_size, self.cfg.model.entity_size * self.cfg.model.entity_neighbors], dim=-1)

            cand_origin_entity_emb = self.local_entity_encoder(origin_entity, None)
            cand_neighbor_entity_emb = self.global_entity_encoder(neighbor_entity, entity_mask)

            # cand_entity_emb = self.entity_encoder(candidate_entity, entity_mask).view(batch_size, -1, self.news_dim) # [8, 5, 400]
        else:
            cand_origin_entity_emb, cand_neighbor_entity_emb = None, None

        cand_final_emb = self.candidate_encoder(cand_title_emb, cand_origin_entity_emb, cand_neighbor_entity_emb)
        # ----------------------------------------- Score ------------------------------------
        sentences = [[d["sentence"] for d in sublist] for sublist in sentence]
        labels = [[d["target"] for d in sublist] for sublist in sentence]
        all_answer_logits = []
        all_labels = []
        labels = torch.LongTensor(labels).cuda()
        for i, sentence in enumerate(sentences):
            encode_dict = tokenizer.batch_encode_plus(
                sentence,
                add_special_tokens=True,
                padding='max_length',
                max_length=500,
                truncation=True,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            batch_enc = encode_dict['input_ids'].clone().detach().cuda()
            batch_attn = encode_dict['attention_mask'].clone().detach().cuda()
            batch_token = encode_dict['token_type_ids'].clone().detach().cuda()

            outputs = self.BERT(input_ids=batch_enc,
                                attention_mask=batch_attn,
                                token_type_ids=batch_token,
                                Uembedding=user_emb,
                                Cembedding=cand_final_emb)
            out_logits = outputs  # out_logits.shape=([500,30522])

            mask_position = batch_enc.eq(self.mask_token_id)  # mask_position.shape=([5,500])
            mask_logits = out_logits[mask_position, :].view(out_logits.size(0), -1, out_logits.size(-1))[:, -1, :]

            answer_logits = mask_logits[:, self.answer_ids]
            if i == 0:
                all_answer_logits = answer_logits
                all_labels = labels[i]
            else:
                all_answer_logits = np.concatenate((all_answer_logits, answer_logits), axis=0)
                all_labels = np.concatenate((all_labels, labels[i]), axis=0)  # 先观察labels的结构，再决定用什么方式连接
        loss = self.loss_func(all_answer_logits, all_labels)

        return loss, all_answer_logits.softmax(dim=1), all_labels

    def validation_process(self, subgraph, mappings, clicked_entity, candidate_emb, candidate_entity, batch_enc,
                           batch_attn, target, batch_token, entity_mask):

        batch_size, num_news, news_dim = 1, len(mappings), candidate_emb.shape[-1]

        title_graph_emb = self.global_news_encoder(subgraph.x, subgraph.edge_index)
        clicked_graph_emb = title_graph_emb[mappings, :].view(batch_size, num_news, news_dim)
        clicked_origin_emb = subgraph.x[mappings, :].view(batch_size, num_news, news_dim)

        # --------------------Attention Pooling
        if self.use_entity:
            clicked_entity_emb = self.local_entity_encoder(clicked_entity.unsqueeze(0), None)
        else:
            clicked_entity_emb = None

        clicked_final_emb = self.click_encoder(clicked_origin_emb, clicked_graph_emb, clicked_entity_emb)

        user_emb = self.user_encoder(clicked_final_emb)  # [1, 400]

        # ----------------------------------------- Candidate------------------------------------

        if self.use_entity:
            cand_entity_input = candidate_entity.unsqueeze(0)
            entity_mask = entity_mask.unsqueeze(0)
            origin_entity, neighbor_entity = cand_entity_input.split(
                [self.cfg.model.entity_size, self.cfg.model.entity_size * self.cfg.model.entity_neighbors], dim=-1)

            cand_origin_entity_emb = self.local_entity_encoder(origin_entity, None)
            cand_neighbor_entity_emb = self.global_entity_encoder(neighbor_entity, entity_mask)

        else:
            cand_origin_entity_emb = None
            cand_neighbor_entity_emb = None

        cand_final_emb = self.candidate_encoder(candidate_emb.unsqueeze(0), cand_origin_entity_emb,
                                                cand_neighbor_entity_emb)
        # ---------------------------------------------------------------------------------------
        # ----------------------------------------- Score ------------------------------------
        outputs = self.BERT(input_ids=batch_enc,
                            attention_mask=batch_attn,
                            token_type_ids=batch_token,
                            Uembedding=user_emb,
                            Cembedding=cand_final_emb)
        out_logits = outputs

        mask_position = batch_enc.eq(self.mask_token_id)
        mask_logits = out_logits[mask_position, :].view(out_logits.size(0), -1, out_logits.size(-1))[:, -1, :]

        answer_logits = mask_logits[:, self.answer_ids]

        loss = self.loss_func(answer_logits, target)

        return loss, answer_logits.softmax(dim=1)

