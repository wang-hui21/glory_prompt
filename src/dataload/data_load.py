import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.utils import to_undirected
from tqdm import tqdm
import pickle
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer
from dataload.dataset import *


def load_data(cfg, mode='train', model=None, local_rank=0, tokenizer=None, conti_tokens=None):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    # ------------- load news.tsv-------------
    news_index = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))

    news_input = pickle.load(open(Path(data_dir[mode]) / "nltk_token_news.bin", "rb"))
    # ------------- load behaviors_np{X}.tsv --------------
    if mode == 'train':
        target_file = Path(data_dir[mode]) / f"behaviors_np{cfg.npratio}_{local_rank}.tsv"
        if cfg.model.use_graph:
            news_graph = torch.load(Path(data_dir[mode]) / "nltk_news_graph.pt")

            if cfg.model.directed is False:
                news_graph.edge_index, news_graph.edge_attr = to_undirected(news_graph.edge_index, news_graph.edge_attr)
            print(f"[{mode}] News Graph Info: {news_graph}")

            news_neighbors_dict = pickle.load(open(Path(data_dir[mode]) / "news_neighbor_dict.bin", "rb"))

            if cfg.model.use_entity:
                entity_neighbors = pickle.load(open(Path(data_dir[mode]) / "entity_neighbor_dict.bin", "rb"))
                total_length = sum(len(lst) for lst in entity_neighbors.values())
                print(f"[{mode}] entity_neighbor list Length: {total_length}")
            else:
                entity_neighbors = None

            dataset = MyDataset(
                filename=target_file,
                news_index=news_index,
                news_input=news_input,
                local_rank=local_rank,
                cfg=cfg,
                neighbor_dict=news_neighbors_dict,
                news_graph=news_graph,
                entity_neighbors=entity_neighbors,
                tokenizer=tokenizer,
                conti_tokens=conti_tokens,
                status=mode
            )
            # train_sampler = DistributedSampler(dataset,
            #                                    rank=local_rank,
            #                                    num_replicas=cfg.gpu_num,
            #                                    shuffle=True)
            # train_kwargs = {'batch_size': cfg.batch_size, 'sampler': train_sampler,
            #                 'shuffle': False, 'pin_memory': True, 'collate_fn': dataset.collate_fn}
            # nw = 8
            # cuda_kwargs = {'num_workers': nw, 'pin_memory': True}
            # train_kwargs.update(cuda_kwargs)
            dataloader = DataLoader(dataset,
                                    batch_size=1,
                                    # batch_size=int(cfg.batch_   size / cfg.gpu_num),
                                    # pin_memory=True, # collate_fn already puts data to GPU
                                    collate_fn=lambda b: collate_fn(b, tokenizer, cfg))
            print("加载完毕")

        return dataloader
    elif mode in ['val', 'test']:
        news_dataset = NewsDataset(news_input)
        news_dataloader = DataLoader(news_dataset,
                                     batch_size=int(cfg.batch_size * cfg.gpu_num),
                                     num_workers=cfg.num_workers)

        stacked_news = []
        with torch.no_grad():
            for news_batch in tqdm(news_dataloader, desc=f"[{local_rank}] Processing validation News Embedding"):
                if cfg.model.use_graph:
                    batch_emb = model.module.local_news_encoder(news_batch.long().unsqueeze(0).to(local_rank)).squeeze(
                        0).detach()
                else:
                    batch_emb = model.module.local_news_encoder(news_batch.long().unsqueeze(0).to(local_rank)).squeeze(
                        0).detach()
                stacked_news.append(batch_emb)
        news_emb = torch.cat(stacked_news, dim=0).cpu().numpy()

        target_file = Path(data_dir[mode]) / f"behaviors_np{cfg.npratio}_{local_rank}.tsv"
        news_graph = torch.load(Path(data_dir[mode]) / "nltk_news_graph.pt")

        news_neighbors_dict = pickle.load(open(Path(data_dir[mode]) / "news_neighbor_dict.bin", "rb"))

        if cfg.model.directed is False:
            news_graph.edge_index, news_graph.edge_attr = to_undirected(news_graph.edge_index, news_graph.edge_attr)
        print(f"[{mode}] News Graph Info: {news_graph}")

        if cfg.model.use_entity:
            # entity_graph = torch.load(Path(data_dir[mode]) / "entity_graph.pt")
            entity_neighbors = pickle.load(open(Path(data_dir[mode]) / "entity_neighbor_dict.bin", "rb"))
            total_length = sum(len(lst) for lst in entity_neighbors.values())
            print(f"[{mode}] entity_neighbor list Length: {total_length}")
        # convert the news to embedding
        if mode == 'val':
            dataset = MyDataset(
                filename=target_file,
                news_index=news_index,
                news_input=news_emb,
                local_rank=local_rank,
                cfg=cfg,
                neighbor_dict=news_neighbors_dict,
                news_graph=news_graph,
                entity_neighbors=entity_neighbors,
                tokenizer=tokenizer,
                conti_tokens=conti_tokens,
                status=mode,
                news_entity=news_input[:, -8:-3]
            )

            dataloader = DataLoader(dataset,
                                    batch_size=1,
                                    # batch_size=int(cfg.batch_   size / cfg.gpu_num),
                                    # pin_memory=True, # collate_fn already puts data to GPU
                                    collate_fn=lambda b: collate_fn(b, tokenizer, cfg))

        return dataloader


def collate_fn(batch, tokenizer, cfg):
    sentence = [x['sentence'] for x in batch]
    target = [x['target'] for x in batch]
    imp = [x['imp'] for x in batch]
    subgraph = [x['subgraph'] for x in batch]
    mapping_idx = [x['mapping_idx'] for x in batch]
    candidate_news = [torch.from_numpy(x['candidate_news']) for x in batch]
    candidate_entity = [torch.from_numpy(x['candidate_entity']) for x in batch]
    entity_mask = [torch.from_numpy(x['entity_mask']) for x in batch]
    if 'num_nodes' in batch[0]:
        num_nodes = [x['num_nodes'] for x in batch]

        num_news = 0
        for i, mapping in enumerate(mapping_idx):
            mapping_idx[i] = mapping + num_news
            num_news = num_news + num_nodes[i]
            mapping_idx[i] = F.pad(mapping_idx[i], (cfg.model.his_size - len(mapping_idx[i]), 0), "constant",
                                   -1)  # 填充mapping_idx长度

        # 这段代码的作用是将多个数据组合成一个批次，然后以生成器的方式逐个返回批次数据，同时清空用于下一批次的数据。

        batch = Batch.from_data_list(subgraph)

        candidates = torch.stack(candidate_news)
        mappings = torch.stack(mapping_idx)
        candidate_entity_list = torch.stack(candidate_entity)
        entity_mask_list = torch.stack(entity_mask)

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
        batch_enc = encode_dict['input_ids']
        batch_attn = encode_dict['attention_mask']
        batch_token = encode_dict['token_type_ids']
        target = torch.LongTensor(target)
        return batch, mappings, candidates, candidate_entity_list, entity_mask_list, batch_enc, batch_token, batch_attn, target, imp

    else:
        batch = Batch.from_data_list(subgraph)
        clicked_entity = [x['clicked_entity'] for x in batch]
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
        batch_enc = encode_dict['input_ids']
        batch_attn = encode_dict['attention_mask']
        batch_token = encode_dict['token_type_ids']
        target = torch.LongTensor(target)
        return batch, mapping_idx, clicked_entity, candidate_news, candidate_entity, entity_mask, batch_enc, batch_token, batch_attn, target, imp
