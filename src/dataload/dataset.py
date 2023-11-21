import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
import numpy as np
import re


class TrainDataset(IterableDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg, tokenizer, conti_tokens):
        super().__init__()
        self.filename = filename
        self.news_index = news_index
        self.news_input = news_input
        self.cfg = cfg
        self.local_rank = local_rank
        self.world_size = cfg.gpu_num

        self.tokenizer = tokenizer
        self.conti_tokens = conti_tokens

    def trans_to_nindex(self, nids):
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]  # 返回点击新闻列表对应的索引

    def pad_to_fix_len(self, x, fix_length, padding_front=True, padding_value=0):
        if padding_front:
            pad_x = [padding_value] * (fix_length - len(x)) + x[-fix_length:]
            mask = [0] * (fix_length - len(x)) + [1] * min(fix_length, len(x))
        else:
            pad_x = x[-fix_length:] + [padding_value] * (fix_length - len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
        return pad_x, np.array(mask, dtype='float32')

    # 函数的作用是将原始文本数据转化为模型所需的输入形式。
    def line_mapper(self, line):

        line = line.strip().split('\t')
        click_id = line[3].split()
        sess_pos = line[4].split()
        sess_neg = line[5].split()
        # clicked_index 和 clicked_mask 通过调用 pad_to_fix_len 函数来处理用户点击历史。这个函数的目的是将用户点击历史序列 click_id 填充到指定的固定长度，以便输入模型。
        # clicked_index 包含了填充后的用户点击历史的新闻索引，clicked_mask 是一个掩码，指示了哪些部分是真实的历史点击，哪些部分是填充的
        clicked_index, clicked_mask = self.pad_to_fix_len(self.trans_to_nindex(click_id), self.cfg.model.his_size)
        # clicked_input 根据 clicked_index 从预先加载的新闻特征中提取用户点击历史的新闻特征。
        clicked_input = self.news_input[clicked_index]

        label = 0
        sample_news = self.trans_to_nindex(sess_pos + sess_neg)
        candidate_input = self.news_input[sample_news]

        return clicked_input, clicked_mask, candidate_input, label

    def __iter__(self):
        file_iter = open(self.filename)
        return map(self.line_mapper, file_iter)


class TrainGraphDataset(TrainDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors,
                 tokenizer, conti_tokens):
        super().__init__(filename, news_index, news_input, local_rank, cfg, tokenizer, conti_tokens)
        self.neighbor_dict = neighbor_dict
        self.news_graph = news_graph.to(local_rank, non_blocking=True)

        self.batch_size = cfg.batch_size / cfg.gpu_num
        self.entity_neighbors = entity_neighbors
        self.tokenizer = tokenizer
        self.conti_tokens = conti_tokens

    def line_mapper(self, line, sum_num_news):

        line = line.strip().split('\t')
        click_id = line[3].split()[-self.cfg.model.his_size:]  # 取出指定数量的新闻 最新阅读的新闻
        sess_pos = line[4].split()  # 正样本只有一个
        sess_neg = line[5].split()  # 负样本有多个
        imp = line[0].split()  # 序号
        # ------------------ Clicked News ----------------------
        # ------------------ News Subgraph ---------------------
        top_k = len(click_id)
        click_idx = self.trans_to_nindex(click_id)  # 返回历史新闻对应的索引
        source_idx = click_idx
        for _ in range(self.cfg.model.k_hops):  # 指定寻找几跳的邻居，此处循环就执行几次
            current_hop_idx = []
            for news_idx in source_idx:
                current_hop_idx.extend(self.neighbor_dict[news_idx][:self.cfg.model.num_neighbors])  # 取出指定数量的新闻的邻居
            source_idx = current_hop_idx  # 更新索引信息，跳数加一
            click_idx.extend(current_hop_idx)  # 将挑选出来的新闻合并起来

        sub_news_graph, mapping_idx = self.build_subgraph(click_idx, top_k, sum_num_news)

        padded_maping_idx = F.pad(mapping_idx, (self.cfg.model.his_size - len(mapping_idx), 0), "constant",
                                  -1)  # 填充mapping_idx长度

        # ------------------ Candidate News ---------------------
        label = 0
        sample_news = self.trans_to_nindex(sess_pos + sess_neg)  # 取出正样本和负样本的序列
        candidate_input = self.news_input[sample_news]  # 取出候选新闻

        # ------------------ Entity Subgraph --------------------
        if self.cfg.model.use_entity:
            origin_entity = candidate_input[:,
                            -3 - self.cfg.model.entity_size:-3]  # [5, 5]     此处截取原文中实体的embedding，正样本有一个，负样本有四个，所以合起来有五个候选新闻，第一个表示正样本
            candidate_neighbor_entity = np.zeros(
                ((self.cfg.npratio + 1) * self.cfg.model.entity_size, self.cfg.model.entity_neighbors),
                dtype=np.int64)  # [5*5, 20]
            for cnt, idx in enumerate(origin_entity.flatten()):
                if idx == 0: continue
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0: continue
                valid_len = min(entity_dict_length, self.cfg.model.entity_neighbors)
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]

            candidate_neighbor_entity = candidate_neighbor_entity.reshape(self.cfg.npratio + 1,
                                                                          self.cfg.model.entity_size * self.cfg.model.entity_neighbors)  # [5, 5*20]
            entity_mask = candidate_neighbor_entity.copy()
            entity_mask[entity_mask > 0] = 1
            candidate_entity = np.concatenate((origin_entity, candidate_neighbor_entity),
                                              axis=-1)  # 将邻居实体和自身实体连接起来，准备下一步的操作
        else:
            candidate_entity = np.zeros(1)
            entity_mask = np.zeros(1)

        template1 = ''.join(self.conti_tokens[0]) + "<user_sentence>"
        template2 = ''.join(self.conti_tokens[1]) + "<candidate_news>"
        # template1 = ''.join(self.conti_tokens[0]) + " 的类别是 "+"<ucate>"
        # template2 = ''.join(self.conti_tokens[1])  +" 的类别是 " +"<ccate>"
        template3 = "Does the user click the news? [MASK]"
        template = template1 + "[SEP]" + template2 + "[SEP]" + template3
        # 此处不用做过多的处理，只需要将新闻的数量用数字代替，在模板中占据所需要的位置，其中新闻的数量默认是50，候选新闻的数量为5
        his_news_num = []
        data = []
        for i, news in enumerate(click_id):
            his_news_num.append(str(i))  # 用数字表示表示浏览历史，占位，以便后面将embedding进行替换
            # hcate=cate+" "+subcate
            # his_cate.append(hcate)
        his_sen = '[NSEP] ' + ' [NSEP] '.join(his_news_num)
        # his_cat = '[NSEP] ' + ' [NSEP] '.join(his_cate)
        his_sen_ids = self.tokenizer.encode(his_sen,
                                            add_special_tokens=False)  # add_special_tokens=False表示在tokenize时不添加特殊token,如[CLS]等。
        # his_cat_ids = self.tokenizer.encode(his_cat, add_special_tokens=False)[:max_his_len]
        his_sen = self.tokenizer.decode(his_sen_ids)
        # his_cat = self.tokenizer.decode(his_cat_ids)
        base_sentence = template.replace("<user_sentence>", his_sen)

        #             base_sentence = base_sentence.replace("<ucate>", his_cat)
        # base_sentence = template.replace("<ucate>", his_cat)
        for i, news in enumerate(sess_pos):

            sentence = base_sentence.replace("<candidate_news>", str(i))
            # sentence = sentence.replace("<ccate>", cate+" "+subcate)
            # sentence = base_sentence.replace("<ccate>", cate+" "+subcate)
            data.append({'sentence': sentence, 'target': 1, 'imp': imp})

            for j, n in enumerate(sess_neg):
                sentence = base_sentence.replace("<candidate_news>", str(j))
                # sentence = sentence.replace("<ccate>", neg_cate + " " + neg_subcate)
                # sentence = base_sentence.replace("<ccate>", neg_cate + " " + neg_subcate)
                data.append({'sentence': sentence, 'target': 0, 'imp': imp})
        return data, sum_num_news + sub_news_graph.num_nodes


    # return sub_news_graph, padded_maping_idx, candidate_input, candidate_entity, entity_mask, label, \
    #         , sentence


    # 这个函数的主要作用是为了构建一个与原始图相关的子图，该子图包含唯一的节点和相应的边信息，以便后续进行进一步的计算和处理。
    def build_subgraph(self, subset, k, sum_num_nodes):
        device = self.news_graph.x.device  # 获取设备信息，用于确保新创新的张量也位于相同的设备上

        if not subset:  # 如果传入的subset是空列表，将其设置为包含一个元素0的列表，确保后面的代码可以正常运行
            subset = [0]

        subset = torch.tensor(subset, dtype=torch.long, device=device)  # 将subset转化为pytorch张量

        unique_subset, unique_mapping = torch.unique(subset, sorted=True,
                                                     return_inverse=True)  # 获取 subset 中的唯一值，并返回两个张量。unique_subset 包含了唯一的节点索引，而 unique_mapping 包含了将原始 subset 中的值映射到 unique_subset 中的索引。
        subemb = self.news_graph.x[unique_subset]  # 使用唯一的节点索引 unique_subset 从原始 self.news_graph.x 中提取节点特征，以构建子图的节点特征。
        # 调用了一个名为 subgraph 的函数，用于构建子图的边信息。这个函数接受唯一的节点索引 unique_subset，原始图的边信息 self.news_graph.edge_index 和边属性 self.news_graph.edge_attr，还有一些其他参数。它会返回子图的边索引 sub_edge_index 和边属性 sub_edge_attr。
        sub_edge_index, sub_edge_attr = subgraph(unique_subset, self.news_graph.edge_index, self.news_graph.edge_attr,
                                                 relabel_nodes=True, num_nodes=self.news_graph.num_nodes)

        sub_news_graph = Data(x=subemb, edge_index=sub_edge_index, edge_attr=sub_edge_attr)

        return sub_news_graph, unique_mapping[:k] + sum_num_nodes

    def __iter__(self):
        while True:
            clicked_graphs = []
            candidates = []
            mappings = []
            labels = []

            candidate_entity_list = []
            entity_mask_list = []
            sentences = []
            sum_num_news = 0
            with open(self.filename) as f:
                for line in f:  # 对于每一个用户，调用一次line_mapper函数，产生该用户对应的子图，候选输入，候选实体等
                    # if line.strip().split('\t')[3]:
                    sub_newsgraph, padded_mapping_idx, candidate_input, candidate_entity, entity_mask, label, sum_num_news, sentence = self.line_mapper(
                        line, sum_num_news)

                    clicked_graphs.append(sub_newsgraph)
                    candidates.append(torch.from_numpy(candidate_input))
                    mappings.append(padded_mapping_idx)
                    labels.append(label)
                    sentences.append(sentence)
                    candidate_entity_list.append(torch.from_numpy(candidate_entity))
                    entity_mask_list.append(torch.from_numpy(entity_mask))

                    # 这段代码的作用是将多个数据组合成一个批次，然后以生成器的方式逐个返回批次数据，同时清空用于下一批次的数据。
                    if len(clicked_graphs) == self.batch_size:
                        batch = Batch.from_data_list(clicked_graphs)

                        candidates = torch.stack(candidates)
                        mappings = torch.stack(mappings)
                        candidate_entity_list = torch.stack(candidate_entity_list)
                        entity_mask_list = torch.stack(entity_mask_list)

                        # sentences = stack(sentences)
                        labels = torch.tensor(labels, dtype=torch.long)
                        yield batch, mappings, candidates, candidate_entity_list, entity_mask_list, sentences, labels
                        clicked_graphs, mappings, candidates, labels, candidate_entity_list, entity_mask_list, sentences = [], [], [], [], [], [], []
                        sum_num_news = 0

                if (len(clicked_graphs) > 0):
                    batch = Batch.from_data_list(clicked_graphs)

                    candidates = torch.stack(candidates)
                    mappings = torch.stack(mappings)

                    sentences = torch.stack(sentences)
                    candidate_entity_list = torch.stack(candidate_entity_list)
                    entity_mask_list = torch.stack(entity_mask_list)
                    labels = torch.tensor(labels, dtype=torch.long)

                    yield batch, mappings, candidates, candidate_entity_list, entity_mask_list, sentences, labels
                    f.seek(0)


class ValidGraphDataset(TrainGraphDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors,
                 news_entity, tokenizer, conti_tokens):
        super().__init__(filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors,
                         tokenizer, conti_tokens)
        self.news_graph.x = torch.from_numpy(self.news_input).to(local_rank, non_blocking=True)
        self.news_entity = news_entity
        self.tokenizer = tokenizer
        self.conti_tokens = conti_tokens

    def line_mapper(self, line):

        line = line.strip().split('\t')
        click_id = line[3].split()[-self.cfg.model.his_size:]
        imp = line[0].split()

        click_idx = self.trans_to_nindex(click_id)
        clicked_entity = self.news_entity[click_idx]
        source_idx = click_idx
        for _ in range(self.cfg.model.k_hops):
            current_hop_idx = []
            for news_idx in source_idx:
                current_hop_idx.extend(self.neighbor_dict[news_idx][:self.cfg.model.num_neighbors])
            source_idx = current_hop_idx
            click_idx.extend(current_hop_idx)
        sub_news_graph, mapping_idx = self.build_subgraph(click_idx, len(click_id), 0)

        # ------------------ Entity --------------------
        labels = np.array([int(i.split('-')[1]) for i in line[4].split()])
        candidate_index = self.trans_to_nindex([i.split('-')[0] for i in line[4].split()])
        candidate_input = self.news_input[candidate_index]
        pos = []
        neg = []
        for i, news in zip(labels, candidate_input):
            if i == 1:
                pos.append(news)
            else:
                neg.append(news)

        sentence = self.build_prompt(click_id, pos, neg, imp)
        if self.cfg.model.use_entity:
            origin_entity = self.news_entity[candidate_index]
            candidate_neighbor_entity = np.zeros(
                (len(candidate_index) * self.cfg.model.entity_size, self.cfg.model.entity_neighbors), dtype=np.int64)
            for cnt, idx in enumerate(origin_entity.flatten()):
                if idx == 0: continue
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0: continue
                valid_len = min(entity_dict_length, self.cfg.model.entity_neighbors)
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]

            candidate_neighbor_entity = candidate_neighbor_entity.reshape(len(candidate_index),
                                                                          self.cfg.model.entity_size * self.cfg.model.entity_neighbors)

            entity_mask = candidate_neighbor_entity.copy()
            entity_mask[entity_mask > 0] = 1

            candidate_entity = np.concatenate((origin_entity, candidate_neighbor_entity), axis=-1)
        else:
            candidate_entity = np.zeros(1)
            entity_mask = np.zeros(1)

        batch = Batch.from_data_list([sub_news_graph])

        return batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, sentence, labels

    def __iter__(self):
        for line in open(self.filename):
            if line.strip().split('\t')[3]:
                batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, sentence, labels = self.line_mapper(
                    line)
            yield batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, sentence, labels


class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


