import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
import numpy as np
import re


class MyDataset(Dataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors,
                 tokenizer, conti_tokens, status, news_entity=None):
        self.filename = filename
        self.news_index = news_index
        self.news_input = news_input
        self.neighbor_dict = neighbor_dict
        self.news_graph = news_graph.to(local_rank, non_blocking=True)

        self.batch_size = cfg.batch_size / cfg.gpu_num
        self.entity_neighbors = entity_neighbors
        self.tokenizer = tokenizer
        self.conti_tokens = conti_tokens
        self.cfg = cfg
        self.data = []

        self.news_graph.x = torch.from_numpy(self.news_input).to(local_rank, non_blocking=True)
        self.news_entity = news_entity
        self.status = status
        self.load()

    # def __len__(self):
    #     return len(self.data)

    # def __getitem__(self, item):
    #     return self.data[item]

    def trans_to_nindex(self, nids):
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]  # 返回点击新闻列表对应的索引

    def prepro_train(self, filename):

        with open(filename) as f:
            sum_num_news = 0
            for i, line in enumerate(f):
                if i >= 10000:
                    break
                sum_num_news = sum_num_news + 1
                print(sum_num_news)
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
                        current_hop_idx.extend(
                            self.neighbor_dict[news_idx][:self.cfg.model.num_neighbors])  # 取出指定数量的新闻的邻居
                    source_idx = current_hop_idx  # 更新索引信息，跳数加一
                    click_idx.extend(current_hop_idx)  # 将挑选出来的新闻合并起来

                sub_news_graph, mapping_idx = self.build_subgraph(click_idx, top_k)

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
                    self.data.append({'sentence': sentence, 'target': 1, 'imp': imp, 'subgraph': sub_news_graph,
                                      'mapping_idx': mapping_idx,
                                      'candidate_news': candidate_input[0], 'candidate_entity': candidate_entity[0],
                                      'entity_mask': entity_mask[0], 'num_nodes': sub_news_graph.num_nodes})

                    for j, n in enumerate(sess_neg):
                        sentence = base_sentence.replace("<candidate_news>", str(j))
                        # sentence = sentence.replace("<ccate>", neg_cate + " " + neg_subcate)
                        # sentence = base_sentence.replace("<ccate>", neg_cate + " " + neg_subcate)
                        self.data.append({'sentence': sentence, 'target': 0, 'imp': imp, 'subgraph': sub_news_graph,
                                          'mapping_idx': mapping_idx,
                                          'candidate_news': candidate_input[j + 1],
                                          'candidate_entity': candidate_entity[j + 1],
                                          'entity_mask': entity_mask[j + 1], 'num_nodes': sub_news_graph.num_nodes})

    # return sub_news_graph, padded_maping_idx, candidate_input, candidate_entity, entity_mask, label, \
    #         , sentence

    # 这个函数的主要作用是为了构建一个与原始图相关的子图，该子图包含唯一的节点和相应的边信息，以便后续进行进一步的计算和处理。
    def build_subgraph(self, subset, k):
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

        return sub_news_graph, unique_mapping[:k]

    def prepro_val(self, filename):
        for line in open(self.filename):
            if line.strip().split('\t')[3]:
                line = line.strip().split('\t')
                imp = line[0].split()
                click_id = line[3].split()[-self.cfg.model.his_size:]

                click_idx = self.trans_to_nindex(click_id)
                clicked_entity = self.news_entity[click_idx]
                source_idx = click_idx
                for _ in range(self.cfg.model.k_hops):
                    current_hop_idx = []
                    for news_idx in source_idx:
                        current_hop_idx.extend(self.neighbor_dict[news_idx][:self.cfg.model.num_neighbors])
                    source_idx = current_hop_idx
                    click_idx.extend(current_hop_idx)
                sub_news_graph, mapping_idx = self.build_subgraph(click_idx, len(click_id))

                # ------------------ Entity --------------------
                labels = np.array([int(i.split('-')[1]) for i in line[4].split()])
                candidate_index = self.trans_to_nindex([i.split('-')[0] for i in line[4].split()])
                candidate_input = self.news_input[candidate_index]

                if self.cfg.model.use_entity:
                    origin_entity = self.news_entity[candidate_index]
                    candidate_neighbor_entity = np.zeros(
                        (len(candidate_index) * self.cfg.model.entity_size, self.cfg.model.entity_neighbors),
                        dtype=np.int64)
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

                template1 = ''.join(self.conti_tokens[0]) + "<user_sentence>"
                template2 = ''.join(self.conti_tokens[1]) + "<candidate_news>"
                # template1 = ''.join(self.conti_tokens[0]) + " 的类别是 "+"<ucate>"
                # template2 = ''.join(self.conti_tokens[1])  +" 的类别是 " +"<ccate>"
                template3 = "Does the user click the news? [MASK]"
                template = template1 + "[SEP]" + template2 + "[SEP]" + template3
                # 此处不用做过多的处理，只需要将新闻的数量用数字代替，在模板中占据所需要的位置，其中新闻的数量默认是50，候选新闻的数量为5
                his_news_num = []
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
                for i, news in enumerate(candidate_index):
                    sentence = base_sentence.replace("<candidate_news>", str(i))
                    # sentence = sentence.replace("<ccate>", cate+" "+subcate)
                    # sentence = base_sentence.replace("<ccate>", cate+" "+subcate)
                    self.data.append({'sentence': sentence, 'target': labels[i], 'imp': imp, 'subgraph': sub_news_graph,
                                      'mapping_idx': mapping_idx,
                                      'candidate_news': candidate_input[i], 'candidate_entity': candidate_entity[i],
                                      'entity_mask': entity_mask[i], 'num_nodes': sub_news_graph.num_nodes,
                                      'clicked_entity': clicked_entity})

    def __iter__(self):
        if self.status == 'train':
            self.prepro_train(self.filename)
        else:
            self.prepro_val(self.filename)
        return self.data


class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


