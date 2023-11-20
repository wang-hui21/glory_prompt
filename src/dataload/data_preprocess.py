import collections
import os
from pathlib import Path
from nltk.tokenize import word_tokenize
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

import torch.nn.functional as F
from tqdm import tqdm
import random
import pickle
from collections import Counter
import numpy as np
import torch
import json
import itertools


def update_dict(target_dict, key, value=None):
    """
    Function for updating dict with key / key+value

    Args:
        target_dict(dict): target dict
        key(string): target key
        value(Any, optional): if None, equals len(dict+1)
    """
    if key not in target_dict:
        if value is None:
            target_dict[key] = len(target_dict) + 1
        else:
            target_dict[key] = value


def get_sample(all_elements, num_sample):
    if num_sample > len(all_elements):
        return random.sample(all_elements * (num_sample // len(all_elements) + 1), num_sample)
    else:
        return random.sample(all_elements, num_sample)


def prepare_distributed_data(cfg, mode="train"):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir,
                "test": cfg.dataset.test_dir}  # 不同类型的数据地址和数据类型结合成键值对，生成字典
    # check
    target_file = os.path.join(data_dir[mode], f"behaviors_np{cfg.npratio}_0.tsv")  # 自定义实验结果的存储文件
    if os.path.exists(target_file) and not cfg.reprocess:  # 目标文件如果存在，跳出此函数
        return 0
    print(f'Target_file is not exist. New behavior file in {target_file}')

    behaviors = []
    behavior_file_path = os.path.join(data_dir[mode], 'behaviors.tsv')  # 文件名和地址拼接

    if mode == 'train':
        with open(behavior_file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):  # 按行取出文件内容
                iid, uid, time, history, imp = line.strip().split('\t')  # 取出前后空格，按照‘\t’为间隔分割数据
                impressions = [x.split('-') for x in imp.split(' ')]  # 先按照空格分割，再按照‘-’分割，列表推导
                pos, neg = [], []
                for news_ID, label in impressions:  # 将正样本和负样本分开
                    if label == '0':
                        neg.append(news_ID)
                    elif label == '1':
                        pos.append(news_ID)
                if len(pos) == 0 or len(neg) == 0:
                    continue
                for pos_id in pos:  # 把每个用户中的正样本分散放在behaviors中，每组对应一个正样本，多个负样本，此处怎样处理不影响后续操作，都是判断正样本是否会选中，其中有几个正样本不影响结果
                    neg_candidate = get_sample(neg, cfg.npratio)  # 挑选指定数量的负样本
                    neg_str = ' '.join(neg_candidate)
                    new_line = '\t'.join([iid, uid, time, history, pos_id, neg_str]) + '\n'
                    behaviors.append(new_line)  # 可能会有用户名重复，因为正样本可能不止一个
        random.shuffle(behaviors)  # 使behaviors中的数据产生新的序列

        behaviors_per_file = [[] for _ in range(cfg.gpu_num)]  # 创建一个包含多个子列表的列表，每个子列表都用于存储用户交互行为数据的不同部分。提高代码运行效率
        for i, line in enumerate(behaviors):
            behaviors_per_file[i % cfg.gpu_num].append(line)  # 将数据添加到对应的gpu分组中，方便并行运行

    elif mode in ['val', 'test']:
        behaviors_per_file = [[] for _ in range(cfg.gpu_num)]
        behavior_file_path = os.path.join(data_dir[mode], 'behaviors.tsv')
        with open(behavior_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f)):
                behaviors_per_file[i % cfg.gpu_num].append(line)

    print(f'[{mode}]Writing files...')
    for i in range(cfg.gpu_num):  # 处理好的数据写入新的文件
        processed_file_path = os.path.join(data_dir[mode], f'behaviors_np{cfg.npratio}_{i}.tsv')
        with open(processed_file_path, 'w') as f:
            f.writelines(behaviors_per_file[i])

    return len(behaviors)


def read_raw_news(cfg, file_path, mode='train'):
    """
    Function for reading the raw news file, news.tsv

    Args:
        cfg:
        file_path(Path):                path of news.tsv
        mode(string, optional):        train or test


    Returns:
        tuple:     (news, news_index, category_dict, subcategory_dict, word_dict)

    """
    import nltk
    nltk.download('punkt')

    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    if mode in ['val', 'test']:  # 测试时打开训练过程中保存的文件
        news_dict = pickle.load(open(Path(data_dir["train"]) / "news_dict.bin", "rb"))
        entity_dict = pickle.load(open(Path(data_dir["train"]) / "entity_dict.bin", "rb"))
        news = pickle.load(open(Path(data_dir["train"]) / "nltk_news.bin", "rb"))
    else:  # 训练时，初始化各个列表
        news = {}
        news_dict = {}
        entity_dict = {}

    category_dict = {}
    subcategory_dict = {}
    word_cnt = Counter()  # Counter is a subclass of the dictionary dict.

    num_line = len(open(file_path, encoding='utf-8').readlines())  # 统计新闻行数
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=num_line, desc=f"[{mode}]Processing raw news"):
            # split one line
            split_line = line.strip('\n').split('\t')
            news_id, category, subcategory, title, abstract, url, t_entity_str, _ = split_line
            update_dict(target_dict=news_dict, key=news_id)  # 将新闻id添加到列表中，值用数字表示

            # Entity
            if t_entity_str:
                entity_ids = [obj["WikidataId"] for obj in json.loads(t_entity_str)]
                [update_dict(target_dict=entity_dict, key=entity_id) for entity_id in entity_ids]  # 实体也添加到列表中
            else:
                entity_ids = t_entity_str

            tokens = word_tokenize(title.lower(), language=cfg.dataset.dataset_lang)  # 将标题划分为多个token

            update_dict(target_dict=news, key=news_id, value=[tokens, category, subcategory, entity_ids,
                                                              news_dict[news_id]])  # 新闻id和新闻序号对应上

            if mode == 'train':
                update_dict(target_dict=category_dict, key=category)
                update_dict(target_dict=subcategory_dict, key=subcategory)
                word_cnt.update(tokens)

        if mode == 'train':
            word = [k for k, v in word_cnt.items() if v > cfg.model.word_filter_num]  # 列表推导，提取出要求数量的键
            word_dict = {k: v for k, v in zip(word, range(1, len(word) + 1))}  # 字典推导
            return news, news_dict, category_dict, subcategory_dict, entity_dict, word_dict
        else:  # val, test
            return news, news_dict, None, None, entity_dict, None


# 将新闻的各个属性，全部对应存储到列表中
def read_parsed_news(cfg, news, news_dict,
                     category_dict=None, subcategory_dict=None, entity_dict=None,
                     word_dict=None):
    news_num = len(news) + 1
    news_category, news_subcategory, news_index = [np.zeros((news_num, 1), dtype='int32') for _ in range(3)]  # 列表推导
    news_entity = np.zeros((news_num, 5), dtype='int32')

    news_title = np.zeros((news_num, cfg.model.title_size), dtype='int32')

    for _news_id in tqdm(news, total=len(news), desc="Processing parsed news"):
        _title, _category, _subcategory, _entity_ids, _news_index = news[_news_id]  # 从字典中提取数据

        news_category[_news_index, 0] = category_dict[_category] if _category in category_dict else 0
        news_subcategory[_news_index, 0] = subcategory_dict[_subcategory] if _subcategory in subcategory_dict else 0
        news_index[_news_index, 0] = news_dict[_news_id]

        # entity
        entity_index = [entity_dict[entity_id] if entity_id in entity_dict else 0 for entity_id in _entity_ids]
        news_entity[_news_index, :min(cfg.model.entity_size, len(_entity_ids))] = entity_index[:cfg.model.entity_size]

        for _word_id in range(min(cfg.model.title_size, len(_title))):
            if _title[_word_id] in word_dict:
                news_title[_news_index, _word_id] = word_dict[_title[_word_id]]

    return news_title, news_entity, news_category, news_subcategory, news_index


def prepare_preprocess_bin(cfg, mode):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    if cfg.reprocess is True:
        # Glove
        nltk_news, nltk_news_dict, category_dict, subcategory_dict, entity_dict, word_dict = read_raw_news(
            file_path=Path(data_dir[mode]) / "news.tsv",
            cfg=cfg,
            mode=mode,
        )

        if mode == "train":
            pickle.dump(category_dict, open(Path(data_dir[mode]) / "category_dict.bin", "wb"))
            pickle.dump(subcategory_dict, open(Path(data_dir[mode]) / "subcategory_dict.bin", "wb"))
            pickle.dump(word_dict, open(Path(data_dir[mode]) / "word_dict.bin", "wb"))
        else:
            category_dict = pickle.load(open(Path(data_dir["train"]) / "category_dict.bin", "rb"))
            subcategory_dict = pickle.load(open(Path(data_dir["train"]) / "subcategory_dict.bin", "rb"))
            word_dict = pickle.load(open(Path(data_dir["train"]) / "word_dict.bin", "rb"))

        pickle.dump(entity_dict, open(Path(data_dir[mode]) / "entity_dict.bin", "wb"))
        pickle.dump(nltk_news, open(Path(data_dir[mode]) / "nltk_news.bin", "wb"))
        pickle.dump(nltk_news_dict, open(Path(data_dir[mode]) / "news_dict.bin", "wb"))
        nltk_news_features = read_parsed_news(cfg, nltk_news, nltk_news_dict,
                                              category_dict, subcategory_dict, entity_dict,
                                              word_dict)  # 将返回值全都存储到一个变量中
        news_input = np.concatenate([x for x in nltk_news_features], axis=1)  # 将变量中所有属性拼接起来
        pickle.dump(news_input, open(Path(data_dir[mode]) / "nltk_token_news.bin", "wb"))
        print("Glove token preprocess finish.")
    else:
        print(f'[{mode}] All preprocessed files exist.')


def prepare_news_graph(cfg, mode='train'):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    nltk_target_path = Path(data_dir[mode]) / "nltk_news_graph.pt"  # 自定义文件名

    reprocess_flag = False
    if nltk_target_path.exists() is False:
        reprocess_flag = True

    if (reprocess_flag == False) and (cfg.reprocess == False):
        print(f"[{mode}] All graphs exist !")
        return

    # -----------------------------------------News Graph------------------------------------------------
    behavior_path = Path(data_dir['train']) / "behaviors.tsv"
    origin_graph_path = Path(data_dir['train']) / "nltk_news_graph.pt"

    news_dict = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))
    nltk_token_news = pickle.load(open(Path(data_dir[mode]) / "nltk_token_news.bin", "rb"))

    # ------------------- Build Graph -------------------------------
    if mode == 'train':
        edge_list, user_set = [], set()  # 定义边的列表和用户集合
        num_line = len(open(behavior_path, encoding='utf-8').readlines())
        with open(behavior_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=num_line, desc=f"[{mode}] Processing behaviors news to News Graph"):
                line = line.strip().split('\t')

                # check duplicate user
                used_id = line[1]  # 取出用户ID
                if used_id in user_set:
                    continue
                else:
                    user_set.add(used_id)  # 将用户id加入集合

                # record cnt & read path
                history = line[3].split()
                if len(history) > 1:
                    long_edge = [news_dict[news_id] for news_id in history]
                    edge_list.append(long_edge)  # 同一个用户之间的历史新闻开始建边，直接将新闻列表添加到edge_list

        # edge count
        node_feat = nltk_token_news
        target_path = nltk_target_path
        num_nodes = len(news_dict) + 1  # number of nodes

        short_edges = []
        for edge in tqdm(edge_list, total=len(edge_list), desc=f"Processing news edge list"):  # 取出edge_list中的每一个列表
            # Trajectory Graph
            if cfg.model.use_graph_type == 0:  # 代码中默认为0
                for i in range(len(edge) - 1):
                    short_edges.append((edge[i], edge[i + 1]))  # 从小到大连接，箭头方向暂时未知
                    # short_edges.append((edge[i + 1], edge[i]))
            elif cfg.model.use_graph_type == 1:
                # Co-occurence Graph        共现图？
                for i in range(len(edge) - 1):  # 构建方式类似于完全图，具体用来干什么暂时未知
                    for j in range(i + 1, len(edge)):
                        short_edges.append((edge[i], edge[j]))
                        short_edges.append((edge[j], edge[i]))
            else:
                assert False, "Wrong"

        edge_weights = Counter(short_edges)  # 统计边的权重，用字典形式存储
        unique_edges = list(edge_weights.keys())

        edge_index = torch.tensor(list(zip(*unique_edges)), dtype=torch.long)
        edge_attr = torch.tensor([edge_weights[edge] for edge in unique_edges], dtype=torch.long)
        # Data是PyTorch Geometric（PyG）中的一个类，用于表示图数据。这类被设计用来存储图的节点特征、边的信息以及其他相关数据。
        #
        # x = torch.from_numpy(node_feat)：这里将节点特征存储在x属性中。 node_feat是一个NumPy数组，它包含了图中每个节点的特征。通过
        # torch.from_numpy(node_feat)，将NumPy数组转换为PyTorch张量，以便与PyTorch一起使用。这里假设node_feat包含了图中每个节点的特征信息。
        #
        # edge_index = edge_index：这里将边的索引信息存储在edge_index属性中。 edge_index是一个PyTorch张量，之前的代码段已经解释过，它包含了图的边信息，每一行代表一条边，每行有两个元素，分别表示边的源节点和目标节点。
        #
        # edge_attr = edge_attr：这里将边的属性信息存储在edge_attr属性中。 edge_attr可能包含了每条边的额外属性信息，例如边的权重或类型。如果没有额外的边属性，可以将此参数省略。
        #
        # num_nodes = num_nodes：这里将图中的节点数量存储在num_nodes属性中。 num_nodes是一个整数，表示图中节点的总数。
        data = Data(x=torch.from_numpy(node_feat),
                    edge_index=edge_index, edge_attr=edge_attr,
                    num_nodes=num_nodes)

        torch.save(data, target_path)
        print(data)
        print(f"[{mode}] Finish News Graph Construction, \nGraph Path: {target_path} \nGraph Info: {data}")

    elif mode in ['test', 'val']:
        origin_graph = torch.load(origin_graph_path)
        edge_index = origin_graph.edge_index
        edge_attr = origin_graph.edge_attr
        node_feat = nltk_token_news

        data = Data(x=torch.from_numpy(node_feat),
                    edge_index=edge_index, edge_attr=edge_attr,
                    num_nodes=len(news_dict) + 1)

        torch.save(data, nltk_target_path)
        print(f"[{mode}] Finish nltk News Graph Construction, \nGraph Path: {nltk_target_path}\nGraph Info: {data}")


def prepare_neighbor_list(cfg, mode='train', target='news'):
    # --------------------------------Neighbors List-------------------------------------------
    print(f"[{mode}] Start to process neighbors list")

    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    neighbor_dict_path = Path(data_dir[mode]) / f"{target}_neighbor_dict.bin"
    weights_dict_path = Path(data_dir[mode]) / f"{target}_weights_dict.bin"

    reprocess_flag = False
    for file_path in [neighbor_dict_path, weights_dict_path]:
        if file_path.exists() is False:
            reprocess_flag = True

    if (reprocess_flag == False) and (cfg.reprocess == False) and (cfg.reprocess_neighbors == False):  # 判断是否需要进行数据处理
        print(f"[{mode}] All {target} Neighbor dict exist !")
        return

    if target == 'news':
        target_graph_path = Path(data_dir[mode]) / "nltk_news_graph.pt"
        target_dict = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))
        graph_data = torch.load(target_graph_path)
    elif target == 'entity':
        target_graph_path = Path(data_dir[mode]) / "entity_graph.pt"
        target_dict = pickle.load(open(Path(data_dir[mode]) / "entity_dict.bin", "rb"))
        graph_data = torch.load(target_graph_path)
    else:
        assert False, f"[{mode}] Wrong target {target} "

    edge_index = graph_data.edge_index
    edge_attr = graph_data.edge_attr

    if cfg.model.directed is False:
        edge_index, edge_attr = to_undirected(edge_index, edge_attr)

    neighbor_dict = collections.defaultdict(list)
    neighbor_weights_dict = collections.defaultdict(list)

    # for each node (except 0)
    for i in range(1, len(target_dict) + 1):
        dst_edges = torch.where(edge_index[1] == i)[0]  # i as dst  返回和第I个新闻有边的横坐标
        neighbor_weights = edge_attr[dst_edges]  # 取出相应的权重
        neighbor_nodes = edge_index[0][dst_edges]  # neighbors as src  取出邻居节点
        sorted_weights, indices = torch.sort(neighbor_weights, descending=True)  # 给权重排序
        neighbor_dict[i] = neighbor_nodes[indices].tolist()
        neighbor_weights_dict[i] = sorted_weights.tolist()  # 给所有的邻接新闻排序，按照权重大小

    pickle.dump(neighbor_dict, open(neighbor_dict_path, "wb"))
    pickle.dump(neighbor_weights_dict, open(weights_dict_path, "wb"))  # 分别存储邻居节点和邻接矩阵
    print(
        f"[{mode}] Finish {target} Neighbor dict \nDict Path: {neighbor_dict_path}, \nWeight Dict: {weights_dict_path}")


def prepare_entity_graph(cfg, mode='train'):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    target_path = Path(data_dir[mode]) / "entity_graph.pt"
    reprocess_flag = False
    if target_path.exists() is False:
        reprocess_flag = True
    if (reprocess_flag == False) and (cfg.reprocess == False) and (cfg.reprocess_neighbors == False):
        print(f"[{mode}] Entity graph exists!")
        return

    entity_dict = pickle.load(open(Path(data_dir[mode]) / "entity_dict.bin", "rb"))
    origin_graph_path = Path(data_dir['train']) / "entity_graph.pt"

    if mode == 'train':
        target_news_graph_path = Path(data_dir[mode]) / "nltk_news_graph.pt"
        news_graph = torch.load(target_news_graph_path)
        print("news_graph,", news_graph)
        entity_indices = news_graph.x[:, -8:-3].numpy()
        print("entity_indices, ", entity_indices.shape)

        entity_edge_index = []
        # -------- Inter-news -----------------
        # for entity_idx in entity_indices:
        #     entity_idx = entity_idx[entity_idx > 0]
        #     edges = list(itertools.combinations(entity_idx, r=2))
        #     entity_edge_index.extend(edges)

        news_edge_src, news_edge_dest = news_graph.edge_index
        edge_weights = news_graph.edge_attr.long().tolist()
        for i in range(news_edge_src.shape[0]):
            src_entities = entity_indices[news_edge_src[i]]
            dest_entities = entity_indices[news_edge_dest[i]]
            src_entities_mask = src_entities > 0
            dest_entities_mask = dest_entities > 0
            src_entities = src_entities[src_entities_mask]
            dest_entities = dest_entities[dest_entities_mask]
            edges = list(itertools.product(src_entities, dest_entities)) * edge_weights[i]
            entity_edge_index.extend(edges)

        edge_weights = Counter(entity_edge_index)
        unique_edges = list(edge_weights.keys())

        edge_index = torch.tensor(list(zip(*unique_edges)), dtype=torch.long)
        edge_attr = torch.tensor([edge_weights[edge] for edge in unique_edges], dtype=torch.long)

        # --- Entity Graph Undirected
        edge_index, edge_attr = to_undirected(edge_index, edge_attr)

        data = Data(x=torch.arange(len(entity_dict) + 1),
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    num_nodes=len(entity_dict) + 1)

        torch.save(data, target_path)
        print(f"[{mode}] Finish Entity Graph Construction, \n Graph Path: {target_path} \nGraph Info: {data}")
    elif mode in ['val', 'test']:
        origin_graph = torch.load(origin_graph_path)
        edge_index = origin_graph.edge_index
        edge_attr = origin_graph.edge_attr

        data = Data(x=torch.arange(len(entity_dict) + 1),
                    edge_index=edge_index, edge_attr=edge_attr,
                    num_nodes=len(entity_dict) + 1)

        torch.save(data, target_path)
        print(f"[{mode}] Finish Entity Graph Construction, \n Graph Path: {target_path} \nGraph Info: {data}")


def prepare_preprocessed_data(cfg):
    prepare_distributed_data(cfg, "train")
    prepare_distributed_data(cfg, "val")

    prepare_preprocess_bin(cfg, "train")
    prepare_preprocess_bin(cfg, "val")
    prepare_preprocess_bin(cfg, "test")

    prepare_news_graph(cfg, 'train')
    prepare_news_graph(cfg, 'val')
    prepare_news_graph(cfg, 'test')

    prepare_neighbor_list(cfg, 'train', 'news')
    prepare_neighbor_list(cfg, 'val', 'news')
    prepare_neighbor_list(cfg, 'test', 'news')

    prepare_entity_graph(cfg, 'train')
    prepare_entity_graph(cfg, 'val')
    prepare_entity_graph(cfg, 'test')

    prepare_neighbor_list(cfg, 'train', 'entity')
    prepare_neighbor_list(cfg, 'val', 'entity')
    prepare_neighbor_list(cfg, 'test', 'entity')

    # Entity vec process
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}
    train_entity_emb_path = Path(data_dir['train']) / "entity_embedding.vec"
    val_entity_emb_path = Path(data_dir['val']) / "entity_embedding.vec"
    test_entity_emb_path = Path(data_dir['test']) / "entity_embedding.vec"

    val_combined_path = Path(data_dir['val']) / "combined_entity_embedding.vec"
    test_combined_path = Path(data_dir['test']) / "combined_entity_embedding.vec"

    os.system("cat " + f"{train_entity_emb_path} {val_entity_emb_path}" + f" > {val_combined_path}")
    os.system("cat " + f"{train_entity_emb_path} {test_entity_emb_path}" + f" > {test_combined_path}")

