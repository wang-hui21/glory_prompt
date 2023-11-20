# -*- coding: utf-8 -*-
# @Time : 2023/11/13 10:37
# @Author : Wang Hui
# @File : myBertForMaskedLM
# @Project : glory_prompt
from transformers import BertModel, BertTokenizer, BertConfig, AutoTokenizer
from transformers import BertPreTrainedModel
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertEncoder


class CustomizedBertModel(nn.Module):
    def __init__(self, config, cfg):         #初始化，可以在此处传入各种参数
        super(CustomizedBertModel, self).__init__()
        # self.myembedding=myembedding
        self.myEmbeddings = MyBERTEmbedding(config, cfg)  # 使用自定义的嵌入层，初始化嵌入层，此处传入自己定义的各项参数
        self.encoder = BertEncoder(config)                     #只修改嵌入层，编码层不做改动

    def forward(self, input_ids, attention_mask, token_type_ids,Uembedding, Cembedding):
        # print(myembedding)

        embedding_output = self.myEmbeddings(input_ids=input_ids, token_type_ids=token_type_ids,Uembedding=Uembedding,
                            Cembedding=Cembedding)    #调用自定的嵌入层，此处传入的参数，由前向传播函数接受
        encoder_output = self.encoder(embedding_output)
        # embedding_output: 这是嵌入层的输出，通常是BERT模型输入的一部分，是必需的。
        #
        # attention_mask: 用于指定哪些位置的输入应该被忽略。通常，这是一个二进制掩码，其中1表示应该关注的位置，0
        # 表示应该忽略的位置。这是一个可选参数，默认为None，通常在文本序列长度不同的情况下使用。
        #
        # head_mask: 用于指定哪些注意力头应该被忽略。这是一个可选参数，默认为None。
        #
        # encoder_hidden_states: 这是来自其他编码器的隐藏状态，通常不需要提供，默认为None。
        #
        # encoder_attention_mask: 这是来自其他编码器的注意力掩码，通常不需要提供，默认为None。
        #
        # past_key_values: 用于存储过去的键值对，以便在生成时重用。这在生成任务中有用，通常不需要提供，默认为None。
        #
        # use_cache: 用于指示是否要使用缓存。通常在生成任务中使用，也是可选参数，默认为None。
        #
        # output_attentions: 控制是否输出注意力权重。这是一个布尔值，通常用于调试或特定任务的分析，默认为False。
        #
        # output_hidden_states: 控制是否输出隐藏状态。这是一个布尔值，通常用于调试或特定任务的分析，默认为False。
        #
        # return_dict: 控制是否返回输出作为字典。如果设置为True，则返回一个字典，其中包含各种输出，如last_hidden_state，pooler_output等。如果设置为False，则以元组形式返回输出。这是一个可选参数，默认为True。
        #
        # 哪些参数是必要的取决于你的具体任务和应用。通常，最重要的参数是embedding_output和attention_mask，其他参数通常在特定任务中使用或不使用，取决于需要。
        return encoder_output[0]


# 自定义嵌入层
class MyBERTEmbedding(nn.Module):
    def __init__(self, config, cfg):
        super(MyBERTEmbedding, self).__init__()
        self.word_embeddings = nn.Embedding(cfg.token.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(cfg.token.vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids,token_type_ids, Uembedding,Cembedding):
        words_embeddings = self.word_embeddings(input_ids)
        # words_embeddings[0][2]=Myembedding           #此处修改需要改变的embedding层，在此处修改embedding层
        position_embeddings = self.position_embeddings(torch.arange(input_ids.size(1), device=input_ids.device))
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class CustomBertForMaskedLM(BertPreTrainedModel):

    def __init__(self, config, cfg):
        super().__init__(config)
        self.bert = CustomizedBertModel(config, cfg)
        self.cls = BertOnlyMLMHead(config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,  Uembedding=None,
                            Cembedding=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            Uembedding=Uembedding,
                            Cembedding=Cembedding
                            )

        sequence_output = outputs
        prediction_scores = self.cls(sequence_output)
        return prediction_scores