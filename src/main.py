import os.path
import shutil
from datetime import datetime
from pathlib import Path
import time
import hydra
import wandb
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.cuda import amp
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

from dataload.data_load import load_data
from dataload.data_preprocess import prepare_preprocessed_data
from src.utils.utils import evaluate
from utils.metrics import *
from utils.common import *
from transformers import BertTokenizer
from transformers import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
### custom your wandb setting here ###
# os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "offline"
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '23342'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
def cleanup():
    dist.destroy_process_group()
def train(model, optimizer,  dataloader, local_rank,world_size,cfg):
    model.train()
    torch.set_grad_enabled(True)

    mean_loss = torch.zeros(2).to(local_rank)
    acc_cnt = torch.zeros(2).to(local_rank)
    acc_cnt_pos = torch.zeros(2).to(local_rank)
    for cnt, (subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask, batch_enc, batch_attn, batch_labs, labels) \
            in enumerate(tqdm(dataloader,
                              total=int(cfg.num_epochs * (cfg.dataset.pos_count // cfg.batch_size + 1)),
                              desc=f"[{local_rank}] Training"), start=1):
        subgraph = subgraph.to(local_rank, non_blocking=True)
        mapping_idx = mapping_idx.to(local_rank, non_blocking=True)
        candidate_news = candidate_news.to(local_rank, non_blocking=True)
        labels = labels.to(local_rank, non_blocking=True)
        candidate_entity = candidate_entity.to(local_rank, non_blocking=True)
        entity_mask = entity_mask.to(local_rank, non_blocking=True)
        batch_enc = batch_enc.to(local_rank, non_blocking=True)
        batch_attn = batch_attn.to(local_rank, non_blocking=True)
        target = target.to(local_rank, non_blocking=True)



        with amp.autocast():
            loss, scores = model(subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask, batch_enc, batch_attn, target, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("loss: ", loss.item())
        print("score: ", scores[10])
        mean_loss[0] += loss.item()
        mean_loss[1] += 1

        predict = torch.argmax(scores.detach(), dim=1)
        num_correct = (predict == batch_labs).sum()
        acc_cnt[0] += num_correct
        acc_cnt[1] += predict.size(0)

        positive_idx = torch.where(batch_labs == 1)[0]
        num_correct_pos = (predict[positive_idx] == batch_labs[positive_idx]).sum()
        acc_cnt_pos[0] += num_correct_pos
        acc_cnt_pos[1] += positive_idx.size(0)

    dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(acc_cnt, op=dist.ReduceOp.SUM)
    dist.all_reduce(acc_cnt_pos, op=dist.ReduceOp.SUM)

    loss_epoch = mean_loss[0] / mean_loss[1]
    acc = acc_cnt[0] / acc_cnt[1]
    acc_pos = acc_cnt_pos[0] / acc_cnt_pos[1]
    pos_ratio = acc_cnt_pos[1] / acc_cnt[1]

    return loss_epoch.item(), acc.item(), acc_pos.item(), pos_ratio.item()


def val(model, local_rank, cfg):
    model.eval()
    tokenizer, conti_tokens1, conti_tokens2 = load_tokenizer(cfg)
    conti_tokens = [conti_tokens1, conti_tokens2]
    dataloader = load_data(cfg, mode='val', model=model, local_rank=local_rank, tokenizer=tokenizer, conti_tokens=conti_tokens)
    tasks = []

    with torch.no_grad():
        for cnt, (subgraph, mappings, clicked_entity, candidate_input, candidate_entity, entity_mask, batch_enc, batch_attn, target, labels) \
                in enumerate(tqdm(dataloader,
                                  total=int(cfg.dataset.val_len / cfg.gpu_num ),
                                  desc=f"[{local_rank}] Validating")):
            candidate_emb = torch.FloatTensor(np.array(candidate_input)).to(local_rank, non_blocking=True)
            candidate_entity = candidate_entity.to(local_rank, non_blocking=True)
            entity_mask = entity_mask.to(local_rank, non_blocking=True)
            clicked_entity = clicked_entity.to(local_rank, non_blocking=True)

            batch_enc = batch_enc.to(local_rank, non_blocking=True)
            batch_attn = batch_attn.to(local_rank, non_blocking=True)
            target = target.to(local_rank, non_blocking=True)

            scores = model.module.validation_process(subgraph, mappings, clicked_entity, candidate_emb, candidate_entity, entity_mask)
            
            tasks.append((labels.tolist(), scores))

    with mp.Pool(processes=cfg.num_workers) as pool:
        results = pool.map(cal_metric, tasks)
    val_auc, val_mrr, val_ndcg5, val_ndcg10 = np.array(results).T

    # barrier
    torch.distributed.barrier()

    reduced_auc = reduce_mean(torch.tensor(np.nanmean(val_auc)).float().to(local_rank), cfg.gpu_num)
    reduced_mrr = reduce_mean(torch.tensor(np.nanmean(val_mrr)).float().to(local_rank), cfg.gpu_num)
    reduced_ndcg5 = reduce_mean(torch.tensor(np.nanmean(val_ndcg5)).float().to(local_rank), cfg.gpu_num)
    reduced_ndcg10 = reduce_mean(torch.tensor(np.nanmean(val_ndcg10)).float().to(local_rank), cfg.gpu_num)

    res = {
        "auc": reduced_auc.item(),
        "mrr": reduced_mrr.item(),
        "ndcg5": reduced_ndcg5.item(),
        "ndcg10": reduced_ndcg10.item(),
    }
    
    return res

def load_tokenizer(cfg):
    tokenizer = BertTokenizer.from_pretrained(cfg.token.bertmodel)
    conti_tokens1 = []
    for i in range(cfg.token.num_conti1):
        conti_tokens1.append('[P' + str(i + 1) + ']')
    conti_tokens2 = []
    for i in range(cfg.token.num_conti2):
        conti_tokens2.append('[Q' + str(i + 1) + ']')

    new_tokens = ['[NSEP]']
    tokenizer.add_tokens(new_tokens)

    conti_tokens = conti_tokens1 + conti_tokens2
    tokenizer.add_tokens(conti_tokens)

    new_vocab_size = len(tokenizer)
    cfg.token.vocab_size = new_vocab_size

    return tokenizer, conti_tokens1, conti_tokens2
def main_worker(local_rank, world_size, cfg):
    # -----------------------------------------Environment Initial
    seed_everything(cfg.seed)   #设置随机数种子
    setup(local_rank, world_size)

    # -----------------------------------------Dataset & Model Load
    num_training_steps = int(cfg.num_epochs * cfg.dataset.pos_count / (cfg.batch_size * cfg.accumulation_steps))
    num_warmup_steps = int(num_training_steps * cfg.warmup_ratio + 1)

    tokenizer, conti_tokens1, conti_tokens2 = load_tokenizer(cfg)
    conti_tokens = [conti_tokens1, conti_tokens2]

    train_dataloader = load_data(cfg, mode='train', local_rank=local_rank, tokenizer=tokenizer, conti_tokens=conti_tokens)   #加载训练数据
    val_dataloader = load_data(cfg, mode='val',local_rank=local_rank, tokenizer=tokenizer, conti_tokens=conti_tokens)
    model = load_model(cfg, tokenizer).to(local_rank)
    net = DDP(model, device_ids=[local_rank])
    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)

    # AdamW
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': cfg.train.wd},
        {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=cfg.train.lr)

    # lr_lambda = lambda step: 1.0 if step > num_warmup_steps else step / num_warmup_steps
    # scheduler = LambdaLR(optimizer, lr_lambda)
    
    # ------------------------------------------Load Checkpoint & optimizer

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    metrics = ['auc', 'mrr', 'ndcg5', 'ndcg10']
    best_val_result = {}
    best_val_epoch = {}
    for m in metrics:
        best_val_result[m] = 0.0
        best_val_epoch[m] = 0

    for epoch in range(cfg.num_epochs):
        # #################################  train  ###################################
        st_tra = time.time()
        if local_rank == 0:
            print('--------------------------------------------------------------------')
            print('start training: ', datetime.now())
            print('Epoch: ', epoch)
            print('lr:', optimizer.state_dict()['param_groups'][0]['lr'])
        loss, acc_tra, acc_pos_tra, pos_ratio_tra = \
            train(net, optimizer, train_dataloader, local_rank, world_size, cfg)
        # scheduler.step()

        end_tra = time.time()
        train_spend = (end_tra - st_tra) / 3600
        if local_rank == 0:
            print("Train Loss: %0.4f" % loss)
            print("Train ACC: %0.4f\tACC-Positive: %0.4f\tPositiveRatio: %0.4f\t[%0.2f]" %
                  (acc_tra, acc_pos_tra, pos_ratio_tra, train_spend))
            if cfg.train.model_save:
                file = cfg.path.ckp_dir + '/Epoch-' + str(epoch) + '.pt'
                print('save file', file)
                torch.save(net.module.state_dict(), file)

        # #################################  val  ###################################
        st_val = time.time()
        val_scores, acc_val, acc_pos_val, pos_ratio_val, val_impids, val_labels = \
            eval(net, local_rank, world_size, val_dataloader)
        impressions = {}  # {1: {'score': [], 'lab': []}}
        for i in range(world_size):
            scores, imp_id, labs = val_scores[i], val_impids[i], val_labels[i]
            assert scores.size() == imp_id.size() == labs.size()
            scores = scores.cpu().numpy().tolist()
            imp_id = imp_id.cpu().numpy().tolist()
            labs = labs.cpu().numpy().tolist()
            for j in range(len(scores)):
                sco, imp, lab = scores[j], imp_id[j], labs[j]
                if imp not in impressions:
                    impressions[imp] = {'score': [], 'lab': []}
                    impressions[imp]['score'].append(sco)
                    impressions[imp]['lab'].append(lab)
                else:
                    impressions[imp]['score'].append(sco)
                    impressions[imp]['lab'].append(lab)

        predicts, truths = [], []
        for imp in impressions:
            sims, labs = impressions[imp]['score'], impressions[imp]['lab']
            sl_zip = sorted(zip(sims, labs), key=lambda x: x[0], reverse=True)
            sort_sims, sort_labs = zip(*sl_zip)
            predicts.append(list(range(1, len(sort_labs) + 1, 1)))
            truths.append(sort_labs)

        auc_val, mrr_val, ndcg5_val, ndcg10_val = evaluate(predicts, truths)
        if auc_val > best_val_result['auc']:
            best_val_result['auc'] = auc_val
            best_val_epoch['auc'] = epoch
        if mrr_val > best_val_result['mrr']:
            best_val_result['mrr'] = mrr_val
            best_val_epoch['mrr'] = epoch
        if ndcg5_val > best_val_result['ndcg5']:
            best_val_result['ndcg5'] = ndcg5_val
            best_val_epoch['ndcg5'] = epoch
        if ndcg10_val > best_val_result['ndcg10']:
            best_val_result['ndcg10'] = ndcg10_val
            best_val_epoch['ndcg10'] = epoch
        end_val = time.time()
        val_spend = (end_val - st_val) / 3600

        if local_rank == 0:
            print("Validate: AUC: %0.4f\tMRR: %0.4f\tnDCG@5: %0.4f\tnDCG@10: %0.4f\t[Val-Time: %0.2f]" %
                  (auc_val, mrr_val, ndcg5_val, ndcg10_val, val_spend))
            print('Best Result: AUC: %0.4f \tMRR: %0.4f \tNDCG@5: %0.4f \t NDCG@10: %0.4f' %
                  (best_val_result['auc'], best_val_result['mrr'], best_val_result['ndcg5'], best_val_result['ndcg10']))
            print('Best Epoch: AUC: %d \tMRR: %d \tNDCG@5: %d \t NDCG@10: %d' %
                  (best_val_epoch['auc'], best_val_epoch['mrr'], best_val_epoch['ndcg5'], best_val_epoch['ndcg10']))
        dist.barrier()
    if local_rank == 0:
        best_epochs = [best_val_epoch['auc'], best_val_epoch['mrr'], best_val_epoch['ndcg5'], best_val_epoch['ndcg10']]
        best_epoch = max(set(best_epochs), key=best_epochs.count)
        if cfg.train.model_save:
            old_file = cfg.path.ckp_dir + '/Epoch-' + str(best_epoch) + '.pt'
            # old_file = args.save_dir + '/Epoch-2.pt'
            if not os.path.exists('./temp'):
                os.makedirs('./temp')
            copy_file = './temp' + '/BestModel.pt'
            shutil.copy(old_file, copy_file)
            print('Copy ' + old_file + ' >>> ' + copy_file)

    if local_rank == 0:
        wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
                   project=cfg.logger.exp_name, name=cfg.logger.run_name)
        print(model)

    if local_rank == 0:
        wandb.finish()
    cleanup()

@hydra.main(version_base="1.2", config_path=os.path.join(get_root(), "configs"), config_name="small")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    cfg.gpu_num = torch.cuda.device_count()
    prepare_preprocessed_data(cfg)
    mp.spawn(main_worker, nprocs=cfg.gpu_num, args=(cfg, cfg.gpu_num))    # 这行代码的目的是使用 PyTorch 的分布式训练功能，以并行方式执行名为 main_worker 的函数。
                                                              # 每个工作进程都会执行相同的 main_worker 函数，但可以根据配置中的参数来控制不同的行为


if __name__ == "__main__":
    main()

