import os
from collections import defaultdict

import hnswlib
import numpy as np
import torch
from tqdm import tqdm

from rqvae_embed.rqvae_clip import RQVAE_EMBED_CLIP

prefix = '/mnt/workspace/xianming/DSI_generation/rqvae/'

gallery_path_dict = {
    'i2v': '/mnt/workspace/xianming/DSI_generation/rqvae/train_data/i2v_d512/5kquery_50wgallery_i2v_d512.npz',
}


def read(gallery_path):
    # query
    with open(prefix + 'eval_data/query_gulnew5k_0930.txt', 'r') as f1:
        querys = f1.readlines()
        querys = [q.strip().split('||$$||') for q in querys]

    # gallery
    if gallery_path.endswith('.npz'):
        data = np.load(gallery_path)
        g_ids = data['ids']
        g_embs = data['embeds']
        print(g_ids.shape, g_embs.shape)
    else:
        g_ids = np.load(f"{gallery_path}_ids.npy")
        g_embs = np.load(f"{gallery_path}_embs.npy").astype(np.float32)
        print(g_ids.shape, g_embs.shape)

    g_id2emb = {}
    for iid, emb in zip(g_ids, g_embs):
        g_id2emb[iid] = emb

    return g_ids, g_embs, g_id2emb, querys


def extract(model, g_embs, g_ids):
    span = 512
    rqvae_output = []
    rq_vae_index = []
    device = next(model.parameters()).device
    for idx in range(0, len(g_embs), span):
        t = torch.from_numpy(g_embs[idx:idx + span])
        index = model.rq_model.get_codes(t.to(device))
        output = model.get_decode_feature(t.to(device))
        # output = model.rq_model.quantizer.embed_code(index)
        output = output.detach().cpu().numpy()
        rqvae_output.append(output)
        rq_vae_index.append(index.cpu().numpy())
    rqvae_output = np.concatenate(rqvae_output, axis=0)
    print('rqvae output shape: ', rqvae_output.shape)
    rqvae_output = rqvae_output / np.linalg.norm(rqvae_output, axis=1, keepdims=True)

    # rqvae_output = g_embs

    rqvae_output = rqvae_output / np.linalg.norm(rqvae_output, axis=1, keepdims=True)
    g_id2emb_vqvae = {}
    for iid, emb in zip(g_ids, rqvae_output):
        g_id2emb_vqvae[iid] = emb

    return rqvae_output, g_id2emb_vqvae


def cal2_hnsw(querys, g_id2emb_rqvae, rqvae_output, g_ids):
    # query2 calc hitrate
    all_num = 0.0
    hit_num = 0.0
    k = 31

    dim = rqvae_output.shape[-1]
    num_elements = 5000000  
    assert num_elements > rqvae_output.shape[0]

    # build HNSW index
    p = hnswlib.Index(space='ip', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=256, M=16)
    p.set_ef(50)  # setting ef
    ids = np.arange(num_elements)
    p.add_items(rqvae_output, g_ids)

    unique_qids = set()
    for _, _, _, session_item_list in querys:
        # for _, session_item_list, _, _ in tqdm(querys):
        unique_qids.update([int(i) for i in session_item_list.split(',') if int(i) in g_id2emb_rqvae])

    unique_qids = list(unique_qids)
    qid_fea = np.array([g_id2emb_rqvae[id] for id in unique_qids])
    retrieve_item_ids_unique = defaultdict(set)

    span = 1000
    for i in range(0, len(unique_qids), span):
        topk_ids, sorted_top30_values = p.knn_query(qid_fea[i:i + span], k=k)

        for rids, j in zip(topk_ids, list(range(i, min(i + span, len(unique_qids))))):
            rids = [iid for iid in rids if iid != unique_qids[j]][:k - 1]  # except itself
            retrieve_item_ids_unique[unique_qids[j]].update(rids)

    all_num, hit_num = 0, 0
    for sess_id, label, target, session_item_list in tqdm(querys[:]):
        # for sess_id, session_item_list, target, label in tqdm(querys):
        if int(target) not in g_id2emb_rqvae:
            continue
        if float(label) == 1.0:
            session_item_list = [int(i) for i in session_item_list.split(',') if int(i) in g_id2emb_rqvae]
            for id in session_item_list:
                if int(target) in retrieve_item_ids_unique[id]:
                    hit_num += 1
                    break
            if session_item_list:
                all_num += 1
    print(hit_num, all_num, hit_num / all_num * 100)
    return hit_num, all_num, hit_num / all_num * 100


def cal2(querys, g_id2emb_rqvae, rqvae_output, g_ids):
    # query2 calc hitrate
    all_num = 0.0
    hit_num = 0.0
    k = 31

    unique_qids = set()
    for _, _, _, session_item_list in tqdm(querys):
        # for _, session_item_list, _, _ in tqdm(querys):
        unique_qids.update([int(i) for i in session_item_list.split(',') if int(i) in g_id2emb_rqvae])

    unique_qids = list(unique_qids)
    qid_fea = np.array([g_id2emb_rqvae[id] for id in unique_qids])
    print('unque qid fea shape: ', qid_fea.shape)
    retrieve_item_ids_unique = defaultdict(set)

    span = 500
    for i in tqdm(range(0, len(unique_qids), span)):
        sim = qid_fea[i:i + span] @ rqvae_output.T
        # top30
        top30_indices = np.argpartition(sim, -k, axis=1)[:, -k:]
        top30_values = np.take_along_axis(sim, top30_indices, axis=1)
        sorted_order = np.argsort(-top30_values, axis=1)
        sorted_top30_indices = np.take_along_axis(top30_indices, sorted_order, axis=1)
        # sorted_top30_values = np.take_along_axis(top30_values, sorted_order, axis=1)
        sorted_item_id_tmp = g_ids[sorted_top30_indices]
        for rids, j in zip(sorted_item_id_tmp, list(range(i, min(i + span, len(unique_qids))))):
            rids = [iid for iid in rids if iid != unique_qids[j]][:k - 1]  # 不召回自己
            retrieve_item_ids_unique[unique_qids[j]].update(rids)

    all_num, hit_num = 0, 0
    for sess_id, label, target, session_item_list in tqdm(querys[:]):
        # for sess_id, session_item_list, target, label in tqdm(querys):
        if float(label) == 1.0:
            session_item_list = [int(i) for i in session_item_list.split(',') if int(i) in g_id2emb_rqvae]
            for id in session_item_list:
                if int(target) in retrieve_item_ids_unique[id]:
                    hit_num += 1
                    break
            if session_item_list:
                all_num += 1
    print(hit_num, all_num, hit_num / all_num * 100)
    return hit_num, all_num, hit_num / all_num * 100


def calculate_metric2(model, g_ids, g_embs, g_id2emb, querys):
    rqvae_output, g_id2emb_rqvae = extract(model, g_embs, g_ids)
    hitrates = []
    for _ in range(3):
        hit_num, all_num, hitrate = cal2_hnsw(querys, g_id2emb_rqvae, rqvae_output, g_ids)
        hitrates.append(hitrate)
    return hit_num, all_num, np.mean(hitrates)


def test_model(model_path_list, model, all_emb_path=gallery_path_dict['80msideinfo']):
    g_ids, g_embs, g_id2emb, querys = read(all_emb_path)

    best_path, best_hitrate = '', -1.0
    for model_path in model_path_list:
        print(model_path)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict['model'], strict=False)
        try:
            hit, all, hitrate = calculate_metric2(model, g_ids, g_embs, g_id2emb, querys)
        except Exception as e:
            print(e)
            continue

        print(hit, all, hitrate)

        if hitrate > best_hitrate:
            best_path = model_path
            best_hitrate = hitrate
            print('=' * 100)
            print('best: ', best_hitrate, best_path)
            print('=' * 100)
    print('best: ', best_hitrate, best_path)
    return best_hitrate, best_path


def filter_existing_paths(model_path_list):
    existing_paths = []
    for path in model_path_list:
        if os.path.exists(path):
            existing_paths.append(path)
    return existing_paths


if __name__ == '__main__':
    codebook_num = 3
    codebook_size = 8192  # 8192 # [4096, 4096, 32768] # 48
    codebook_dim = 64  # 64
    input_dim = 512

    hps = {
        "bottleneck_type": "rq",
        "embed_dim": codebook_dim,
        "n_embed": codebook_size,
        "latent_shape": [8, 8, codebook_dim],
        "code_shape": [8, 8, codebook_num],
        "shared_codebook": False,
        "decay": 0.99,
        "restart_unused_codes": True,
        "loss_type": "cosine",
        "latent_loss_weight": 0.15,
        "masked_dropout": 0.0,
        "use_padding_idx": False,
        "VQ_ema": False,
        "do_bn": True,
        'rotation_trick': False
    }
    ddconfig = {
        "double_z": False,
        "z_channels": codebook_dim,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 1, 2, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [8],
        "dropout": 0.00,
        "input_dim": input_dim
    }
    model = RQVAE_EMBED_CLIP(hps, ddconfig=ddconfig, checkpointing=True)
    model = model.cuda()
    model = model.eval()

    # test data

    model_path_list = [
        prefix + f"ckpts/RQ_gmesideinfo_d512_3_8192_wd1e-4_initK_tongwosh_bs8192_lr0.008_ep150_20250826_214839/checkpoint-{i}.pth"
        for i in list(range(149, 0, -10)) + [0]]
    model_path_list = filter_existing_paths(model_path_list)

    test_model(model_path_list, model, all_emb_path=gallery_path_dict['i2v'])
