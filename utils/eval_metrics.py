from __future__ import print_function, absolute_import
import numpy as np
import torch
from utils.re_ranking import re_ranking

#评估函数
'''
index:索引
good_index:正确索引
junk_index:错位索引
'''
def compute_ap_cmc(index, good_index, junk_index):
    ap = 0
    cmc = np.zeros(len(index)) 
    
    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1.0
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        ap = ap + d_recall*precision

    return ap, cmc


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids):
    num_q, num_g = distmat.shape#是两个int变量，num_q:1980,num_g:11310 11310=1980+9330
    index = np.argsort(distmat, axis=1)
    # index = torch.argsort(distmat, dim=1) # from small to large每一行从小到大排列,返回得是每一行对应得下标
    # index = index.numpy()#转化成numpy类型

    num_no_gt = 0 # num of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros(len(g_pids))#构建一个g_pids长度的空数组
    AP = 0

    for i in range(num_q):
        # ground truth index
        query_index = np.argwhere(g_pids==q_pids[i])#argwhere返回非0张量的索引
        camera_index = np.argwhere(g_camids==q_camids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)#setdiff1d的作用是求两个数组的集合差。返回' ar1 '中不在' ar2 '中的唯一值。
        if good_index.size == 0:
            num_no_gt += 1
            continue
        # remove gallery samples that have the same pid and camid with query
        
        junk_index = np.intersect1d(query_index, camera_index)

        ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        if CMC_tmp[0]==1:
            num_r1 += 1
        CMC = CMC + CMC_tmp
        AP += ap_tmp

    # if num_no_gt > 0:
    #     print("{} query imgs do not have groundtruth.".format(num_no_gt))

    CMC = CMC / (num_q - num_no_gt)
    mAP = AP / (num_q - num_no_gt)

    return CMC, mAP



# 引入重排列
def evaluate_reranking(dismat,qf, q_pids, q_camids, gf, g_pids, g_camids, ranks):
    m, n = qf.size(0), gf.size(0)

    #计算余弦相似度
    q_g_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    q_g_dist.addmm_(1, -2, qf, gf.t())#query与gallery的相似度矩阵
    q_q_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m) + \
               torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m).t()
    q_q_dist.addmm_(1, -2, qf, qf.t())#query与query的相似度矩阵
    g_g_dist = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n).t()
    g_g_dist.addmm_(1, -2, gf, gf.t())#gallery与gallery的相似度矩阵

    q_g_dist = q_g_dist.cpu().numpy()
    # dismat=dismat.cpu().numpy()#2023/11/10 改
    
    print("Computing CMC and mAP")
    cmc, mAP = evaluate(dismat, q_pids, g_pids, q_camids, g_camids)
    
    #cmc, mAP = evaluate(q_g_dist, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")

    q_q_dist = q_q_dist.cpu().numpy()
    g_g_dist = g_g_dist.cpu().numpy()
    
    rerank_dis = re_ranking(q_g_dist, q_q_dist, g_g_dist)
    
    
    print("Computing rerank CMC and mAP")
    rerank_cmc, rerank_mAP = evaluate(rerank_dis, q_pids, g_pids, q_camids, g_camids)

    print("rerank Results ----------")
    print("mAP: {:.1%}".format(rerank_mAP))
    print("CMC curve")
    for r in ranks:
        print("rerank Rank-{:<3}: {:.1%}".format(r, rerank_cmc[r - 1]))
    print("------------------")
    # return cmc + [mAP]
    return cmc ,mAP

if __name__ == "__main__":
    a = np.random.rand(3, 2)
    b = np.random.rand(4, 2)
    q_g_dist = np.power(a, 2).sum(1, keepdims=True).repeat(4, axis=1) + \
               np.power(b, 2).sum(1, keepdims=True).repeat(3, axis=1).t()
    q_g_dist = q_g_dist - 2 * a.matmul(b.t())

    a = torch.Tensor(a)
    b = torch.Tensor(b)
    q_g_dist2 = torch.pow(a, 2).sum(dim=1, keepdim=True).expand(3, 4) + \
               torch.pow(b, 2).sum(dim=1, keepdim=True).expand(4, 3).t()
    q_g_dist2.addmm_(1, -2, a, b.t())