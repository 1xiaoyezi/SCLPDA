
import time
import argparse
from clac_metric import get_metrics
import copy
from model import  GCL,GCN
from graph_learners import *
from load_data import *
from loss import Myloss
import os

import pandas as pd
import random
import torch
import numpy as np
start_time = time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = t.device("cuda" if t.cuda.is_available() else "cpu")

class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()


    def setup_seed(self, seed):
        t.manual_seed(seed)
        t.cuda.manual_seed_all(seed)
        t.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)
        dgl.seed(seed)
        dgl.random.seed(seed)


    def loss_cls(self, model,features,train_matrix,regression_crit):
        pir_dis_res=model(features) #1444*1444
        loss1 = nn.MSELoss()
        loss = loss1(pir_dis_res,train_matrix)
        return loss,pir_dis_res


    def loss_gcl(self, model, graph_learner, features, anchor_adj):

        data = np.genfromtxt("D:\\pyworkspace\\MSGCL-main\\m_d.csv", delimiter=',')
        adj1 = torch.FloatTensor(data).to(device)
        learned_adj = graph_learner(adj1)

        # learned_adj = SVD_learner(adj1)
        if args.maskfeat_rate_anchor:
            # mask_v1 ,_ = get_feat_mask(features, args.maskfeat_rate_anchor)
            # features_v1 = features * mask_v1
            features_v1 = features.to(device)
            anchor_adj = anchor_adj.to(device)

        else:
            features_v1 = copy.deepcopy(features)  # 深拷贝
        model.to(device)
        z1, _ = model(features_v1, anchor_adj, 'anchor')

        if args.maskfeat_rate_anchor:
            mask,_ = get_feat_mask(features, args.maskfeat_rate_anchor)
            features_v2 = features *  mask


        else:
            features_v2 = copy.deepcopy(features)

        if not args.sparse:
            learned_adj = symmetrize(learned_adj)
            learned_adj = normalize(learned_adj, 'sym', args.sparse)


        z2, _ = model(features_v2, learned_adj, 'learner')


        loss = model.calc_loss(z1, z2)

        return loss, learned_adj

    def evaluate_adj_by_cls(self, Adj, features, nfeats,args,train_matrix,out1,pir_sim,pir_fea,dis_sim,dis_fea):

        model = GCN(in_channels=nfeats, hidden_channels=args.hidden_dim_cls, out_channels=out1, num_layers=args.nlayers_cls,
                    dropout=args.dropout_cls, dropout_adj=args.dropedge_rate, Adj=Adj, sparse=args.sparse,pir_sim=pir_sim,
                    pir_fea=pir_fea,dis_sim=dis_sim,dis_fea=dis_fea)



        optimizer = t.optim.Adam(model.parameters(), lr=args.lr_cls, weight_decay=args.w_decay_cls)




        if t.cuda.is_available():
            model = model.cuda()
            features = features.cuda()

        for epoch in range(1, args.epochs_cls + 1): #1-201
            model.train()
            regression_crit = Myloss()

            loss,pir_dis_res= self.loss_cls(model, features,train_matrix,regression_crit)
            optimizer.zero_grad()
            loss = loss.requires_grad_()
            loss.backward()
            optimizer.step()

        return pir_dis_res


    def train(self, args,sizes):

        torch.cuda.device(args.gpu)

        if args.gsl_mode == 'structure_refinement':
            features, nfeats, pir_dis_matrix,adj_original,pir_sim,pir_fea,dis_sim,dis_fea = load_data(args)
        elif args.gsl_mode == 'structure_inference':
            features, nfeats, pir_dis_matrix, _ ,pir_sim,pir_fea,dis_sim,dis_fea= load_data(args)

        if args.downstream_task == 'linkpre':
            AUPR_accuracies = []
            AUC_accuracies = []
            ACC_accuracies=[]
        roc_data = []
        prc_data = []
        all_train_indices, all_val_indices = crossval_index(pir_dis_matrix, sizes)
        index=all_val_indices
        # index = crossval_index(pir_dis_matrix, sizes)
        metric = np.zeros((1, 7))
        pre_matrix = np.zeros(pir_dis_matrix.shape)
        print("seed=%d, evaluating pir-disrobe...." % (sizes.seed))

        for k in range(args.k_fold):
            print("------this is %dth cross validation------" % (k + 1))

            train_matrix = np.matrix(pir_dis_matrix, copy=True)
            train_matrix[tuple(np.array(index[k]).T)] = 0
            self.setup_seed(k)

            # Initialize anchor_adj_raw based on gsl_mode
            if args.gsl_mode == 'structure_inference':
                anchor_adj_raw = torch_sparse_eye(features.shape[0]) if args.sparse else torch.eye(features.shape[0])
            elif args.gsl_mode == 'structure_refinement':
                anchor_adj_raw = adj_original if args.sparse else t.from_numpy(adj_original)

            anchor_adj = normalize(anchor_adj_raw, 'sym', args.sparse).float()
            anchor_adj_torch_sparse = copy.deepcopy(anchor_adj) if args.sparse else None
            if args.sparse:
                anchor_adj_torch_sparse = copy.deepcopy(anchor_adj)
                anchor_adj = torch_sparse_to_dgl_graph(anchor_adj)

            # Load data and initialize models
            data = np.genfromtxt("D:\\pyworkspace\\MSGCL-main\\m_d.csv", delimiter=',')
            adj1 = torch.FloatTensor(data).to(device)

            if args.type_learner == 'fgp':
                graph_learner = FGP_learner(features.cpu(), args.k, args.sim_function, 6, args.sparse)
            elif args.type_learner == 'att':
                graph_learner = ATT_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                        args.activation_learner)
            elif args.type_learner == 'SVD':
                graph_learner = SVD_learner(adj1)

            model = GCL(nlayers=args.nlayers, in_dim=nfeats, hidden_dim=args.hidden_dim,
                        emb_dim=args.rep_dim, proj_dim=args.proj_dim,
                        dropout=args.dropout, dropout_adj=args.dropedge_rate, sparse=args.sparse)

            optimizer_cl = t.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
            optimizer_learner = t.optim.Adam(graph_learner.parameters(), lr=args.lr, weight_decay=args.w_decay)

            # Move to GPU if available
            if t.cuda.is_available():
                model, graph_learner = model.cuda(), graph_learner.cuda()
                features = features.cuda()
                train_matrix = t.FloatTensor(train_matrix).cuda()
                if not args.sparse:
                    anchor_adj = anchor_adj.cuda()

            # Training loop
            for epoch in range(1, args.epochs + 1):
                model.train()
                graph_learner.train()
                loss, Adj = self.loss_gcl(model, graph_learner, features, anchor_adj)

                optimizer_cl.zero_grad()
                optimizer_learner.zero_grad()
                loss.backward()
                optimizer_cl.step()
                optimizer_learner.step()


                anchor_adj = anchor_adj * args.tau + Adj.detach() * (1 - args.tau)

                print("Epoch {:05d} | CL Loss {:.4f}".format(epoch, loss.item()))

                if epoch % args.epochs == 0 and args.downstream_task == 'linkpre':
                    model.eval()
                    graph_learner.eval()
                    f_adj = Adj.detach()
                    out1 = train_matrix.shape[1]
                    pir_dis_res = self.evaluate_adj_by_cls(f_adj, features, nfeats, args, train_matrix, out1, pir_sim,
                                                            pir_fea, dis_sim, dis_fea)

            # Predictions and metrics
            predict_y_proba = pir_dis_res.reshape(sizes.pir_size + sizes.dis_size,
                                                   sizes.pir_size + sizes.dis_size).cpu().detach().numpy()
            pre_matrix[tuple(np.array(index[k]).T)] = predict_y_proba[tuple(np.array(index[k]).T)]
            real_score = pir_dis_matrix[tuple(np.array(index[k]).T)]
            metric_tmp = get_metrics(real_score, predict_y_proba[tuple(np.array(index[k]).T)])
            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
            results_file = os.path.join(results_dir, f"fold_{k + 1}_prediction_scores.csv")

            # Save only prediction scores
            pd.DataFrame(predict_y_proba).to_csv(results_file, index=False)


            print(metric_tmp)
            metric += metric_tmp
        if args.downstream_task == 'linkpre' and k == 4:
            print(metric / sizes.k_fold)


def random_index(index_matrix, sizes=None):
    # 调试输出：检查 index_matrix 的形状
    print(f"index_matrix shape: {index_matrix.shape}")

    association_nam = index_matrix.shape[0]  # 行数
    if association_nam == 0:
        raise ValueError("association_nam is 0, indicating that there are no valid indices to work with.")

    random_index = index_matrix.tolist()  # 将索引矩阵转换为列表
    random.seed(sizes.seed)
    random.shuffle(random_index)  # 随机打乱索引
    return random_index  # 返回打乱后的索引列表

def crossval_index(pir_dis_matrix, sizes=None):
    random.seed(sizes.seed)

    pos_index_matrix = torch.nonzero(pir_dis_matrix == 1, as_tuple=False)  # 获取正样本索引
    neg_index_matrix = torch.nonzero(pir_dis_matrix == 0, as_tuple=False)  # 获取负样本索引

    pos_index = random_index(pos_index_matrix, sizes)
    neg_index = random_index(neg_index_matrix, sizes)

    combined_index = pos_index + neg_index

    random.shuffle(combined_index)

    k_folds = 5  # 设定为 5 折
    total_count = len(combined_index)
    fold_size = total_count // k_folds

    folds = []
    for i in range(k_folds):
        start = i * fold_size
        if i == k_folds - 1:
            folds.append(combined_index[start:])
        else:
            folds.append(combined_index[start:start + fold_size])


    all_train_indices = []
    all_val_indices = []

    for i in range(k_folds):
        val_indices = folds[i]
        train_indices = [item for j in range(k_folds) if j != i for item in folds[j]]

        all_train_indices.append(train_indices)
        all_val_indices.append(val_indices)
    return all_train_indices, all_val_indices



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experimental setting

    parser.add_argument('-k_fold', type=int, default=5)
    parser.add_argument('-sparse', type=int, default=0)
    parser.add_argument('-gsl_mode', type=str, default="structure_refinement",
                        choices=['structure_inference', 'structure_refinement'])
    parser.add_argument('-downstream_task', type=str, default='linkpre',
                        choices=['linkpre', 'classification'])
    parser.add_argument('-gpu', type=int, default=0)
    parser.add_argument('-k', type=int, default=30)
    parser.add_argument('-sim_function', type=str, default='cosine', choices=['cosine', 'minkowski'])
    # GCL Module - Framework
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-w_decay', type=float, default=0.3)
    parser.add_argument('-hidden_dim', type=int, default=512)
    parser.add_argument('-rep_dim', type=int, default=128)
    parser.add_argument('-proj_dim', type=int, default=128)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-contrast_batch_size', type=int, default=0)
    parser.add_argument('-nlayers', type=int, default=2)
    parser.add_argument('-activation_learner', type=str, default='relu', choices=["relu", "tanh"])
    parser.add_argument('-dropout_cls', type=float, default=0.0)
    parser.add_argument('-maskfeat_rate_anchor', type=float, default=0.0)
    parser.add_argument('-dropedge_rate', type=float, default=0.0)

    parser.add_argument('-type_learner', type=str, default='SVD', choices=["fgp", "att", "SVD"])
    parser.add_argument('-epochs_cls', type=int, default=200)
    parser.add_argument('-lr_cls', type=float, default=0.001)
    parser.add_argument('-w_decay_cls', type=float, default=0.00004)
    parser.add_argument('-hidden_dim_cls', type=int, default=256)
    parser.add_argument('-nlayers_cls', type=int, default=2)

    # Structure Bootstrapping
    parser.add_argument('-tau', type=float, default=1)
    parser.add_argument('-c', type=int, default=0)

    args = parser.parse_args()

    experiment = Experiment()
    sizes = Sizes()
    experiment.train(args,sizes)

max_allocated_memory = torch.cuda.max_memory_allocated()
print(f"最大已分配内存量: {max_allocated_memory / 1024 ** 2} MB")


elapsed_time = (time.time() - start_time) * 1000
print(f"Elapsed time: {elapsed_time} milliseconds")