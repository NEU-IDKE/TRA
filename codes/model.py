#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math

import numpy as np
import h5py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, datapath, alpha,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.alpha = alpha

        pretrain = np.load(os.path.join(datapath, 'entity_embedding.npy'))
        pretrain = torch.tensor(pretrain)
        self.ent_embed_tp = torch.nn.Embedding(self.nentity, pretrain.shape[1]).from_pretrained(pretrain)
        self.ent_embed_tp.requires_grad_(False)
        
        tp_rel = np.load(os.path.join(datapath, 'tp_rel.npy'))
        self.tp_rel = nn.Parameter(
            torch.Tensor(tp_rel),
            requires_grad=False
        )

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        self.ent_embed_struct = nn.Parameter(torch.zeros(nentity, self.hidden_dim))
        nn.init.uniform_(
            tensor=self.ent_embed_struct, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        self.relation_dim = hidden_dim + pretrain.shape[1]

        if model_name == 'RotatE':
            self.relation_dim //= 2
        elif model_name == 'CompoundE':
            self.relation_dim *= 3

        self.rel_emb = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.rel_emb, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )

        # self.tpfc = nn.Linear(pretrain.shape[1], pretrain.shape[1])

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'CompoundE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')


    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head_struct = torch.index_select(
                self.ent_embed_struct, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            head_tp = self.ent_embed_tp(sample[:, 0]).unsqueeze(1)

            relation = torch.index_select(
                self.rel_emb, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)

            tail_struct = torch.index_select(
                self.ent_embed_struct, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            tail_tp = self.ent_embed_tp(sample[:, 2]).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head_tp = self.ent_embed_tp(head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            head_struct = torch.index_select(
                self.ent_embed_struct, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.rel_emb, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail_tp = self.ent_embed_tp(tail_part[:, 2]).unsqueeze(1)
            tail_struct = torch.index_select(
                self.ent_embed_struct, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
           
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head_tp = self.ent_embed_tp(head_part[:, 0]).unsqueeze(1)
            head_struct = torch.index_select(
                self.ent_embed_struct, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.rel_emb,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail_tp = self.ent_embed_tp(tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
            tail_struct = torch.index_select(
                self.ent_embed_struct, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'RotatE': self.RotatE,
            'CompoundE': self.CompoundE
        }

        score_tps = []
        for i in range(0, self.tp_rel.shape[0]):
            tp_rel = self.tp_rel[i, :].unsqueeze(0).unsqueeze(1)
            score_tps.append(self.Topology_score(head_tp, tp_rel, tail_tp, mode).unsqueeze(-1))

        score_tp = torch.concat(score_tps, dim=-1)
        
        score_tp = score_tp.min(dim=-1).values

        head_emb = torch.concat([head_struct, head_tp], dim=-1)
        rel_emb = relation
        tail_emb = torch.concat([tail_struct, tail_tp], dim=-1)
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head_emb, rel_emb, tail_emb, mode) 
            score = score - self.alpha * score_tp
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score
    

    def Topology_score(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = torch.norm(score, p=1, dim=2)
        return score
    
    def CompoundE(self, head, relation, tail, mode):
        tail_scale, tail_translate, theta = torch.chunk(relation, 3, dim=2)
        theta, _ = torch.chunk(theta, 2, dim=2)

        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)
        
        pi = 3.14159265358979323846

        theta = theta/(self.embedding_range.item()/pi)

        re_rotation = torch.cos(theta)
        im_rotation = torch.sin(theta)

        re_rotation = re_rotation.unsqueeze(-1)
        im_rotation = im_rotation.unsqueeze(-1)

        tail = tail.view((tail.shape[0], tail.shape[1], -1, 2))

        tail_r = torch.cat((re_rotation * tail[:, :, :, 0:1], im_rotation * tail[:, :, :, 0:1]), dim=-1)
        tail_r += torch.cat((-im_rotation * tail[:, :, :, 1:], re_rotation * tail[:, :, :, 1:]), dim=-1)

        tail_r = tail_r.view((tail_r.shape[0], tail_r.shape[1], -1))

        tail_r += tail_translate
        tail_r *= tail_scale

        score = head - tail_r
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score


    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score
    
    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = -1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = -1)

        positive_score = model(positive_sample)
        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
