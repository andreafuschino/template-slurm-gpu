#!/usr/bin/env python

import torch
import torch.nn as nn


class DistanceLoss(nn.Module):
    def __init__(self, margin, distance):
      super(DistanceLoss, self).__init__()
      self.margin = margin
      self.pdist = nn.PairwiseDistance(p=distance)

    def forward(self, dist, positive_pairs, negative_pairs, all_pairs_indices, labels, embeddings):
      
      L2dist_pos=self.pdist(embeddings[positive_pairs[:, 0]],embeddings[positive_pairs[:, 1]])
      L2dist_neg=self.pdist(embeddings[negative_pairs[:, 0]],embeddings[negative_pairs[:, 1]])

      dist_pos=dist[(labels[all_pairs_indices[:, 0]] == labels[all_pairs_indices[:, 1]]).nonzero()].squeeze()
      dist_neg=dist[(labels[all_pairs_indices[:, 0]] != labels[all_pairs_indices[:, 1]]).nonzero()].squeeze()

      positive_contr = (dist_pos - L2dist_pos + self.margin).double()
      positive_loss =  torch.where(positive_contr> 0.0, positive_contr, 0.0) 

      negative_contr = (L2dist_neg - dist_neg +self.margin).double()
      negative_loss =  torch.where(negative_contr> 0.0, negative_contr, 0.0) 

      '''print("-------------------")
      print("L2")
      print(L2dist_pos)
      print(L2dist_neg)
      print("-------------------")
      print("Custom dist")
      print(dist_pos)
      print(dist_neg)
      print("-------------------")
      print("pos/neg contrib")
      print(positive_contr)
      print(negative_contr)
      print("-------------------")
      print("pos/neg loss")
      print(torch.sum(positive_loss))
      print(torch.sum(negative_loss))'''
      

      loss=torch.sum(positive_loss)+torch.sum(negative_loss)

      return loss
