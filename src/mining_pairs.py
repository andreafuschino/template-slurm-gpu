#!/usr/bin/env python


import itertools
from itertools import combinations_with_replacement, combinations
import torch
import numpy as np

def get_pairs(embeddings, labels, distNet, loss_func, device):
  with torch.no_grad():
    distNet.eval()
    labels = labels.cpu().data.numpy()
    all_pairs_indices = np.array(list(combinations_with_replacement(range(len(labels)), 2)))
    #print(all_pairs_indices)

    max=-999
    num_trials=4
    for i in range(0,num_trials):
      idx = np.random.choice(len(all_pairs_indices), 100, replace=True)
      all_pairs_indices=all_pairs_indices[idx]
      #print(negative_pairs)
      positive_pairs = all_pairs_indices[(labels[all_pairs_indices[:, 0]] == labels[all_pairs_indices[:, 1]]).nonzero()]
      negative_pairs = all_pairs_indices[(labels[all_pairs_indices[:, 0]] != labels[all_pairs_indices[:, 1]]).nonzero()]

      #print(len(positive_pairs), len(negative_pairs))

      all_pairs_diff = torch.abs(embeddings[all_pairs_indices[:, 0]] - embeddings[all_pairs_indices[:, 1]])

      all_pairs_diff = all_pairs_diff.to(device)
      dist=distNet(all_pairs_diff)

      loss=loss_func(dist, positive_pairs, negative_pairs,all_pairs_indices,labels, embeddings)
      #print("loss",i," ",loss)

      if loss>max:
        max=loss
        selected_pairs_indices=all_pairs_indices
        selected_negative=negative_pairs
        selected_positive=positive_pairs
        selected_all_pairs_diff=all_pairs_diff
        #print("new max = ",loss)


    return  selected_all_pairs_diff, selected_positive, selected_negative, selected_pairs_indices
