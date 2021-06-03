#!/usr/bin/env python


import torch
import torch.nn as nn


# Define block
class BasicBlock(nn.Module):
	def __init__(self, input_dim):
		super(BasicBlock, self).__init__()

		#TODO: fc -> relu
		self.fc_block1 = nn.Sequential( nn.Linear(input_dim, 16),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(),)
		self.fc_block2 = nn.Sequential( nn.Linear(16, 32),
                                    nn.BatchNorm1d(32),)
		self.relu = nn.ReLU()

  #TODO: forward
	def forward(self, x):
		residual = x
		x = self.fc_block1(x)
		x = self.fc_block2(x)
		x = x + residual
		out = self.relu(x)
		return out



class Net_residuals(nn.Module):
	def __init__(self):
		super(Net_residuals, self).__init__()

		##TODO: 1x1 convolution -> relu (to convert feature channel number)
		self.init_block = nn.Sequential( nn.Linear(128, 32),
                                     nn.ReLU(),)

		#TODO: stack 2 BasicBlocks
		self.basic_blocks = nn.ModuleList([BasicBlock(32) for i in range(4)])

		#TODO: 1x1 convolution -> sigmoid (to convert feature channel number)
		self.final_block = nn.Sequential( nn.Linear(32, 1),
                                      nn.ReLU(),)

	def forward(self, x):
		#TODO: forward
		x = self.init_block(x)
		for i, _ in enumerate(self.basic_blocks):
			x = self.basic_blocks[i](x)
		out = self.final_block(x)
		return out




class Net_fc(nn.Module):
  def __init__(self):
    super(Net_fc, self).__init__()

    self.fc_block = nn.Sequential( nn.Linear(128, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 64),
                                   nn.ReLU(),
                                   nn.Linear(64,1),
                                   nn.ReLU(),)
                                  
    
  def forward(self, x):
      #TODO: forward
      x = self.fc_block(x)
      out = x
      return out
