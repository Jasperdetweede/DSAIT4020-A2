import torch
import pandas as pd
from torch import nn
from collections import OrderedDict
from shallow_models import JointModel, SplitModel

class DeepJointModel(JointModel):
	"""==============================
	Deeper version of the Joint Model
	=============================="""
	def __init__( self, n_features, hidden_size, l, device="cpu"):
		super().__init__( n_features, hidden_size, l, device )
		
		self.regressor = nn.Sequential(
			OrderedDict([
				( 'linear1', nn.Linear( n_features, 100 ) ),
				( 'activation1', nn.ReLU() ),
				( 'linear2', nn.Linear( 100, 75 ) ),
				( 'activation2', nn.ReLU()),
				( 'linear3', nn.Linear( 75, 50 ) ),
				( 'activation3', nn.ReLU()),
				( 'linear4', nn.Linear( 50, 50 ) ),
				( 'activation4', nn.ReLU()),
				( 'linear5', nn.Linear( 50, 50 ) ),
				( 'activation5', nn.ReLU()),
				( 'linear6', nn.Linear( 50, 25 ) ),
				( 'activation6', nn.ReLU()),
				( 'linear7', nn.Linear( 25, hidden_size ) )
			])
		)

		self.classifier = nn.Sequential(
			OrderedDict([
				( 'linear1', nn.Linear(hidden_size, 10) ),
				( 'activation1', nn.ReLU() ),
				( 'linear2', nn.Linear(10, 10) ),
				( 'activation2', nn.ReLU() ),
				( 'linear3', nn.Linear(10, 10) ),
				( 'activation3', nn.ReLU() ),
				( 'linear4', nn.Linear(10, 4) ),
				( 'activation4', nn.ReLU() ),
				( 'linear5', nn.Linear( 4, 2 ) )
			])
		)
		self.to(self.device)

class DeepSplitModel(SplitModel):
	"""==============================
	Deeper version of the Split Model
	=============================="""
	def __init__( self, n_features, hidden_size, device="cpu" ):
		super().__init__( n_features, hidden_size, device )

		self.regressor = nn.Sequential(
			OrderedDict([
				( 'linear1', nn.Linear( n_features, 100 ) ),
				( 'activation1', nn.ReLU() ),
				( 'linear2', nn.Linear( 100, 75 ) ),
				( 'activation2', nn.ReLU()),
				( 'linear3', nn.Linear( 75, 50 ) ),
				( 'activation3', nn.ReLU()),
				( 'linear4', nn.Linear( 50, 50 ) ),
				( 'activation4', nn.ReLU()),
				( 'linear5', nn.Linear( 50, 50 ) ),
				( 'activation5', nn.ReLU()),
				( 'linear6', nn.Linear( 50, 25 ) ),
				( 'activation6', nn.ReLU()),
				( 'linear7', nn.Linear( 25, hidden_size ) )
			])
		)

		self.classifier = nn.Sequential(
			OrderedDict([
				( 'linear1', nn.Linear(hidden_size, 10) ),
				( 'activation1', nn.ReLU() ),
				( 'linear2', nn.Linear(10, 10) ),
				( 'activation2', nn.ReLU() ),
				( 'linear3', nn.Linear(10, 10) ),
				( 'activation3', nn.ReLU() ),
				( 'linear4', nn.Linear(10, 4) ),
				( 'activation4', nn.ReLU() ),
				( 'linear5', nn.Linear( 4, 2 ) )
			])
		)
		self.to(self.device)

if __name__ == "__main__":
	print("Usage only as a module, provides classes:")
	print("DeepJointModel( n_features, hidden_size, l, device=\"cpu\" )")
	print("DeepSplitModel( n_features, hidden_size, device=\"cpu\" )")
