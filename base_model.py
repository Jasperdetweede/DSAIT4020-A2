from abc import ABC, abstractmethod
import torch
import pandas as pd
from torch import nn
from collections import OrderedDict

class BaseModel(nn.Module, ABC):
	"""=======================================
	Base class to reduce boilerplate in models
	======================================="""
	def __init__( self, n_features, hidden_size, device="cpu" ):
		super().__init__()
		
		self.regressor = nn.Sequential(
			OrderedDict([
				( 'linear1', nn.Linear( n_features, 100 ) ),
				( 'activation1', nn.ReLU() ),
				( 'linear2', nn.Linear( 100, 25 ) ),
				( 'activation2', nn.ReLU()),
				( 'linear3', nn.Linear( 25, hidden_size ) )
			])
		)

		self.classifier = nn.Sequential(
			OrderedDict([
				( 'linear1', nn.Linear(hidden_size, 10) ),
				( 'activation1', nn.ReLU() ),
				#( 'linear2', nn.Linear(10, 10) ),
				#( 'activation2', nn.ReLU() ),
				#( 'linear3', nn.Linear(10, 10) ),
				#( 'activation3', nn.ReLU() ),
				( 'linear4', nn.Linear(10, 4) ),
				( 'activation4', nn.ReLU() ),
				( 'linear5', nn.Linear( 4, 2 ) )
			])
		)

		self.loss_clf = nn.CrossEntropyLoss()
		self.loss_reg = nn.MSELoss()
		self.device = device
	
	def forward( self, X, end_to_end=True ):
		e_pred = self.regressor( X )
		y_pred = self.classifier( e_pred )

		return e_pred, y_pred
	
	@abstractmethod
	def backward(self, y_pred, y_true, e_pred, e_true=None):
		pass
	
	def _get_batch(self, batch):
		# Check if type dataloader has embedding or not
		if len(batch) == 3:
			X, y, embedding = batch
		elif len(batch) == 2:
			X, y = batch
			embedding = None
		else:
			raise ValueError("DataLoader must have 2 or 3 elements per batch.")
		return X, y, embedding

	def fit( self, dataloader, epochs=100, early_stop_epochs=None ):
		self.train()
		last_loss, epochs_without_improvement = 1e10,  0
		for i in range( epochs ):
			percentage = i / epochs
			progress = int( 50 * percentage )
			print( "\rTraining:\t" + "#" * ( progress ) + "-" * int( 50 - progress ) + f"\t[{100*percentage:.1f}%]", end="" )
			
			loss = 0
			for batch in dataloader:
				# Check if type dataloader has embedding or not
				X, y, embedding = self._get_batch(batch)
				X, y = X.to(self.device), y.to(self.device)
				if embedding is not None:
					embedding = embedding.to(self.device)

				e_pred, y_pred = self.forward(X, end_to_end=embedding is None)
				loss += self.backward(y_pred, y, e_pred, embedding)
			loss /= len(dataloader)
		
			if early_stop_epochs is not None:
				if loss < last_loss:
					last_loss = loss
					epochs_without_improvement = 0
				else:
					epochs_without_improvement += 1
				if epochs_without_improvement >= early_stop_epochs:
					break
		print( "\tTraining:\t" + "#" * 50 + "\t[100%]" )

	def predict( self, dataloader ):
		self.eval()
		e_preds = []
		y_preds = []

		with torch.no_grad():
			for batch in dataloader:
				X, y, embedding = self._get_batch(batch)
				X = X.to(self.device)
				e_pred, y_pred = self.forward( X )
				e_preds.append(e_pred)
				y_preds.append(y_pred)
			
			e_preds = torch.cat(e_preds)
			y_preds = torch.cat(y_preds)
			return e_preds.cpu().numpy(), torch.argmax( y_preds, dim=1 ).cpu().numpy()

	def fit_predict( self, dataloader, epochs=100, early_stop_epochs=None ):
		self.fit( dataloader, epochs, early_stop_epochs )
		return self.predict( dataloader )

if __name__ == "__main__":
	print("Usage only as a module, provides class BaseModel( device=\"cpu\" )")
