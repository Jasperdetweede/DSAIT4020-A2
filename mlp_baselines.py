import torch
import pandas as pd
from torch import nn
from collections import OrderedDict

class BaseMLP(nn.Module):
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
		self.loss_reg = nn.MSELoss()
		self.optim = torch.optim.Adam( self.regressor.parameters(), lr=1e-3 )
		self.device = device
        self.to(self.device)
	
	def forward( self, X ):
		e_pred = self.regressor( X )
		return e_pred
	
	def backward(self, e_pred, e_true):
		loss = self.loss_reg( e_pred, e_true )
		self.optim.zero_grad()
		loss.backward()
		self.optim.step()
		return loss

	def _get_batch(self, batch):
		# Check if type dataloader has embedding or not
		if len(batch) == 3:
			X, y, embedding = batch
		else:
			raise ValueError("DataLoader must have 3 elements per batch.")
		return X, y, embedding

	def fit( self, dataloader, epochs=100, early_stop_epochs=None ):
		self.train()
		last_loss, epochs_without_improvement, early_stop_threshold = 1e10,  0, 5e-2
		for i in range( epochs ):
			percentage = i / epochs
			progress = int( 50 * percentage )
			print( "\rTraining:\t" + "#" * ( progress ) + "-" * int( 50 - progress ) + f"\t[{100*percentage:.1f}%]", end="" )
			
			loss = 0
			for batch in dataloader:
				# Check if type dataloader has embedding or not
				X, y, embedding = self._get_batch(batch)
				X, y, embedding = X.to(self.device), y.to(self.device), embedding.to(self.device)

				e_pred = self.forward(X)
				loss += self.backward(e_pred, embedding)
			loss /= len(dataloader)
		
			if early_stop_epochs is not None:
				if loss < last_loss:
					last_loss = loss
					epochs_without_improvement = 0
				elif last_loss - loss <= early_stop_threshold * last_loss:
					epochs_without_improvement += 1
				else:
					print( "\rTraining:\t" + "#" * ( progress ) + "-" * int( 50 - progress ) + f"\t[{100*percentage:.1f}% - DONE]\n" )
					return
				
				if epochs_without_improvement >= early_stop_epochs:
					print( "\rTraining:\t" + "#" * ( progress ) + "-" * int( 50 - progress ) + f"\t[{100*percentage:.1f}% - DONE]\n" )
					return
		print( "\rTraining:\t" + "#" * 50 + "\t[100.0%]\n" )

	def predict( self, dataloader ):
		self.eval()
		e_preds = []

		with torch.no_grad():
			for batch in dataloader:
				X, y, embedding = self._get_batch(batch)
				X = X.to(self.device)
				e_pred = self.forward( X )
				e_preds.append(e_pred)
			
			e_preds = torch.cat(e_preds)
			return e_preds.cpu().numpy()

	def fit_predict( self, dataloader, epochs=100, early_stop_epochs=None ):
		self.fit( dataloader, epochs, early_stop_epochs )
		return self.predict( dataloader )

if __name__ == "__main__":
	print("Usage only as a module, provides class BaseModel( device=\"cpu\" )")
