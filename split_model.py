import torch
import pandas as pd
from torch import nn
from collections import OrderedDict

class SplitModel(nn.Module):
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
				( 'linear1', nn.Linear(hidden_size, 4) ),
				( 'activation1', nn.ReLU() ),
				( 'linear2', nn.Linear( 4, 2 ) )
			])
		)

		self.loss_emb = nn.MSELoss()
		self.loss_clf = nn.CrossEntropyLoss()
		self.optim_emb = torch.optim.Adam(self.regressor.parameters(), lr=1e-3)
		self.optim_clf = torch.optim.Adam(self.classifier.parameters(), lr=1e-3)
		self.optim_e2e = torch.optim.Adam(self.parameters(), lr=1e-3)
		self.device = device
		self.to(device)

	def to_tensor( self, X, dtype=torch.float ):
		if isinstance( X, pd.DataFrame ):
			X = X.values
		return torch.tensor( X, dtype=dtype ).to(self.device)

	def forward( self, X ):
		embedding_pred = self.regressor( X )
		embedding_copy = embedding_pred.clone().detach()
		y_pred = self.classifier( embedding_copy )
		return embedding_pred, y_pred
	
	def backward( self, y_pred, y, embedding_pred, embedding ):
		classification_loss = self.loss_clf( y_pred, y.view(-1) )
		if embedding is not None:
			embedding_loss = self.loss_emb( embedding_pred, embedding )
			self.optim_emb.zero_grad()
			embedding_loss.backward()
			self.optim_emb.step()
			self.optim_clf.zero_grad()
			classification_loss.backward()
			self.optim_clf.step()
		else:
			self.optim_e2e.zero_grad()
			classification_loss.backward()
			self.optim_e2e.step()

	def fit( self, X, y, embedding=None, epochs=100 ):
		X = self.to_tensor( X, dtype=torch.float )
		y = self.to_tensor( y, dtype=torch.long )
		if embedding is not None:
			embedding = self.to_tensor( embedding, dtype=torch.float )
		for _ in range( epochs ):
			embedding_pred, y_pred = self.forward( X )
			self.backward( y_pred, y, embedding_pred, embedding )

	def predict( self, X ):
			X = self.to_tensor( X, dtype=torch.float )
			with torch.no_grad():
				embedding_pred, y_pred = self.forward( X )
				return embedding_pred.cpu().numpy(), torch.argmax( y_pred, dim=1 ).cpu().numpy()

	def fit_predict( self, X, y, embedding=None, epochs=100 ):
		self.fit( X, y, embedding, epochs )
		return self.predict( X )

if __name__ == "__main__":
	print("Usage only as a module, provides class SplitModel( n_features, hidden_size, device=\"cpu\" )")
