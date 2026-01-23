import torch
import pandas as pd
from torch import nn
from collections import OrderedDict

class JointModel(nn.Module):
	def __init__( self, n_features, hidden_size, l, device="cpu"):
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

		self.loss_clf = nn.CrossEntropyLoss()
		self.loss_reg = nn.MSELoss()
		self.optim = torch.optim.Adam( self.parameters(), lr=1e-3 )
		self.device = device
		self.to(device)
		self.l = l

	def to_tensor( self, X, dtype=torch.float ):
		if isinstance( X, pd.DataFrame ):
			X = X.values
		return torch.tensor( X, dtype=dtype ).to(self.device)

	def forward( self, X ):
		e_pred = self.regressor(X)
		y_pred = self.classifier(e_pred)
		return e_pred, y_pred

	def backward( self, y_pred, y_true, e_pred, e_true=None ):
		loss = self.loss_clf( y_pred, y_true.view(-1) )
		if e_true is not None:
			loss += self.l * self.loss_reg( e_pred, e_true )
		self.optim.zero_grad()
		loss.backward()
		self.optim.step()

	def fit( self, X, y, embedding=None, epochs=100 ):
		X = self.to_tensor( X, dtype=torch.float )
		y = self.to_tensor( y, dtype=torch.long )
		if embedding is not None:
			embedding = self.to_tensor( embedding, dtype=torch.float )

		self.train()
		for _ in range( epochs ):
			e_pred, y_pred = self.forward( X )
			self.backward( y_pred, y, e_pred, embedding )

	def predict( self, X ):
		X = self.to_tensor( X, dtype=torch.float )
		self.eval()
		with torch.no_grad():
			e_pred, y_pred = self.forward( X )
			return e_pred.cpu().numpy(), torch.argmax( y_pred, dim=1 ).cpu().numpy()

	def fit_predict( self, X, y, embedding=None, epochs=100 ):
		self.fit( X, y, embedding, epochs )
		return self.predict( X )

if __name__ == "__main__":
	print("Usage only as a module, provides class JointModel( n_features, embedding_size, l, device=\"cpu\" )")
