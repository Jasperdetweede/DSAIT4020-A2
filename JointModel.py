import torch
from torch import nn
from collections import OrderedDict

class JointModel(nn.Module):
	def __init__( self, n_features, embedding_size, embedding_loss, clf_loss, l, optimizer, device="cpu"):
		super().__init__()
		
		embedder = nn.Sequential(
			OrderedDict([
				( 'linear1', nn.Linear( n_features, 100 ) ),
				( 'activation1', nn.ReLU() ),
				( 'linear2', nn.Linear( 100, 25 ) ),
				( 'activation2', nn.ReLU()),
				( 'linear3', nn.Linear( 25, embedding_size ) )
			])
		).to(device)
		classifier = nn.Sequential(
				( 'activation3', nn.ReLU()),
				( 'linear4', nn.Linear( embedding_size, 2 ) ),
				( 'softmax', nn.Softmax() )
		)
		
		self.loss_clf =  clf_loss
		self.loss_emb = embedding_loss
		self.optim = optimizer
		self.device = device
		self.l = l

	def to_tensor( self, X, dtype=torch.float ):
		if isinstance( X, pd.DataFrame ):
			X = X.values
		return torch.tensor( X, dtype=dtype ).to(self.device)
	
	def forward( self, X ):
		embedding_pred = self.embedder(X)
		y_pred = self.classifier(embedding_pred)
		return embedding_pred, y_pred

	def backward( self, y_pred, y, embedding_pred, embedding ):
		loss = self.loss_clf( y_pred, y ) + l*self.loss_emb( embedding_pred, embedding )
		self.optim.zero_grad()
		loss.backward()
		self.optim.step()

	def fit( self, X, y, embedding, epochs ):
		X = self.to_tensor( X, dtype=torch.float )
		y = self.to_tensor( y, dtype=torch.long )
		embedding = self.to_tensor( embedding, dtype=torch.float )
		for _ in range( epochs ):
			embedding_pred, y_pred = self.forward( X )
			self.backward( y_pred, y, embedding_pred, embedding )

	def predict( self, X ):
			X = self.to_tensor( X, dtype=torch.float )
			with torch.no_grad():
				embedding_pred, y_pred = self.forward( X )
				return embedding_pred, torch.argmax( y_pred, dim=1 ).to("cpu").numpy()

	def fit_predict( self, X, y, embedding ):
		self.fit( X, y, embedding )
		return self.predict( X, False )


if __name__ == "__main__":
	print("Usage only as a module, provides class JointModel( n_features, embedding_size, embedding_loss, clf_loss, l, optimizer, device=\"cpu\" )")
