import torch
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

		self.loss_clf = nn.CrossEntropyLoss()
		self.loss_reg = nn.MSELoss()
		self.optim = torch.optim.Adam( self.parameters(), lr=1e-3 )
		self.device = device
		self.to(device)
		self.l = l

	def forward( self, X ):
		e_pred = self.regressor(X)
		y_pred = self.classifier(e_pred)
		return e_pred, y_pred

	def backward( self, y_pred, y_true, e_pred, e_true=None ):
		loss = self.loss_clf( y_pred, y_true )
		if e_true is not None:
			loss += self.l * self.loss_reg( e_pred, e_true )
		self.optim.zero_grad()
		loss.backward()
		self.optim.step()

	def fit( self, dataloader, epochs=100 ):
		self.train()
		for i in range( epochs ):
			percentage = i / epochs
			progress = int( 50 * percentage )
			print( "\rTraining:\t" + "#" * ( progress ) + "-" * int( 50 - progress ) + f"\t[{100*percentage:.1f}%]", end="" )
			for X, y, embedding in dataloader:
				X, y = X.to(self.device), y.to(self.device)
				if embedding is not None:
					embedding = embedding.to(self.device)

				e_pred, y_pred = self.forward( X )
				self.backward( y_pred, y, e_pred, embedding )
		print()

	def predict( self, dataloader ):
		self.eval()
		e_preds = []
		y_preds = []
		with torch.no_grad():
			for X, _, _ in dataloader:
				X = X.to(self.device)
				e_pred, y_pred = self.forward( X )
				e_preds.append(e_pred)
				y_preds.append(y_pred)
			
			e_preds = torch.cat(e_preds)
			y_preds = torch.cat(y_preds)
			return e_preds.cpu().numpy(), torch.argmax( y_preds, dim=1 ).cpu().numpy()

	def fit_predict( self, dataloader, epochs=100 ):
		self.fit( dataloader, epochs )
		return self.predict( dataloader )

if __name__ == "__main__":
	print("Usage only as a module, provides class JointModel( n_features, embedding_size, l, device=\"cpu\" )")
