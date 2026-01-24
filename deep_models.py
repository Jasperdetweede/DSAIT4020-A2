import torch
import pandas as pd
from torch import nn
from collections import OrderedDict

class DeepJointModel(nn.Module):
	def __init__( self, n_features, hidden_size, l, device="cpu"):
		super().__init__()
		
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

		self.loss_clf = nn.CrossEntropyLoss()
		self.loss_emb = nn.MSELoss()
		self.optim = torch.optim.Adam(self.parameters(), lr=1e-3)
		self.device = device
		self.l = l
		self.to(device)

	def forward( self, X ):
		embedding_pred = self.regressor(X)
		y_pred = self.classifier(embedding_pred)
		return embedding_pred, y_pred

	def backward( self, y_pred, y, embedding_pred, embedding=None ):
		loss = self.loss_clf( y_pred, y )
		if embedding is not None:
			loss += self.l * self.loss_emb( embedding_pred, embedding )
		self.optim.zero_grad()
		loss.backward()
		self.optim.step()

	def fit( self, dataloader, epochs=100 ):
		self.train()
		for i in range( epochs ):
			percentage = i / epochs
			progress = int( 50 * percentage )
			print( "\rTraining:\t" + "#" * ( progress ) + "-" * int( 50 - progress ) + f"\t[{100*percentage:.1f}%]", end="" )
			
			for batch in dataloader:

				# Check if type dataloader has embedding or not
				if len(batch) == 3:
					X, y, embedding = batch
				elif len(batch) == 2:
					X, y = batch
					embedding = None
				else:
					raise ValueError("DataLoader must have 2 or 3 elements per batch.")

				X, y = X.to(self.device), y.to(self.device)
				if embedding is not None:
					embedding = embedding.to(self.device)

				e_pred, y_pred = self.forward(X)
				self.backward(y_pred, y, e_pred, embedding)

	def predict( self, dataloader ):
		self.eval()
		e_preds = []
		y_preds = []
		with torch.no_grad():
			for batch in dataloader:

				# Check if type dataloader has embedding or not
				if len(batch) == 3:
					X, _, _ = batch
				elif len(batch) == 2:
					X, _ = batch
				else:
					raise ValueError("DataLoader must have 2 or 3 elements per batch.")

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

class DeepSplitModel(nn.Module):
	def __init__( self, n_features, hidden_size, device="cpu" ):
		super().__init__()

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

		self.loss_emb = nn.MSELoss()
		self.loss_clf = nn.CrossEntropyLoss()
		self.optim_emb = torch.optim.Adam(self.regressor.parameters(), lr=1e-3)
		self.optim_clf = torch.optim.Adam(self.classifier.parameters(), lr=1e-3)
		self.optim_e2e = torch.optim.Adam(self.parameters(), lr=1e-3)
		self.device = device
		self.to(device)

	def forward( self, X, end_to_end=True ):
		embedding_pred = self.regressor( X )
		if end_to_end:
			y_pred = self.classifier( embedding_pred )
		else:
			embedding_copy = embedding_pred.clone().detach()
			y_pred = self.classifier( embedding_copy )

		return embedding_pred, y_pred
	
	def backward( self, y_pred, y, embedding_pred, embedding ):
		classification_loss = self.loss_clf( y_pred, y )
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

	def fit( self, dataloader, epochs=100 ):
		self.train()
		for i in range( epochs ):
			percentage = i / epochs
			progress = int( 50 * percentage )
			print( "\rTraining:\t" + "#" * ( progress ) + "-" * int( 50 - progress ) + f"\t[{100*percentage:.1f}%]", end="" )
			
			for batch in dataloader:

				# Check if type dataloader has embedding or not
				if len(batch) == 3:
					X, y, embedding = batch
				elif len(batch) == 2:
					X, y = batch
					embedding = None
				else:
					raise ValueError("DataLoader must have 2 or 3 elements per batch.")

				X, y = X.to(self.device), y.to(self.device)
				
				end_to_end = embedding is None
				if not end_to_end:
					embedding = embedding.to(self.device)

				e_pred, y_pred = self.forward( X, end_to_end )
				self.backward( y_pred, y, e_pred, embedding )

	def predict( self, dataloader ):
		self.eval()
		e_preds = []
		y_preds = []
		with torch.no_grad():
			for batch in dataloader:

				# Check if type dataloader has embedding or not
				if len(batch) == 3:
					X, _, _ = batch
				elif len(batch) == 2:
					X, _ = batch
				else:
					raise ValueError("DataLoader must have 2 or 3 elements per batch.")

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
	print("Usage only as a module, provides class JointModel( n_features, hidden_size, l, device=\"cpu\" )")
