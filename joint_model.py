import torch
from torch import nn
from collections import OrderedDict
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, classification_report, mean_squared_error

class JointModel(nn.Module):
	def __init__( self, n_features, embedding_size, l, device="cpu"):
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

		self.loss_clf = nn.CrossEntropyLoss()
		self.loss_emb = nn.CrossEntropyLoss()
		self.optim = torch.optim.Adam( mlp.parameters(), lr=1e-3 )
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

	def fit( self, X, y, embedding, epochs=100 ):
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

	def fit_predict( self, X, y, embedding, epochs=100 ):
		self.fit( X, y, embedding, epochs )
		return self.predict( X, False )

def train_joint_model( X_train, X_test, y_train, y_test, y_embed_train, y_embed_test, device="cpu" ):
	n_features, n_samples = X_train.size()
	embedding_size, _ = y_embed_train.size()

	model = JointModel( n_features=n_features, embedding_size=embedding_size, l=0.3 )
	model.fit( X_train, y_train, y_embed_train )
	y_embed_pred_train, y_pred_train = model.predict( X_train )
	y_embed_pred_test, y_pred_test = model.predict( X_test )

	print("\n\n", "#"*40, "Joint MLP ", "#"*40)
	print("F1 score:", f1_score(y_test, y_pred_test))
	print(classification_report(y_test, y_pred))
	print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
	print("Usage only as a module, provides class JointModel( n_features, embedding_size, l, device=\"cpu\" )")