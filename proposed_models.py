import torch
import pandas as pd
from joint_model import JointModel
from split_model import SplitModel
from deep_models import DeepJointModel, DeepSplitModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score, classification_report, mean_squared_error

######################
# DATASETS
######################

class CustomDatasetWithE(Dataset):
	def __init__( self, X, y, embedding, device="cpu" ):
		self.device = device
		self.X = self.to_tensor(X, dtype=torch.float32)
		self.y = self.to_tensor(y, dtype=torch.long).view(-1)
		self.embedding = self.to_tensor(embedding, dtype=torch.float32)

	def __len__(self):
		return len(self.y)
	
	def __getitem__(self, idx):
		return self.X[idx], self.y[idx], self.embedding[idx]

	def to_tensor( self, X, dtype=torch.float ):
		if isinstance( X, pd.DataFrame ) or isinstance( X, pd.Series ):
			X = X.values
		return torch.tensor( X, dtype=dtype )

class CustomDatasetNoE(Dataset):
	def __init__( self, X, y, device="cpu" ):
		self.device = device
		self.X = self.to_tensor(X, dtype=torch.float32)
		self.y = self.to_tensor(y, dtype=torch.long).view(-1)

	def __len__(self):
		return len(self.y)
	
	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]

	def to_tensor( self, X, dtype=torch.float ):
		if isinstance( X, pd.DataFrame ) or isinstance( X, pd.Series ):
			X = X.values
		return torch.tensor( X, dtype=dtype )
	
######################
# MAIN TRAIN FUNCTIONS
######################

def train_joint_model( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio, l=1.0, epochs=1000, fine_tune_epochs=50, device="cpu" ):
	n_samples, n_features = X_train.shape
	_, hidden_size = e_train.shape
	
	dataset_train_w_e, dataset_train_no_e, dataset_test = gen_datasets( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio )
	model = JointModel( n_features=n_features, hidden_size=hidden_size, l=l, device=device )
	train_proposal_model( dataset_train_w_e, dataset_train_no_e, dataset_test, model, title="Joint", epochs=epochs, fine_tune_epochs=fine_tune_epochs, device=device )

def train_split_model( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio, epochs=1000, fine_tune_epochs=50, device="cpu" ):
	n_samples, n_features = X_train.shape
	_, hidden_size = e_train.shape

	dataset_train_w_e, dataset_train_no_e, dataset_test = gen_datasets( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio )
	model = SplitModel( n_features=n_features, hidden_size=hidden_size, device=device )
	train_proposal_model( dataset_train_w_e, dataset_train_no_e, dataset_test, model, title="Split", epochs=epochs,fine_tune_epochs=fine_tune_epochs,  device=device )

def train_deep_joint_model( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio, l=1.0, epochs=1000, fine_tune_epochs=50, device="cpu" ):
	n_samples, n_features = X_train.shape
	_, hidden_size = e_train.shape

	dataset_train_w_e, dataset_train_no_e, dataset_test = gen_datasets( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio )
	model = DeepJointModel( n_features=n_features, hidden_size=hidden_size, l=l, device=device )
	train_proposal_model( dataset_train_w_e, dataset_train_no_e, dataset_test, model, title="Deep Joint", epochs=epochs, fine_tune_epochs=fine_tune_epochs, device=device )

def train_deep_split_model( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio, epochs=1000, fine_tune_epochs=50, device="cpu" ):
	n_samples, n_features = X_train.shape
	_, hidden_size = e_train.shape

	dataset_train_w_e, dataset_train_no_e, dataset_test = gen_datasets( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio )
	model = DeepSplitModel( n_features=n_features, hidden_size=hidden_size, device=device )
	train_proposal_model( dataset_train_w_e, dataset_train_no_e, dataset_test, model, title="Deep Split", epochs=epochs, fine_tune_epochs=fine_tune_epochs, device=device )

######################
# HELPER FUNCTIONS
######################

def gen_datasets( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio, device="cpu" ):
	
	n_samples, n_features = X_train.shape
	_, hidden_size = e_train.shape
	n_samples_w_e = int( n_samples * e_kept_ratio )
	
	dataset_train_w_e = CustomDatasetWithE( X_train[:n_samples_w_e], y_train[:n_samples_w_e], e_train[:n_samples_w_e], device=device )
	dataset_train_no_e = CustomDatasetNoE( X_train[n_samples_w_e:], y_train[n_samples_w_e:], device=device )
	dataset_test = CustomDatasetWithE( X_test, y_test, e_test, device=device )

	return dataset_train_w_e, dataset_train_no_e, dataset_test

def train_proposal_model( dataset_train_w_e, dataset_train_no_e, dataset_test, model, title, batch_size=32, epochs=1000, fine_tune_epochs=50, device="cpu" ):
	dataloader_train_w_e = DataLoader( dataset_train_w_e, batch_size=batch_size, shuffle=True )
	dataloader_train_no_e = DataLoader( dataset_train_no_e, batch_size=batch_size, shuffle=True )
	dataloader_test = DataLoader( dataset_test, batch_size=batch_size, shuffle=False )

	model.fit( dataloader_train_w_e, epochs=epochs)
	e_pred_test, y_pred_test = model.predict( dataloader_test )
	present_model_metrics( dataset_test.y, y_pred_test, dataset_test.embedding, e_pred_test, title=f"{title} MLP" )

	model.fit( dataloader_train_no_e, epochs=fine_tune_epochs)
	e_pred_test, y_pred_test = model.predict( dataloader_test )
	present_model_metrics( dataset_test.y, y_pred_test, dataset_test.embedding, e_pred_test, title=f"{title} MLP (Augmented)" )

def present_model_metrics( y_true, y_pred, e_true, e_pred, title ):
	len_eqs = (72 - len(title))//2
	print( "="*80 )
	print( "="*len_eqs + f"    {title}    " + "="*len_eqs )
	print( "="*80 )
	print( "Regression Results:" )
	print( f"MSE:\t{ mean_squared_error(e_true, e_pred) }\n\n" )
	
	print( "Classification Results:" )
	print( f"F1 score: {f1_score(y_true, y_pred)}" )
	print( classification_report(y_true, y_pred) )
	print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))