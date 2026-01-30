import torch
import pandas as pd
import numpy as np
from shallow_models import JointModel, SplitModel
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
		if isinstance( X, np.ndarray ):
			return torch.from_numpy( X ).to(dtype=dtype)
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
		if isinstance( X, np.ndarray ):
			return torch.from_numpy( X ).to(dtype=dtype)
		return torch.tensor( X, dtype=dtype )
	
######################
# MAIN TRAIN FUNCTIONS
######################

def train_joint_model( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio, l=1.0, epochs=1000, augment_epochs=50, early_stop_epochs=20, device="cpu" ):
	datasets, layer_sizes  = get_datasets_and_layer_sizes( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio )
	model = JointModel( n_features=layer_sizes["n_in"], hidden_size=layer_sizes["n_reg"], l=l, device=device )
	result = train_proposal_model( datasets, model, title="Joint", epochs=epochs, augment_epochs=augment_epochs, early_stop_epochs=early_stop_epochs, device=device )
	return result

def train_split_model( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio, epochs=1000, augment_epochs=50, early_stop_epochs=20, device="cpu" ):
	datasets, layer_sizes  = get_datasets_and_layer_sizes( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio )
	model = SplitModel( n_features=layer_sizes["n_in"], hidden_size=layer_sizes["n_reg"], device=device )
	result = train_proposal_model( datasets, model, title="Split", epochs=epochs,augment_epochs=augment_epochs, early_stop_epochs=early_stop_epochs, device=device )
	return result

def train_deep_joint_model( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio, l=1.0, epochs=1000, augment_epochs=50, early_stop_epochs=20, device="cpu" ):
	datasets, layer_sizes  = get_datasets_and_layer_sizes( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio )
	model = DeepJointModel( n_features=layer_sizes["n_in"], hidden_size=layer_sizes["n_reg"], l=l, device=device )
	result = train_proposal_model( datasets, model, title="Deep Joint", epochs=epochs, augment_epochs=augment_epochs, early_stop_epochs=early_stop_epochs, device=device )
	return result

def train_deep_split_model( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio, epochs=1000, augment_epochs=50, early_stop_epochs=20, device="cpu" ):
	datasets, layer_sizes  = get_datasets_and_layer_sizes( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio )
	model = DeepSplitModel( n_features=layer_sizes["n_in"], hidden_size=layer_sizes["n_reg"], device=device )
	result = train_proposal_model( datasets, model, title="Deep Split", epochs=epochs, augment_epochs=augment_epochs, early_stop_epochs=early_stop_epochs, device=device )
	return result

######################
# HELPER FUNCTIONS
######################

def get_datasets_and_layer_sizes( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio, device="cpu" ):
	n_samples, n_features = X_train.shape
	_, hidden_size = e_train.shape
	n_samples_w_e = int( n_samples * e_kept_ratio )
	
	datasets = {
		"train_w_e": CustomDatasetWithE( X_train[:n_samples_w_e], y_train[:n_samples_w_e], e_train[:n_samples_w_e], device=device ),
		"train_no_e": CustomDatasetNoE( X_train[n_samples_w_e:], y_train[n_samples_w_e:], device=device ),
		"test": CustomDatasetWithE( X_test, y_test, e_test, device=device )
	}

	layer_sizes = {
		"n_in": n_features,
		"n_reg": hidden_size
	}

	return datasets, layer_sizes

def train_proposal_model( datasets, model, title, batch_size=32, epochs=1000, augment_epochs=50, early_stop_epochs=20, device="cpu" ):
	dataloader_train_w_e = DataLoader( datasets["train_w_e"], batch_size=batch_size, shuffle=False )
	dataloader_train_no_e = DataLoader( datasets["train_no_e"], batch_size=batch_size, shuffle=False )
	dataloader_test = DataLoader( datasets["test"], batch_size=batch_size, shuffle=False )

	e_pred_train, y_pred_train = model.fit_predict( dataloader_train_w_e, epochs=epochs, early_stop_epochs=early_stop_epochs )
	train_mse = mean_squared_error(datasets["train_w_e"].embedding, e_pred_train)
	#present_model_metrics( datasets["train_w_e"].y, y_pred_train, , e_pred_train, title=f"{title} MLP [Training]" )
	e_pred_test, y_pred_test = model.predict( dataloader_test )
	test_mse = mean_squared_error(datasets["test"].embedding, e_pred_test)
	#present_model_metrics( datasets["test"].y, y_pred_test, , e_pred_test, title=f"{title} MLP [Testing]" )

	model.fit( dataloader_train_no_e, epochs=augment_epochs, early_stop_epochs=early_stop_epochs )
	e_pred_test, y_pred_test = model.predict( dataloader_test )
	aug_test_mse = mean_squared_error(datasets["test"].embedding, e_pred_test)
	#present_model_metrics( datasets["test"].y, y_pred_test, datasets["test"].embedding, e_pred_test, title=f"{title} MLP (Augmented) [Testing]" )
	return train_mse, test_mse, aug_test_mse

def present_model_metrics( y_true, y_pred, e_true, e_pred, title ):
	len_eqs = (74 - len(title))//2
	print( "="*80 )
	print( "="*len_eqs + f"   {title}   " + "="*len_eqs )
	print( "="*80 )
	print( "Regression Results:" )
	print( f"MSE:\t{ mean_squared_error(e_true, e_pred) }\n\n" )
	
	print( "Classification Results:" )
	print( f"F1 score: {f1_score(y_true, y_pred)}" )
	print( classification_report(y_true, y_pred) )
	print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
