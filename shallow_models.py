import torch
from torch import nn
from collections import OrderedDict
from base_model import BaseModel

class JointModel(BaseModel):
	"""=======================================================
	Shallow model with joint loss for regressor and classifier
	======================================================="""
	def __init__( self, n_features, hidden_size, l, device="cpu" ):
		super().__init__( n_features, hidden_size, device=device )

		self.optim = torch.optim.Adam( self.parameters(), lr=1e-3 )
		self.l = l
		self.to(device)

	def backward( self, y_pred, y_true, e_pred, e_true=None ):
		loss = self.loss_clf( y_pred, y_true )
		if e_true is not None:
			loss += self.l * self.loss_reg( e_pred, e_true )
		self.optim.zero_grad()
		loss.backward()
		self.optim.step()
		return loss

class SplitModel(BaseModel):
	"""=======================================================
	Shallow model with split loss for regressor and classifier
	======================================================="""
	def __init__( self, n_features, hidden_size, device="cpu" ):
		super().__init__( n_features, hidden_size, device=device )

		self.optim_reg = torch.optim.Adam(self.regressor.parameters(), lr=1e-3)
		self.optim_clf = torch.optim.Adam(self.classifier.parameters(), lr=1e-3)
		self.optim_e2e = torch.optim.Adam(self.parameters(), lr=1e-3)
		self.to(self.device)

	def forward( self, X, end_to_end=True ):
		e_pred = self.regressor( X )
		if end_to_end:
			y_pred = self.classifier( e_pred )
		else:
			e_copy = e_pred.clone().detach()
			y_pred = self.classifier( e_copy )

		return e_pred, y_pred
	

	def backward( self, y_pred, y_true, e_pred, e_true=None ):
		classification_loss = self.loss_clf( y_pred, y_true )
		if e_true is not None:
			regression_loss = self.loss_reg( e_pred, e_true )
			self.optim_reg.zero_grad()
			regression_loss.backward()
			self.optim_reg.step()
			self.optim_clf.zero_grad()
			classification_loss.backward()
			self.optim_clf.step()
		else:
			self.optim_e2e.zero_grad()
			classification_loss.backward()
			self.optim_e2e.step()
		return classification_loss


if __name__ == "__main__":
	print("Usage only as a module, provides classes:")
	print("JointModel( n_features, hidden_size, l, device=\"cpu\" )")
	print("SplitModel( n_features, hidden_size, device=\"cpu\" )")
