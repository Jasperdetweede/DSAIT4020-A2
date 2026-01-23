from joint_model import JointModel
from split_model import SplitModel
from deep_models import DeepJointModel, DeepSplitModel
from sklearn.metrics import confusion_matrix, f1_score, classification_report, mean_squared_error

def train_joint_model( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio, l=1, epochs=1000, device="cpu" ):
	n_samples, n_features = X_train.shape
	_, hidden_size = e_train.shape
	model = JointModel( n_features=n_features, hidden_size=hidden_size, l=l )
	train_proposal_model( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio, model, title="Joint", epochs=epochs, device=device )

def train_split_model( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio, epochs=1000, device="cpu" ):
	n_samples, n_features = X_train.shape
	_, hidden_size = e_train.shape
	model = SplitModel( n_features=n_features, hidden_size=hidden_size )
	train_proposal_model( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio, model, title="Split", epochs=epochs, device=device )

def train_deep_joint_model( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio, l=1, epochs=1000, device="cpu" ):
	n_samples, n_features = X_train.shape
	_, hidden_size = e_train.shape
	model = DeepJointModel( n_features=n_features, hidden_size=hidden_size, l=l )
	train_proposal_model( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio, model, title="Deep Joint", epochs=epochs, device=device )

def train_deep_split_model( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio, epochs=1000, device="cpu" ):
	n_samples, n_features = X_train.shape
	_, hidden_size = e_train.shape
	model = DeepSplitModel( n_features=n_features, hidden_size=hidden_size )
	train_proposal_model( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio, model, title="Deep Split", epochs=epochs, device=device )

def train_proposal_model( X_train, X_test, y_train, y_test, e_train, e_test, e_kept_ratio, model, title, epochs=1000, device="cpu" ):
	n_samples, n_features = X_train.shape
	_, hidden_size = e_train.shape
	
	n_samples_w_e = int( n_samples * e_kept_ratio )
	X_train_w_e = X_train[:n_samples_w_e]
	X_train_no_e = X_train[n_samples_w_e:]
	y_train_w_e = y_train[:n_samples_w_e]
	y_train_no_e = y_train[n_samples_w_e:]
	e_train = e_train[:n_samples_w_e]

	model.fit( X_train_w_e, y_train_w_e, e_train, epochs=epochs )
	e_pred_test, y_pred_test = model.predict( X_test )
	present_model_metrics( y_test, y_pred_test, e_test, e_pred_test, title=f"{title} MLP" )

	model.fit( X_train_no_e, y_train_no_e, epochs=epochs )
	e_pred_test, y_pred_test = model.predict( X_test )
	present_model_metrics( y_test, y_pred_test, e_test, e_pred_test, title=f"{title} MLP (Augmented)" )

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

