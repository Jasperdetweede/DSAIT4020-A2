import os
import json
import numpy as np
import pandas as pd

from raw_data_loader import load_raw_data

from preprocessing_depression import clean_and_preprocess_depression_data
from preprocessing_insomnia import clean_and_preprocess_insomnia_data
from preprocessing_electrical_circuit import gen_and_preprocess_ec_data

from data_balancing import resample_training_data

from baseline_models import train_multitarget_baseline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from proposed_models import train_baseline_mlp, train_joint_model, train_split_model, train_deep_joint_model, train_deep_split_model

def run_everything():
	run_crossvalidation( 'depression' )
	run_crossvalidation( 'insomnia' )
	run_crossvalidation( 'electrical_circuit' )

def run_crossvalidation( dataset_name ):

	#################
	# Data
	#################

	# Paths
	RAW_DATA_FOLDER = 'raw_data'
	TARGET_FILE_PATH = 'unprocessed_data'
	
	if not os.path.exists(TARGET_FILE_PATH):
		print( "Recreating Unprocessed Data" )
		load_raw_data(RAW_DATA_FOLDER, TARGET_FILE_PATH)

	# Flow Controls
	DATA = dataset_name

	# System variables
	STATE = 42
	TEST_SET_FRACTION = 0.20
	MISSING_VALUES_THRESHOLD = 0.50
	SAMPLES_ELECTRICAL_CIRCUIT = 5000
	VERBOSE = False
	FLIP_LABEL_FRACTION = 0.03

	FOLDS = 7
	
	baseline_MSE_per_fold = []
	prop_results_per_fold = {
		"joint": [],
		"split": [],
		"deep_joint": [],
		"deep_split": [],
		"baseline_MLP": []
	}

	# Assertions before starting
	assert FLIP_LABEL_FRACTION > 0.0 and FLIP_LABEL_FRACTION < 1.0, "FLIP_LABEL_FRACTION should be beween 0.0 and 1.0"

	#################
	# Cross val loop
	#################

	for fold in range(FOLDS):
		# Data loading
		if DATA == 'depression':
			dataset = pd.read_csv(TARGET_FILE_PATH + '/' + DATA + '_data.csv')
			X_train, X_test, y_train, y_test, y_embed_train, y_embed_test = clean_and_preprocess_depression_data(dataset, RAW_DATA_FOLDER, TEST_SET_FRACTION, STATE, MISSING_VALUES_THRESHOLD, True, FOLDS, fold)
			DO_SMOTE = True
		elif DATA == 'insomnia':
			dataset = pd.read_csv(TARGET_FILE_PATH + '/' + DATA + '_data.csv')
			X_train, X_test, y_train, y_test, y_embed_train, y_embed_test = clean_and_preprocess_insomnia_data(dataset, RAW_DATA_FOLDER, TEST_SET_FRACTION, STATE, MISSING_VALUES_THRESHOLD, True, FOLDS, fold)
			y_embed_train = y_embed_train.astype(np.float64)
			y_embed_test = y_embed_test.astype(np.float64)
			DO_SMOTE = False
		elif DATA == 'electrical_circuit':
			X_train, X_test, y_train, y_test, y_embed_train, y_embed_test = gen_and_preprocess_ec_data(SAMPLES_ELECTRICAL_CIRCUIT, TEST_SET_FRACTION, STATE, True, FOLDS, fold)
			DO_SMOTE = False
		else:
			raise ValueError("Invalid dataset selected")

		# Balancing for depression
		if DO_SMOTE:
			X_train, y_train, y_embed_train = resample_training_data(X_train, y_train, y_embed_train, random_state=STATE)

		# Introduce noise
		if FLIP_LABEL_FRACTION > 0.0:
			num_to_flip = int(FLIP_LABEL_FRACTION * len(y_train))
			flip_indices = np.random.choice(len(y_train), size=num_to_flip, replace=False)

			# If y_train is a pandas Series, convert to int for safe arithmetic
			if hasattr(y_train, 'iloc'):
				y_train = y_train.astype(int)
				y_train.iloc[flip_indices] = 1 - y_train.iloc[flip_indices]
			else:  # numpy array
				y_train[flip_indices] = 1 - y_train[flip_indices]

		# Type safety sanity check
		X_train = X_train.values if hasattr(X_train, "values") else np.array(X_train)
		X_test = X_test.values if hasattr(X_test, "values") else np.array(X_test)
		y_train = y_train.values.ravel() if hasattr(y_train, "values") else np.array(y_train).ravel()
		y_test = y_test.values.ravel() if hasattr(y_test, "values") else np.array(y_test).ravel()
		y_embed_train = y_embed_train.values if hasattr(y_embed_train, "values") else np.array(y_embed_train)
		y_embed_test = y_embed_test.values if hasattr(y_embed_test, "values") else np.array(y_embed_test)

		assert(isinstance(X_train, np.ndarray))
		assert(isinstance(X_test, np.ndarray))
		assert(isinstance(y_train, np.ndarray))
		assert(isinstance(y_test, np.ndarray))
		assert(isinstance(y_embed_train, np.ndarray))
		assert(isinstance(y_embed_test, np.ndarray))

		baselines_avg_MSE = train_and_test_baselines(X_train, X_test, y_train, y_test, y_embed_train, y_embed_test, STATE, VERBOSE, fold, DATA)
		baseline_MSE_per_fold.append(baselines_avg_MSE)

		#################
		# Proposed models
		#################

		DEVICE = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
		E_KEEP_RATE = 0.7
		l = 1
		if DATA == 'depression':
			l = 1e-2
		elif DATA == 'insomnia':
			l = 1e-2
		elif DATA == 'electrical_circuit':
			l = 1

		EPOCHS = 100
		AUGMENT_EPOCHS = EPOCHS//2
		EARLY_STOP_EPOCHS = EPOCHS//5

		# Sanity Checks
		assert X_train.shape[0] >= 100 and y_train.shape[0] >= 100 and y_embed_train.shape[0] >= 100, "Arrays must have at least 100 samples for the check."
		assert (len(X_train[:100]) == len(y_train[:100])) and (len(X_train[:100]) == len(y_embed_train[:100])), "First 100 samples of X_train, y_train, and y_embed_train are not aligned."
		# Run and safe
		proposed_model_fold_results = train_and_test_propositions(X_train, X_test, y_train, y_test, y_embed_train, y_embed_test, STATE, E_KEEP_RATE, EPOCHS, AUGMENT_EPOCHS, EARLY_STOP_EPOCHS, DEVICE, l)

		for model_name, metrics in proposed_model_fold_results.items():
			prop_results_per_fold[model_name].append(metrics)

	# Finalize by calculating measures and printing to json log
	# print_baseline_results_to_json(baseline_MSE_per_fold)
	print_proposed_models_results_to_json(prop_results_per_fold)

#################
# Proposed models
#################	

def train_and_test_propositions(X_train, X_test, y_train, y_test, y_embed_train, y_embed_test, STATE, E_KEEP_RATE, EPOCHS, AUGMENT_EPOCHS, EARLY_STOP_EPOCHS, DEVICE, l):
	results = {}

	results["joint"] = train_joint_model( X_train, X_test, y_train, y_test, y_embed_train, y_embed_test,
					e_kept_ratio=E_KEEP_RATE,
					l=l,
					epochs=EPOCHS,
					augment_epochs=AUGMENT_EPOCHS,
					early_stop_epochs=EARLY_STOP_EPOCHS,
					device=DEVICE
				)
	
	results["split"] = train_split_model( X_train, X_test, y_train, y_test, y_embed_train, y_embed_test,
					 e_kept_ratio=E_KEEP_RATE,
					 epochs=EPOCHS,
					 augment_epochs=AUGMENT_EPOCHS,
					 early_stop_epochs=EARLY_STOP_EPOCHS,
					 device=DEVICE
				 )
	
	results["deep_joint"] = train_deep_joint_model( X_train, X_test, y_train, y_test, y_embed_train, y_embed_test,
						 e_kept_ratio=E_KEEP_RATE,
						 l=l,
						 epochs=EPOCHS,
						 augment_epochs=AUGMENT_EPOCHS,
						 early_stop_epochs=EARLY_STOP_EPOCHS,
						 device=DEVICE
					 )
	
	results["deep_split"] = train_deep_split_model( X_train, X_test, y_train, y_test, y_embed_train, y_embed_test,
						 e_kept_ratio=E_KEEP_RATE,
						 epochs=EPOCHS,
						 augment_epochs=AUGMENT_EPOCHS,
						 early_stop_epochs=EARLY_STOP_EPOCHS,
						 device=DEVICE
					 )
	
	results["baseline_MLP"] = train_baseline_mlp( X_train, X_test, y_train, y_test, y_embed_train, y_embed_test,
						e_kept_ratio=E_KEEP_RATE,
						epochs=EPOCHS,
						augment_epochs=AUGMENT_EPOCHS,
						early_stop_epochs=EARLY_STOP_EPOCHS,
						device=DEVICE
					)

	return results


#################
# Baselines
#################	

def train_and_test_baselines(X_train, X_test, y_train, y_test, y_embed_train, y_embed_test, STATE, VERBOSE, fold, DATA):
	nb_model = GaussianNB()
	y_pred_nb, json_metrics_nb = train_multitarget_baseline(
								model=nb_model,
								is_classifier=False,
								X_train=X_train,
								X_test=X_test,
								y_embed_train=y_embed_train,
								y_embed_test=y_embed_test,
								verbose=VERBOSE)
	
	# Parameter
	N_ESTIMATORS = 100

	rf_model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=STATE, n_jobs=-1)
	y_pred_rf, json_metrics_rf = train_multitarget_baseline(
										model=rf_model, 
										is_classifier=False, 
										X_train=X_train, 
										X_test=X_test, 
										y_embed_train=y_embed_train, 
										y_embed_test=y_embed_test,
										verbose=VERBOSE)
	
	# Parameters
	MAX_ITERATIONS = 1000

	log_model = LogisticRegression(max_iter=MAX_ITERATIONS, class_weight='balanced', random_state=STATE)
	y_pred_log, json_metrics_logistic = train_multitarget_baseline(
								model=log_model,
								is_classifier=False,
								X_train=X_train,
								X_test=X_test,
								y_embed_train=y_embed_train,
								y_embed_test=y_embed_test,
								verbose=VERBOSE)
	
	json_metrics_nb["cv_fold"] = fold
	json_metrics_rf["cv_fold"] = fold
	json_metrics_logistic["cv_fold"] = fold
	json_metrics_nb["dataset"] = DATA
	json_metrics_rf["dataset"] = DATA
	json_metrics_logistic["dataset"] = DATA
	
	with open('log.json', 'a') as f: 
		json.dump(json_metrics_nb, f, indent=4)
		json.dump(json_metrics_rf, f, indent=4)
		json.dump(json_metrics_logistic, f, indent=4)

	return np.array([json_metrics_nb["avg_train_MSE"], 
			json_metrics_nb["avg_test_MSE"],
			json_metrics_rf["avg_train_MSE"], 
			json_metrics_rf["avg_test_MSE"], 
			json_metrics_logistic["avg_train_MSE"], 
			json_metrics_logistic["avg_test_MSE"]
	])

#################
# Baselines
#################  

def print_baseline_results_to_json(baseline_MSE_per_fold):
	baseline_MSE_per_fold = np.array(baseline_MSE_per_fold)

	mean_MSE = baseline_MSE_per_fold.mean(axis=0)
	var_MSE  = baseline_MSE_per_fold.var(axis=0)

	results = {
		"NB": {
			"train_mean": mean_MSE[0],
			"train_var":  var_MSE[0],
			"test_mean":  mean_MSE[1],
			"test_var":   var_MSE[1],
		},
		"RF": {
			"train_mean": mean_MSE[2],
			"train_var":  var_MSE[2],
			"test_mean":  mean_MSE[3],
			"test_var":   var_MSE[3],
		},
		"LOG": {
			"train_mean": mean_MSE[4],
			"train_var":  var_MSE[4],
			"test_mean":  mean_MSE[5],
			"test_var":   var_MSE[5],
		}
	}

	with open('baseline_results.json', 'a') as f: 
		json.dump(results, f, indent=4)

def print_proposed_models_results_to_json(prop_results_per_fold):

	results_list = []

	for model_name, lst in prop_results_per_fold.items():
		arr = np.array(lst)  

		# Skip if empty
		if arr.size == 0:
			print(f"No results for model {model_name}, skippin it")
			continue

		result = {
			"model_name": model_name,
			"train_mean": float(arr[:, 0].mean()),
			"train_var":  float(arr[:, 0].var()),
			"test_mean":  float(arr[:, 1].mean()),
			"test_var":   float(arr[:, 1].var()),
			"aug_test_mean": float(arr[:, 2].mean()),
			"aug_test_var":  float(arr[:, 2].var()),
		}

		results_list.append(result)

	# Save all results to JSON at once
	with open("proposed_models_results.json", "w") as f:
		json.dump(results_list, f, indent=4)
		
if __name__ == "__main__":
	run_everything()
