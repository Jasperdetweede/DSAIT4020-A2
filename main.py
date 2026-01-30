import json
import numpy as np
import pandas as pd

#from preprocessing_depression import clean_and_preprocess_depression_data
from preprocessing_insomnia import clean_and_preprocess_insomnia_data
from preprocessing_electrical_circuit import gen_and_preprocess_ec_data

#from data_balancing import resample_training_data

from baseline_models import train_multitarget_baseline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

def run_everything():
	#run_crossvalidation( 'depression' )
	run_crossvalidation( 'insomnia' )
	#run_crossvalidation( 'electrical_circuit' )

def run_crossvalidation( dataset_name ):

	#################
	# Data
	#################

	# Paths
	RAW_DATA_FOLDER = 'raw_data'
	TARGET_FILE_PATH = 'unprocessed_data'

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

	np.random.seed(STATE)

	# Assertions before starting
	assert FLIP_LABEL_FRACTION > 0.0 and FLIP_LABEL_FRACTION < 1.0, "FLIP_LABEL_FRACTION should be beween 0.0 and 1.0"

	#################
	# Cross val loop
	#################

	for fold in range(FOLDS):
		# Data loading
		dataset = pd.read_csv(TARGET_FILE_PATH + '/' + DATA + '_data.csv')

		#if DATA == 'depression':
			#X_train, X_test, y_train, y_test, y_embed_train, y_embed_test = clean_and_preprocess_depression_data(dataset, RAW_DATA_FOLDER, TEST_SET_FRACTION, STATE, MISSING_VALUES_THRESHOLD, True, FOLDS, fold)
			#DO_SMOTE = True
		#elif DATA == 'insomnia':
		if DATA == 'insomnia':
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
		#if DO_SMOTE:
		#	X_train, y_train, y_embed_train = resample_training_data(X_train, y_train, y_embed_train, random_state=STATE)

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

		#################
		# Baselines
		#################

		baselines_avg_MSE = train_and_test_baselines(X_train, X_test, y_train, y_test, y_embed_train, y_embed_test, STATE, VERBOSE, fold, DATA)
		# TODO Keep baselines_avg_MSE per iteration to calculate variance and mean per model.

		#################
		# Proposed models
		#################
	
		# TODO something similar to baselines, where it runs and keeps the results., 

#################
# Baselines
#################	

def train_and_test_baselines(X_train, X_test, y_train, y_test, y_embed_train, y_embed_test, STATE, VERBOSE, fold, DATA):
	print(X_train.shape, type( y_embed_train ))
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

	return {
		"bayes": (json_metrics_nb["avg_train_MSE"], json_metrics_nb["avg_test_MSE"]),
		"forest": (json_metrics_rf["avg_train_MSE"], json_metrics_rf["avg_test_MSE"]), 
		"bayes": (json_metrics_logistic["avg_train_MSE"], json_metrics_logistic["avg_test_MSE"]),  
	}

if __name__ == "__main__":
	run_everything()
