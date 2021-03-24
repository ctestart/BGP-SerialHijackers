from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.base import clone
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import numpy as np
import pandas as pd


def readFile(fileName):
	'''Returns list of lines in file'''
	with open(fileName,'r') as inputFile:
		lines=inputFile.readlines()
		return lines


def dropcol_importances(rf, X_train, y_train):
	'''Return a list of tuples (feature_name, importance, index) reverse sorted by importance usuing drop column method'''
	rf_ = clone(rf)
	rf_.random_state = 999
	rf_.fit(X_train, y_train)
	baseline = rf_.oob_score_
	imp = []
	for col in X_train.columns:
		X = X_train.drop(col, axis=1)
		rf_ = clone(rf)
		rf_.random_state = 999
		rf_.fit(X, y_train)
		o = rf_.oob_score_
		imp.append(baseline - o)
	imp = np.array(imp)
	return sorted(zip(X_train.columns, imp, range(len(X_train.columns))), reverse = True, key = lambda kv : kv[1])

def main():
	#Files (assuming all files in a folder named ML)
	gt_file = 'ML/groundtruth_dataset.pkl'
	pred_set = 'ML/prediction_set_with_class.pkl'

	# IPv
	ipv= 'v4'

	#Creating pandas dataframe and converting to numerical variables.
	data = pd.read_pickle(gt_file)
	data_predict = pd.read_pickle(pred_set)
	#Checking file are correct
	print 'Ground Truth data'
	print data.shape
	print 'Prediction set'
	print data_predict.shape
	cols = data.columns
	data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')
	# print data_predict.columns # To check all columns of prediction dataset.
	data_predict.drop('HardVotePred', inplace=True, axis=1) # dropping the column with the result of the classifier used for cam ready submission (in github)
	pred_cols = data_predict.columns
	# print pred_cols # To check all columns of prediction dataset.
	data_predict[pred_cols] = data_predict[pred_cols].apply(pd.to_numeric, errors='coerce')
	# print data.dtypes # To check the types of columns, all should be int or float.

	# Feature selection 
	features = list(data.columns)
	print 'Features: %d' %(len(features))
	print features

	# Getting ASN class
	print data.head()
	print 'Ground truth shape :' 
	print data['class'].value_counts()
	X_train, y_train = data.drop('class', axis = 1), data['class']


	# Sampling method for balancing (not all are used, meta-structure left for further trials)
	rus = RandomUnderSampler(return_indices=False)
	ros = RandomOverSampler()
	smote = SMOTE(ratio='minority')
	smt = SMOTETomek(ratio='auto')
	tl = TomekLinks(return_indices=False, ratio='majority')
	cc_models =[ClusterCentroids(ratio={0: x}) for x in range(15, 30, 1)]
	SMs =[rus, ros, smote, smt, tl]+ cc_models # This array encodes the order for accessing balancing techniques
	without_asn = [2,3]


	model_accuracy = []
	oob_decisions = []
	estimators_list = []
	forest_count = 0
	forest_size = 500 # Picking forest size
	models_reps = [0,10,10,10,4]+[0]*len(cc_models) # Picking number of ensembles and technique to be used to balance training set for those ensembles (following the SMs list line 83)
	for i, sm in enumerate(SMs):
		rep = models_reps[i]

		if i in without_asn:
			X_train, y_train = data.drop(['ASN','class'], axis = 1), data['class']
		else:
			X_train, y_train = data.drop('class', axis = 1), data['class']

		for r in range (rep):
			X_sm, y_sm = sm.fit_sample(X_train, y_train)

			#Balanced dataset
			data_balanced = pd.DataFrame({X_train.columns[k] : X_sm[:,k] for k in range(len(X_train.columns))})
			data_balanced.loc [:, 'class'] = y_sm.tolist()
			if r == 0:
				print 'Balanced dataset shape :' 
				print data_balanced['class'].value_counts()
			# print data_balanced.head()
			if i in without_asn:
				X, y = data_balanced.drop('class', axis = 1), data_balanced['class']
			else:
				X, y = data_balanced.drop(['ASN','class'], axis = 1), data_balanced['class']

			# Train a forest and get accuracy 
			forest = ExtraTreesClassifier(n_estimators=forest_size, bootstrap=True, oob_score=True)
			forest.fit(X, y)
			print 'Sampling method %d, Forest index %d, Model accuracy using OOB %f'%(i, forest_count, forest.oob_score_)
			model_accuracy.append(forest.oob_score_)

			# Getting OOB prediction
			data_balanced.loc [:, 'oob_class_pred'] = [forest.classes_[x] for x in np.argmax(forest.oob_decision_function_, axis=1)]
			data_balanced.loc [:, 'class_%d'%forest.classes_[0]] = forest.oob_decision_function_[:,0]
			data_balanced.loc [:, 'class_%d'%forest.classes_[1]] = forest.oob_decision_function_[:,1]
			# Saving OOB predictions in ML/OOBpred folder for later analysis (folder needs to be created in advance)
			# outfile = 'ML/OOBpred/OOBpredictions_%dt_%de_eIndex%d'%(forest_size, sum(models_reps),len(estimators_list))
			# data_balanced.to_csv(outfile+'.csv')
			# data_balanced.to_pickle(outfile+'.pkl')

			# Appending to estimators list
			estimators_list.append(('forest%d'%forest_count, forest))
			forest_count +=1

	#Saving Model Accuracy in folder ML/MA (needs to be created in advance)
	# outfile = "ML/MA/"+'OOBscores_%df_%dt_%de'%(len(features), forest_size, len(estimators_list))
	# with open(outfile, 'w') as f:
	# 	for oobscore in model_accuracy:
	# 		f.write(str(oobscore)+'\n')

	#Getting prediction in prediction dataset
	weights = None
	hard_vote_pred = np.asarray([clf[1].predict(data_predict.drop('ASN', axis=1)) for clf in estimators_list]).T
	hard_vote_pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=weights)), axis=1, arr=hard_vote_pred.astype('int'))
	
	# Adding Predictions
	data_predict.loc[:,'HardVotePred'] = hard_vote_pred
	print 'Hard Vote'
	print data_predict['HardVotePred'].value_counts()
	
	# Output to file 
	outfile = 'ML/predictions_%df_%dt_%de'%(len(features), forest_size, len(estimators_list))
	data_predict.to_csv(outfile+'.csv')
	data_predict.to_pickle(outfile+'.pkl')

if __name__ == '__main__':
	main()
