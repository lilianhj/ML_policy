'''
HOMEWORK 5 - Fixing the Pipeline

Fixing the pipeline according to feedback on homework 3.
'''

from __future__ import division
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing, svm, tree, decomposition
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
from scipy import optimize
import time
import seaborn as sns
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

'''
Read Data
'''

def read_data(filename):
	'''
	Converts a csv file into a Pandas dataframe.

	Input:
	filename (str): the name of a csv file

	Returns: 
	a Pandas dataframe
	'''
	return pd.read_csv(filename)

'''
Explore Data
'''

def info(df):
	'''
	Shows basic information about a dataframe:
	its size, its columns, their types, and the count of non-null values
	in each column.

	Input:
	df: a Pandas dataframe
	'''
	return df.info(null_counts=True)

def summ_stats(df):
	'''
	Shows summary statistics (count, median, standard deviation, minimum,
	percentiles, maximum) for all columns in a dataframe.

	Input:
	df: a Pandas dataframe
	'''
	return df.describe()

def null_count(df):
	'''
	Explicitly counts the number of missing values in each column.

	Input:
	df: a Pandas dataframe
	'''
	return df.isnull().sum()

def make_percent_table(df, column):
	'''
	Makes a table counting the percentage of observations that fall into
	a particular category of a variable.

	Input:
	df: a Pandas dataframe
	column (str): the column name of the variable
	'''
	table = df.groupby(column).size().reset_index(name="count")
	table["percent"] = (table["count"]/len(df)) * 100
	return table

def make_hist(df, width, height):
	'''
	Creates histograms showing the distribution of every variable in the
	dataframe.

	Input:
	df: a Pandas dataframe
	width (int): the desired width of the output figure 
	height (int): the desired height of the output figure
	'''
	df.hist(figsize=(width, height))

def corr_heatmap(df, width, height):
	'''
	Creates a heatmap showing the correlations between all variables in the
	dataframe.

	Input:
	df: a Pandas dataframe
	width (int): the desired width of the output figure 
	height (int): the desired height of the output figure
	'''
	corr = df.corr()
	png, ax = plt.subplots(figsize=(width, height))
	ax = sns.heatmap(corr, center=0, cmap=sns.diverging_palette(250, 10, 
		 as_cmap=True), annot=True)

def make_box(df, width, height):
	'''
	Creates boxplots showing the distribution of every variable in the
	dataframe, with 3 boxplots per row.

	Input:
	df: a Pandas dataframe
	width (int): the desired width of the output figure 
	height (int): the desired height of the output figure
	'''
	rows = math.ceil(len(df.columns)/3)
	df.plot(kind='box', subplots=True, layout=(rows, 3), figsize=(width, height), 
			sharex=False, sharey=False)

def crosstab(df, dep, indep):
	'''
	Makes a crosstab to compare how the distribution of the independent variable
	differs between the dependent variable's classes.

	Input:
	df: a Pandas dataframe
	dep (str): the column name of the dependent variable
	indep (str): the column name of the independent variable
	'''
	return pd.crosstab(df[dep], df[indep], normalize='index')

def crosshist(df, dep, width=5, height=5):
	'''
	Makes layered histograms to compare how the distribution of
	every independent variable differs between the dependent variable's classes.

	Input:
	df: a Pandas dataframe
	dep (str): the column name of the dependent variable
	width (int): the desired width of the output figures
	height (int): the desired height of the output figures

	Credit to:
	https://machinelearningmastery.com/quick-and-dirty-data-analysis-with-pandas/
	'''
	for col in df.columns:
		df.groupby(dep)[col].hist(alpha=0.4, figsize=(width, height))
		plt.title(col)
		plt.legend(['Negative', 'Positive'])
		plt.show()

'''
Pre-Process Data
'''

def impute_median(df):
	'''
	Fills in missing values with the median value of that variable. 

	Input:
	df: a Pandas dataframe

	Returns:
	a dataframe with missing values filled in
	'''
	return df.fillna(df.median())

'''
Generate Features
'''

def get_thresholds(df, column):
	'''
	Returns values at the 0th, 20th, 40th, 60th, 80th, and 100th percentile of a
	variable.
	Inputs:
	df: a Pandas dataframe
	column (str): name of a column in the dataframe df
	
	Returns:
	A list of quantile values.
	'''
	rv = []
	for i in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
		threshold = df[column].quantile(i)
		rv.append(threshold)
	return rv

def discretize(df, colname, bins=5, labels=False):
	'''
	Makes a continuous variable into discrete categories.

	Input:
	df: a Pandas dataframe
	colname (str): column name of the continuous variable to be discretized
	bins (int): number of categories to be created
	labels: an optional list of names to label the created categories

	Returns:
	- a dataframe with a new variable made by discretizing the continuous variable
	- a list with the automatically generated category edges/limits
	'''
	new_colname = colname + "_discrete"
	df[new_colname], bins = pd.cut(df[colname], bins=bins, labels=labels, 
							include_lowest=True, retbins=True)
	return df, bins

def make_dummies_all_cat(df):
	'''
	Identifies all categorical variables in dataframe and makes dummies from all of them.

	Input:
	df: a Pandas dataframe

	Returns:
	a dataframe with all new dummy variables
	'''
	cat = df.select_dtypes(include=['category'])
	for colname in cat.columns:
		df = make_dummy(df, colname)
	return df

def make_dummy(df, colname):
	'''
	Creates binary/dummy variables from a categorical variable, omitting one
	level of the categorical variable.

	Input:
	df: a Pandas dataframe
	colname (str): column name of the categorical variable to be made into dummies

	Returns:
	a dataframe with the new dummy variables
	'''
	dummy = pd.get_dummies(df[colname], prefix=colname, drop_first=True)
	df = pd.concat([df, dummy], axis = 1)
	return df

def recode_booleans(df, colname, true_val, false_val):
	'''
	Recodes boolean columns to say "0" for false and "1" for true

	Input:
	df: a Pandas dataframe
	colname (str): column name of the variable to be recoded
	true_val: the value currently being used to denote "true", which should be recoded as 1
	false_val: the value currently being used to denote "false", which should be recoded as 0
	'''
	df[colname] = df[colname].map({true_val: 1, false_val: 0})

'''
Build Classifier
'''

def temporal_validate(start_time, end_time, prediction_windows, outcome_days):
	'''
	Identifies times at which to split data into training and
	test sets over time.

	Input:
	start_time: start time of data
	end_time: last date of data, including labels and outcomes that we have
	prediction_windows: a list of how far out we want to predict, in months
	outcome_days: number of days needed to evaluate the outcome

	Returns:
	a list of training set start and end time, and testing set start and end time,
	for each temporally validated dataset

	Adapted with permission from Rayid Ghani: https://github.com/rayidghani/magicloops
	'''
	splits = []
	start_time_date = datetime.strptime(start_time, '%Y-%m-%d')
	end_time_date = datetime.strptime(end_time, '%Y-%m-%d')
	actual_end_time_date = end_time_date - relativedelta(days=+(outcome_days+1))

	for prediction_window in prediction_windows:
		on_window = 1
		test_end_time = start_time_date
		while (actual_end_time_date > test_end_time):
			train_start_time = start_time_date
			train_end_time = train_start_time + on_window * relativedelta(months=+prediction_window) - relativedelta(days=+(outcome_days+2))
			test_start_time = train_start_time + on_window * relativedelta(months=+prediction_window)
			test_end_time = test_start_time + relativedelta(months=+prediction_window) - relativedelta(days=+(outcome_days+2))
			splits.append([train_start_time, train_end_time, test_start_time, test_end_time])
			on_window += 1
	return splits

def temporal_split(full_data, train_start, train_end, test_start, test_end, datetime_var):
	'''
	Splits data into temporally validated training set and test sets.

	Input:
	full_data: full dataframe
	train_start: start datetime for training set
	train_end: end datetime for training set
	test_start: start datetime for test set
	test_end: end datetime for test set
	datetime_var: the name of the temporal variable being split on
	
	Returns:
	Training and testing sets for the independent and dependent variables.
	'''
	train_data = full_data[(full_data[datetime_var] >= train_start) & (full_data[datetime_var] <= train_end)]
	test_data = full_data[(full_data[datetime_var] >= test_start) & (full_data[datetime_var] <= test_end)]
	return train_data, test_data

def make_all_splits(full_data, datetime_var, full_lst, temporal_split_lst):
	'''
	Makes all training/testing datasets for a list of desires temporal splits, and
	appends all these datasets to a list for further processing.

	Input:
	full_data: full dataframe
	datetime_var: the name of the temporal variable being split on
	full_lst: the initially empty list to which to append all training/testing datasets
	temporal_split_lst: the list of desired training start/end times
	'''
	for split in temporal_split_lst:
		print(split)
		sub_lst = []
		train_start, train_end, test_start, test_end = split[0], split[1], split[2], split[3]
		train, test = temporal_split(full_data, train_start, train_end, test_start, test_end, datetime_var)
		train = impute_median(train)
		train = train.apply(lambda x: x.fillna(x.value_counts().index[0]))
		test = impute_median(test)
		test = test.apply(lambda x: x.fillna(x.value_counts().index[0]))
		sub_lst.append(train)
		sub_lst.append(test)
		print("train {}, test {}".format(train.shape, test.shape))
		full_lst.append(sub_lst)

def magic_loop(models_to_run, classifiers, parameters, df_lst, y_var, datetime_var, splits):
	"""
	Given a set of models to run, parameters, and a list of temporally split training/testing datasets, runs each model for each combination of parameters.
	
	Returns table with temporal split, model, and parameter information along with model performance on a number of specified metrics including auc-roc, precision at different levels, recall at different levels, and F1 at different levels.
	
	Table also includes baseline of the prevalence of the positive outcome label in the population for each temporal split. 

	Input:
	- models_to_run: a list of model names to be run
	- classifiers: a dictionary of models
	- parameters: a dictionary of parameters to test for each model that is run
	- df_lst: a list of temporally validated training/testing dataframes
	- y_var: the outcome variable
	- datetime_var: the datetime variable on which the temporal split was conducted
	- splits: a list of all the training and testing start/end times 

	Return:
	- results_df: the dataframe to which all the outcome metrics are written
	- final_params: a list of the parameters run for each model, for easy lookup

	Adapted with permission from Rayid Ghani: https://github.com/rayidghani/magicloops
	"""
	results_df =  pd.DataFrame(columns=('train_start', 'train_end', 'test_start', 'test_end', 'model_type', 'clf', 'parameters', 'auc-roc', 'p_at_1', 'p_at_2', 'p_at_5', 'p_at_10', 'p_at_20', 'p_at_30', 'p_at_50', 'r_at_1', 'r_at_2', 'r_at_5', 'r_at_10', 'r_at_20', 'r_at_30', 'r_at_50', 'f1_at_1', 'f1_at_2', 'f1_at_5', 'f1_at_10', 'f1_at_20', 'f1_at_30', 'f1_at_50'))
	final_params = []
	for i, sub_lst in enumerate(df_lst):
		split = splits[i]
		print(split)
		train_start, train_end, test_start, test_end = split[0], split[1], split[2], split[3]
		train, test = sub_lst[0], sub_lst[1]
		y_train = train[y_var]
		X_train = train.drop([y_var, datetime_var], axis = 1)
		y_test = test[y_var]
		X_test = test.drop([y_var, datetime_var], axis = 1)
		print("set {}: X train {}, y train {}, X test {}, y test {}".format(i, X_train.shape, y_train.shape, X_test.shape, y_test.shape))
		for index, classifier in enumerate([classifiers[x] for x in models_to_run]):
			print("Current model: {}".format(models_to_run[index]))
			parameter_values = parameters[models_to_run[index]]
			for p in ParameterGrid(parameter_values):
				try:
					print(p)
					final_params.append(p)
					classifier.set_params(**p)
					if 'SVM'in models_to_run[index]:
						y_pred_probs = classifier.fit(X_train, y_train).decision_function(X_test)
					else:
						y_pred_probs = classifier.fit(X_train, y_train).predict_proba(X_test)[:,1]
					y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
					prec_lst = []
					rec_lst = []
					f1_lst = []
					for i in [1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0]:
						prec = precision_at_k(y_test_sorted,y_pred_probs_sorted, i)
						rec = recall_at_k(y_test_sorted,y_pred_probs_sorted, i)
						f1 = get_f1_for_k(y_test_sorted,y_pred_probs_sorted, i)
						prec_lst.append(prec)
						rec_lst.append(rec)
						f1_lst.append(f1)
					results_df.loc[len(results_df)] = [train_start, train_end, test_start, test_end,
													   models_to_run[index], classifier, p,
													   roc_auc_score(y_test_sorted, y_pred_probs)] + prec_lst + rec_lst + f1_lst
				except IndexError as e:
					print('Error:',e)
					continue
		print("Writing baseline")
		results_df.loc[len(results_df)] = [train_start, train_end, test_start, test_end, "baseline", '',
					'', y_test.sum()/len(y_test), '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']	
	return results_df, final_params


def build_log(df, yvar, xvars):
	'''
	Builds and trains a logistic regression classifier.

	Input:
	df: a Pandas dataframe
	yvar (str): column name of the dependent/output variable
	xvars (list of str): list of column names of the features

	Returns:
	y: the correct output labels
	y_pred: the predicted labels given by the classifier
	'''
	y = df[yvar]
	x = df[xvars]
	logreg = LogisticRegression()
	logreg.fit(x,y)
	y_pred = logreg.predict(x)
	return y, y_pred

'''
Evaluate Classifier
'''

def generate_binary_at_k(y_scores, k):
	'''
	Converts predicted scores to binary 0/1 outcomes.

	Input:
	y_scores: an array of predicted scores
	k: a threshold proportion

	Returns:
	an array of binary 0/1 outcomes

	Adapted with permission from Rayid Ghani: https://github.com/rayidghani/magicloops
	'''
	cutoff_index = int(len(y_scores) * (k / 100.0))
	predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
	return predictions_binary

def joint_sort_descending(l1, l2):
	'''
	Sorts two numpy arrays in descending order.

	Input:
	l1, l2: numpy arrays

	Adapted with permission from Rayid Ghani: https://github.com/rayidghani/magicloops
	'''
	idx = np.argsort(l1)[::-1]
	return l1[idx], l2[idx]

def precision_at_k(y_true, y_scores, k):
	'''
	Calculates precision of a model at a given threshold.

	Input:
	y_true: an array of true outcome labels
	y_scores: an array of predicted scores
	k: a threshold proportion

	Returns:
	a precision score

	Adapted with permission from Rayid Ghani: https://github.com/rayidghani/magicloops
	'''
	y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
	preds_at_k = generate_binary_at_k(y_scores_sorted, k)
	precision = precision_score(y_true_sorted, preds_at_k)
	return precision

def recall_at_k(y_true, y_scores, k):
	'''
	Calculates recall of a model at a given threshold.

	Input:
	y_true: an array of true outcome labels
	y_scores: an array of predicted scores
	k: a threshold proportion

	Returns:
	a recall score

	Adapted with permission from Rayid Ghani: https://github.com/rayidghani/magicloops
	'''
	y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
	preds_at_k = generate_binary_at_k(y_scores_sorted, k)
	recall = recall_score(y_true_sorted, preds_at_k)
	return recall

def get_f1_for_k(y_true, y_scores, k):
	"""
	Calculates the F1 score using the formula F1 = 2 * (precision * recall) / (precision + recall)

	Input:
	y_true: an array of true outcome labels
	y_scores: an array of predicted scores
	k: a threshold proportion

	Returns: An F1 score.
	"""

	p_at_k = precision_at_k(y_true, y_scores, k)
	r_at_k = recall_at_k(y_true, y_scores, k)

	return 2*(p_at_k*r_at_k)/(p_at_k + r_at_k)	

def plot_precision_recall_n(y_true, y_prob, model_name):
	'''
	Plots precision-recall curve for a model.

	Input:
	y_true: an array of true outcome labels
	y_prob: an array of predicted probabilities
	model_name (str): the name of the model to be used as the graph title

	Adapted with permission from Rayid Ghani: https://github.com/rayidghani/magicloops
	'''
	y_score = y_prob
	precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
	
	precision_curve = precision_curve[:-1]
	recall_curve = recall_curve[:-1]
	pct_above_per_thresh = []
	number_scored = len(y_score)
	for value in pr_thresholds:
		num_above_thresh = len(y_score[y_score >= value])
		pct_above_thresh = num_above_thresh / float(number_scored)
		pct_above_per_thresh.append(pct_above_thresh)
	pct_above_per_thresh = np.array(pct_above_per_thresh)  
	plt.clf()
	fig, ax1 = plt.subplots()
	ax1.plot(pct_above_per_thresh, precision_curve, 'b')
	ax1.set_xlabel('percent of population')
	ax1.set_ylabel('precision', color='b')
	ax2 = ax1.twinx()
	ax2.plot(pct_above_per_thresh, recall_curve, 'r')
	ax2.set_ylabel('recall', color='r')
	ax1.set_ylim([0,1])
	ax1.set_ylim([0,1])
	ax2.set_xlim([0,1])    
	name = model_name
	plt.title(name)
	plt.show()

def get_accuracy(y, y_pred):
	'''
	Evaluates the accuracy of the classifier's predictions.

	Input:
	y: the correct output labels
	y_pred: the predicted labels given by the classifier
	'''
	return metrics.accuracy_score(y, y_pred)

def conf_matrix(y, y_pred, figsize=(10, 7)):
	'''
	Visualizes the confusion matrix of the classifier's predictions.

	Input:
	y: the correct output labels
	y_pred: the predicted labels given by the classifier
	figsize (tuple): a tuple setting the size of the output figure. first value
	is width, second value is height.
	'''	
	cnf_matrix = metrics.confusion_matrix(y, y_pred)
	class_names = ['Negative', 'Positive']
	fig, ax = plt.subplots()
	tick_marks = np.arange(len(class_names))
	plt.xticks(tick_marks, class_names)
	plt.yticks(tick_marks, class_names)
	sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, fmt='g')
	ax.xaxis.set_label_position("top")
	plt.tight_layout()
	plt.ylabel('Actual label')
	plt.xlabel('Predicted label')
