'''
HOMEWORK 4 - Clustering

Pipeline for unsupervised learning.
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

'''
Read Data
'''

def read_data(filename):
	'''
	Converts a csv file into a Pandas dataframe.

	Input:
	filename (str): the name of a csv file

	Returns: a Pandas dataframe
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
	df[colname] = df[colname].map({true_val: 1, false_val: 0})

def cat_to_num(df):
	'''
	Converts all categorical variables to numeric, for use in k-means clustering.

	Input:
	df: a Pandas dataframe

	Returns:
	a dataframe with categorical variables converted to numeric
	'''
	cat_columns = df.select_dtypes(['category']).columns
	df_copy = df.copy()
	df_copy[cat_columns] = df_copy[cat_columns].apply(lambda x: x.cat.codes)
	return df_copy

'''
Build Classifier
'''

def build_k_means(df, k):
	'''
	Builds a K-means model.

	Input:
	df: a Pandas dataframe
	k: number of clusters wanted
	'''
	kmeansmod = KMeans(n_clusters=k)
	kmeansmod.fit(df)
	labels = kmeansmod.predict(df)
	return kmeansmod

def get_cluster_assignments(df, k):
	'''
	Builds a K-means model and writes the assigned cluster labels into the dataframe.

	Input:
	df: a Pandas dataframe
	k: number of clusters wanted
	'''
	labelled_df = df.copy()
	kmeansmod = build_k_means(df, k)
	results = kmeansmod.labels_
	labelled_df['label'] = results
	return labelled_df

def cluster_summary_stats(assignments, cluster_label):
	'''
	Gets summary statistics for the cluster with the specified label.

	Input:
	assignments: a Pandas dataframe with the assigned cluster labels
	cluster_label: the label of the specific cluster
	'''
	cluster = assignments[assignments['label'] == cluster_label]
	return summ_stats(cluster)

def cluster_hist(assignments, cluster_label, width, height):
	'''
	Creates histograms showing the distribution of every variable in the
	cluster.

	Input:
	assignments: a Pandas dataframe with the assigned cluster labels
	cluster_label: the label of the specific cluster
	width (int): the desired width of the output figure 
	height (int): the desired height of the output figure
	'''
	cluster = assignments[assignments['label'] == cluster_label]
	make_hist(cluster, width, height)

def cluster_boxplot(assignments, cluster_label, width, height):
	'''
	Creates boxplots showing the distribution of every variable in the
	cluster, with 3 boxplots per row.

	Input:
	assignments: a Pandas dataframe with the assigned cluster labels
	cluster_label: the label of the specific cluster
	width (int): the desired width of the output figure 
	height (int): the desired height of the output figure
	'''
	cluster = assignments[assignments['label'] == cluster_label]
	make_box(cluster, width, height)

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
For User Interaction
'''

def merge_clusters(assignments, to_be_replaced, replacement):
	'''
	To merge several clusters into one.

	Inputs:
	assignments: a Pandas dataframe with the assigned cluster labels
	to_be_replaced: list of labels of clusters to be merged
	replacement: replacement label

	Returns:
	updated cluster assignments
	'''
	new_assignments = assignments.replace(to_replace=to_be_replaced, value=replacement)

	return new_assignments

def recluster(assignments, new_k):
	'''
	To recluster with a new k (number of clusters).

	Inputs:
	assignments: a Pandas dataframe with the assigned cluster labels
	new_k: new number of clusters wanted

	Returns:
	updated cluster assignments
	'''

	df = assignments.drop('label', axis=1)
	new_assign = get_cluster_assignments(df, new_k)
	return new_assign

def split_cluster(assignments, label_to_split, num_to_split):
	'''
	To split a specific cluster into many.

	Inputs:
	assignments: a Pandas dataframe with the assigned cluster labels
	label_to_split: the label of the specific cluster to be split
	num_to_split: the specific number of new clusters to be created

	Returns:
	updated cluster assignments for the entire original dataframe
	'''
	to_split = assignments[assignments['label'] == label_to_split]
	split_assign = recluster(to_split, num_to_split)
	split_assign['label'] = 'new_' + split_assign['label'].astype(str)
	unsplit = assignments[assignments['label'] != label_to_split]
	fin = pd.concat([unsplit, split_assign])
	return fin