#!/usr/bin/python
###################################################################################################################
### This code is developed by HighEnergyDataScientests Team.
### Do not copy or modify without written approval from one of the team members.
###################################################################################################################

import pandas as pd
import numpy as np
import xgboost as xgb
import operator

#from __future__ import print_function

import math

from neatsociety import nn, parallel, population, visualize

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
import sklearn.metrics

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg") #Needed to save figures
import time
import os


### Controlling Parameters
output_col_name = "target"
test_col_name = "PredictedProb"
enable_feature_analysis = 1
id_col_name = "ID"
num_iterations = 5

### Creating output folders
if not os.path.isdir("../predictions"):
    os.mkdir("../predictions")

if not os.path.isdir("../intermediate_data"):
    os.mkdir("../intermediate_data")

if not os.path.isdir("../saved_states"):
    os.mkdir("../saved_states")


def ceate_feature_map(features,featureMapFile):
    outfile = open(featureMapFile, 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def fitness(genome):
    net = nn.create_feed_forward_phenotype(genome)
    output = net.array_activate(X_train[features].values)
    logloss_error = sklearn.metrics.log_loss(y_train, output[:,0])
    return 1.0 - logloss_error
        
def train_model(features,num_generations):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print("########################## Time Stamp ==== " + timestamp)
    t0 = time.time()
    
    print("## Train a NEAT model")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'bnp_config')
    
    # Use a pool of four workers to evaluate fitness in parallel.
    pe = parallel.ParallelEvaluator(fitness,3,progress_bar=True,verbose=1)

    pop = population.Population(config_path)
    pop.run(pe.evaluate, num_generations)
    
    print("total evolution time {0:.3f} sec".format((time.time() - t0)))
    print("time per generation {0:.3f} sec".format(((time.time() - t0) / pop.generation)))
    print('Number of evaluations: {0:d}'.format(pop.total_evaluations))
    
    # Verify network output against training data.
    print("## Test against verification data.")
    winner = pop.statistics.best_genome()
    
    net = nn.create_feed_forward_phenotype(winner)
    p_train = net.array_activate(X_train[features].values)
    p_valid = net.array_activate(X_valid[features].values)
    
    score_train = sklearn.metrics.log_loss(y_train, p_train[:,0])
    score_valid = sklearn.metrics.log_loss(y_valid, p_valid[:,0])
    print("Score based on training data set = ", score_train)
    print("Score based on validating data set = ", score_valid)

    
    # Visualize the winner network and plot statistics.
    visualize.plot_stats(pop.statistics)
    visualize.plot_species(pop.statistics)
    visualize.draw_net(winner, view=True)
    
    print("## Predicting test data")
    preds = net.array_activate(test[features].values)
    test[test_col_name] = preds
    test[[id_col_name,test_col_name]].to_csv("../predictions/pred_" + timestr + ".csv", index=False)
    

    

if __name__ == '__main__':

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print("########################## Start Time Stamp ==== " + timestamp)
    print("## Loading Data")
    models_predictions_file = "../predictions/models_predictions.csv"
    train = pd.read_csv('../inputs/train.csv')
    test = pd.read_csv('../inputs/test.csv')


    if os.path.isfile(models_predictions_file):
        models_predictions = pd.read_csv(models_predictions_file)
    else:
        models_predictions = pd.DataFrame()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print("########################## Time Stamp ==== " + timestamp)
    print("## Data Processing")
    train = train.drop(id_col_name, axis=1)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print("########################## Time Stamp ==== " + timestamp)
    print("## Data Encoding")
    for f in train.columns:
        if train[f].dtype=='object':
            print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[f].values) + list(test[f].values))
            train[f] = lbl.transform(list(train[f].values))
            test[f] = lbl.transform(list(test[f].values))

    features = [s for s in train.columns.ravel().tolist() if s != output_col_name]
    print("Features: ", features)
    
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(train[features])
    train[features] = imp.transform(train[features])
    test[features] = imp.transform(test[features])

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print("########################## Time Stamp ==== " + timestamp)
    print("## Training")
    numPos = len(train[train[output_col_name] == 1])
    numNeg = len(train[train[output_col_name] == 0])
    scaleRatio = float(numNeg) / float(numPos)
    print("Number of postive " + str(numPos) + " , Number of negative " + str(numNeg) + " , Ratio Negative to Postive : " , str(scaleRatio))
    
    test_size = 0.05
    X_pos = train[train[output_col_name] == 1]
    X_neg = train[train[output_col_name] == 0]
    
    X_train_pos, X_valid_pos = train_test_split(X_pos, test_size=test_size)
    X_train_neg, X_valid_neg = train_test_split(X_neg, test_size=test_size)
    
    X_train = pd.concat([X_train_pos,X_train_neg])
    X_valid = pd.concat([X_valid_pos,X_valid_neg])
    
    X_train = X_train.iloc[np.random.permutation(len(X_train))]
    X_valid = X_valid.iloc[np.random.permutation(len(X_valid))]
    
    y_train = X_train[output_col_name]
    y_valid = X_valid[output_col_name]
    
    num_generations = 1000
    train_model(features,num_generations)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print("########################## End Time Stamp ==== " + timestamp)

