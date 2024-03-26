import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score, log_loss, r2_score
from sklearn.feature_selection import mutual_info_classif, f_classif

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import joblib


###########################
### BUILD DESIGN MATRIX ###
###########################

# Split data into 80% training set and 20% test set
def stratified_split(df):
    df = df.sample(1000000)
    X = df.drop('LABEL', axis=1)
    y = df['LABEL']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=df['LABEL'])
    return X_train, X_test, y_train, y_test


#########################
### FEATURE SELECTION ###
#########################

# Build a Random Forest classifier on the training data and save feature importance scores
def calculate_tree_based_importance_scores(feature_scores, X_train, y_train):
    print('Calculating RF Feature Importance Scores')
    m = RandomForestClassifier(max_depth=6, n_estimators=25)
    m.fit(X_train, y_train)
    feature_scores['rf_importance'] = m.feature_importances_
    return feature_scores

# Compute mutual information on the training data
def calculate_mutual_info(feature_scores, X_train, y_train):
    print('Calculating Mutual Information')
    feature_scores['mutual_info'] = mutual_info_classif(X_train, y_train)
    return feature_scores

# Compute ANOVA scores on the training data
def calculate_anova(feature_scores, X_train, y_train):
    print('Calculating ANOVA Scores')
    feature_scores['anova'] = f_classif(X_train, y_train)[0]
    return feature_scores

# Mark unimportant features for removal in order to control training time and overfitting
def get_feature_scores(X_train, y_train):
    feature_scores = pd.DataFrame(index=X_train.columns) # Initialize empty results dataframe
    scaler = StandardScaler()
    X_train_standardized = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns) # Standardize features to 0 mean and unit variance
    # Calculate feature importance using each method
    feature_scores = calculate_tree_based_importance_scores(feature_scores, X_train_standardized, y_train)
    feature_scores = calculate_mutual_info(feature_scores, X_train_standardized, y_train)
    feature_scores = calculate_anova(feature_scores, X_train_standardized, y_train)
    for col in feature_scores.columns:
        # Standardize importance scores so that they can be compared on a level playing field
        feature_scores[col] = scaler.fit_transform(feature_scores[[col]]) # Zero mean and unit variance
        feature_scores[col] = (feature_scores[col] - feature_scores[col].min()) / (feature_scores[col].max() - feature_scores[col].min()) # 0-1 scale
    feature_scores['avg_score'] = feature_scores.mean(axis=1)
    # If a feature scores never reaches an importance that is 10% of the top feature along any of the three methods, mark it for removal
    feature_scores.loc[feature_scores.drop('avg_score', axis=1).max(axis=1) < 0.1, 'remove'] = 1
    feature_scores['remove'] = feature_scores['remove'].fillna(0)
    feature_scores = feature_scores.sort_values(['remove', 'avg_score'], ascending=[True, False])
    print(feature_scores)
    return feature_scores

# Pare features based on importance scores, making sure not to use more features than minority labels
def pare_features(feature_scores, X):
    print('Paring to top {} features'.format((feature_scores.shape[0] - feature_scores['remove'].sum()).astype(int)))
    pared_feature_set = feature_scores[feature_scores['remove']!=1].index
    return X[pared_feature_set]


###################
### TRAIN MODEL ###
###################

# Define a set of classifiers and hyperparameters to search over when training the model
search_space = {
    'LogisticRegression': {
        'estimator': Pipeline(steps=[('scaler', StandardScaler()), ('clf', LogisticRegression())]),
        'param_grid': {
            'clf__class_weight': [None, 'balanced']
        }
    },
    'AdaBoost': {
        'estimator': Pipeline(steps=[('scaler', StandardScaler()), ('clf', AdaBoostClassifier())]),
        'param_grid': {
            'clf__estimator': [DecisionTreeClassifier(max_depth=3)], # LogisticRegression()
            'clf__n_estimators': [50] # 25
        }
    },
    'RandomForest': {
        'estimator': Pipeline(steps=[('scaler', StandardScaler()), ('clf', RandomForestClassifier())]),
        'param_grid': {
            'clf__max_depth': [6, 8], # 4
            'clf__n_estimators': [50], # 25
            'clf__class_weight': [None, 'balanced']
        }
    },
    'XGB': {
        'estimator': Pipeline(steps=[('scaler', StandardScaler()), ('clf', XGBClassifier())]),
        'param_grid': {
            'clf__max_depth': [4, 6, 8],
            'clf__n_estimators': [25, 50]
        }
    },
    'Gaussian NB': {
        'estimator': Pipeline(steps=[('scaler', StandardScaler()), ('clf', GaussianNB())]),
        'param_grid': {
            'clf__var_smoothing': np.logspace(0,-9, num=10)
        }
    },
    'MLP': {
        'estimator': Pipeline(steps=[('scaler', StandardScaler()), ('clf', MLPClassifier())]),
        'param_grid': {
            'clf__hidden_layer_sizes': [(25, 15)], # (25,)
            'clf__activation': ['tanh'], # relu
            'clf__solver': ['sgd'] # adam
        }
    }
}

# Train N models defined by search space, and pick the winning model based on 4-fold cross validation score on the training set
def train_models(search_space, X_train_fit, y_train):
    winning_grid = {'grid_name': None, 'grid': None, 'score': 0}
    for g in search_space:
        print('------------------------------------------------------------------------------')
        print('Training {}'.format(g))
        print('------------------------------------------------------------------------------')
        grid = GridSearchCV(
            estimator = search_space[g]['estimator'],
            param_grid = search_space[g]['param_grid'],
            verbose = 3,
            scoring = 'roc_auc',
            cv=4)
        grid.fit(X_train_fit, y_train)
        print('Finished training')
        print('  Best Params: {}'.format(grid.best_estimator_))
        print('  Best Score: {}'.format(grid.best_score_))
        search_space[g]['fitted_grid'] = grid
        if grid.best_score_ > winning_grid['score']:
            winning_grid['grid_name'] = g
            winning_grid['grid'] = grid
            winning_grid['score'] = grid.best_score_
    print('------------------------------------------------------------------------------')
    print('Winning grid: {} with score {} and params {}'.format(winning_grid['grid_name'], winning_grid['score'], winning_grid['grid'].best_params_))
    print('------------------------------------------------------------------------------')
    return winning_grid


###############
### PREDICT ###
###############

def predict(winning_grid, X):
    m = winning_grid['grid']
    preds = pd.DataFrame(m.predict_proba(X)[:, 1], index=X.index, columns=['SCORE'])
    return preds


########################
### CALIBRATE SCORES ###
########################

# Place observations into bins based on the percentile rank of their predictions
def bin_predictions(eval_df, num_bins=100):
    eval_df['BIN'] = pd.qcut(eval_df['SCORE'], num_bins, labels=False, duplicates='drop')
    return eval_df

# Compute the actual class balance in each bin (i.e. the likelihood that an observation in that bin is in fact a true positive)
def compute_class_balance_per_bin(eval_df):
    prediction_bin_count = eval_df.groupby('BIN')['SCORE'].count()._set_name('COUNT')
    prediction_bin_avg_score = eval_df.groupby('BIN')['SCORE'].mean()._set_name('AVG_SCORE').astype(float)
    prediction_bin_class_balance = eval_df.groupby('BIN')['LABEL'].mean()._set_name('AVG_LABEL').astype(float)
    prediction_bin_df = pd.DataFrame(prediction_bin_count).join(prediction_bin_avg_score).join(prediction_bin_class_balance).reset_index()
    return prediction_bin_df

# Define curve fit functions
def linear_fit(X, a, b):
    return a * X + b

def log_fit(X, a, b):
    return a * np.log1p(X) + b

def exponential_fit(X, a, b):
    return a * np.exp(-b * X)

def sigmoid_fit(X, a, b):
    return 1.0 / (1 + np.exp(-a * (X - b)))

def arctan_fit(X, a, b):
    return np.arctan(X) * a + b

# Fit a series of monotonic curves to map avg. score in bin to true class probability
def fit_curves(bins):
    curve_functions = [exponential_fit, log_fit, linear_fit, sigmoid_fit, arctan_fit]
    best_fit = {'function': None, 'name': None, 'R2': 0, 'params': None}
    for cf in curve_functions:
        try:
            fit_params, _ = curve_fit(cf, bins['AVG_SCORE'], bins['AVG_LABEL'])
        except RuntimeError:
            print('Unable to fit {} curve'.format(cf.__name__))
            continue
        fit_scores = cf(bins['AVG_SCORE'], fit_params[0], fit_params[1])
        fit_r2 = r2_score(bins['AVG_LABEL'], fit_scores)
        print('Fit {} curve with R2 score {}'.format(cf.__name__, fit_r2))
        if fit_r2 > best_fit['R2']:
            best_fit['function'] = cf
            best_fit['name'] = cf.__name__
            best_fit['R2'] = fit_r2
            best_fit['params'] = [fit_params[0], fit_params[1]]
    print('------------------------------------------------------------------------------')
    print('Winning curve: {} with score {}'.format(best_fit['name'], best_fit['R2']))
    print('------------------------------------------------------------------------------')
    return best_fit

# Apply the best fitting curve to map score to probability
def apply_best_curve(df, best_fit, col_to_transform):
    df_with_probs = df.copy()
    df_with_probs['FITTED_PROBABILITY'] = df_with_probs[col_to_transform].apply(lambda x: best_fit['function'](x, best_fit['params'][0], best_fit['params'][1]))
    df_with_probs['FITTED_PROBABILITY'] = df_with_probs['FITTED_PROBABILITY'].apply(lambda x: max(min(x, 1), 0)) # Make sure the probabilities span a 0-1 range
    return df_with_probs

# Plot calibration curves
def plot_calibration(bins):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # Create subplots
    # Plot calibration curve for Classifier Scores
    axs[0].plot(bins['AVG_LABEL'], bins['AVG_SCORE'], marker='o', linestyle='-', label='Classifier Scores')
    axs[0].plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line for reference
    axs[0].set_xlabel('Mean Predicted Probability')
    axs[0].set_ylabel('True Fraction of Positives')
    axs[0].set_title('Calibration Curve - Classifier Scores')
    axs[0].grid(True)
    # Plot calibration curve for Fitted Probabilities
    axs[1].plot(bins['AVG_LABEL'], bins['FITTED_PROBABILITY'], marker='o', linestyle='-', label='Fitted Probabilities')
    axs[1].plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line for reference
    axs[1].set_xlabel('Mean Predicted Probability')
    axs[1].set_ylabel('True Fraction of Positives')
    axs[1].set_title('Calibration Curve - Fitted Probabilities')
    axs[1].grid(True)
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
    return


################
### EVALUATE ###
################

def get_ranking_metrics(metrics, eval_df):
    metrics['ROC-AUC'] = roc_auc_score(eval_df['LABEL'], eval_df['FITTED_PROBABILITY']) # ROC AUC
    return metrics

def get_threshold_metrics(metrics, eval_df):
    # Choose the threshold where precision = recall (i.e. where the percent of positive predictions equals the percent of positive labels)
    positive_labels = eval_df['LABEL'].sum()
    eval_df['RANK'] = eval_df['FITTED_PROBABILITY'].rank(method='min', ascending=False)
    eval_df.loc[eval_df['RANK'] <= positive_labels, 'PREDICTED_CLASS'] = 1
    eval_df.loc[eval_df['RANK'] > positive_labels, 'PREDICTED_CLASS'] = 0
    # Compute metrics
    metrics['Base Probability'] = eval_df['LABEL'].mean()
    metrics['Precision'] = eval_df[eval_df['PREDICTED_CLASS']==1]['LABEL'].mean()
    metrics['Recall'] = eval_df[eval_df['LABEL']==1]['PREDICTED_CLASS'].mean()
    metrics['Accuracy'] = (eval_df['LABEL'] == eval_df['PREDICTED_CLASS']).mean()
    return metrics

def get_probability_metrics(metrics, eval_df):
    metrics['LogLoss'] = log_loss(eval_df['LABEL'], eval_df['FITTED_PROBABILITY'])
    return metrics

def evaluate_metrics(eval_df):
    metrics = {} # Initialize empty results dictionary
    metrics = get_ranking_metrics({metrics}, eval_df)
    metrics = get_threshold_metrics(metrics, eval_df)
    metrics = get_probability_metrics(metrics, eval_df)
    print(metrics)
    return metrics

def analyze_top_features(feature_scores, eval_df, features):
    eval_df_feats = eval_df.join(features)
    eval_df_feats['DECILE'] = pd.qcut(eval_df_feats['FITTED_PROBABILITY'].rank(method='first'), 10, labels=False, duplicates='drop')
    print('\nAvg feature value by decile')
    feats_to_show = [c for c in list(feature_scores.index) + ['LABEL'] if c in eval_df_feats.columns]
    print(eval_df_feats.groupby('DECILE')[feats_to_show].mean().transpose())
    return feats_to_show


###########
### RUN ###
###########

# Load raw data
df1 = pd.read_csv('~/Downloads/OntimeVoyager/final_cleaned_data_2014-2018.csv').sample(frac=0.02)
df2 = pd.read_csv('~/Downloads/OntimeVoyager/final_cleaned_data_2009-2013.csv').sample(frac=0.02)
raw_df = pd.concat([df1, df2]).reset_index(drop=True)

# Transform  engineering
raw_df['FL_DATE'] = pd.to_datetime(raw_df['FL_DATE'], format = '%Y-%m-%d', errors = 'coerce')
raw_df['CRS_DEP_TIME'] = pd.to_datetime(raw_df['CRS_DEP_TIME'], format = '%H%M', errors = 'coerce')
raw_df['CRS_ARR_TIME'] = pd.to_datetime(raw_df['CRS_ARR_TIME'], format = '%H%M', errors = 'coerce')
raw_df['DISTANCE'] = raw_df['DISTANCE'].astype('float')
raw_df['CRS_ELAPSED_TIME'] = raw_df['CRS_ELAPSED_TIME'].astype('float')
raw_df['ARR_DELAY'] = raw_df['ARR_DELAY'].astype('float')
raw_df['CANCELLED'] = raw_df['CANCELLED'].astype('int')
raw_df['FL_YR'] = raw_df['FL_DATE'].dt.year.astype(int)
raw_df['FL_MONTH'] = raw_df['FL_DATE'].dt.month
raw_df['FL_DAY'] = raw_df['FL_DATE'].dt.day
raw_df['FL_DOW'] = raw_df['FL_DATE'].dt.dayofweek # monday=0, sunday=6
raw_df['CRS_DEP_HR'] = raw_df['CRS_DEP_TIME'].dt.hour
raw_df['CRS_DEP_MIN'] = raw_df['CRS_DEP_TIME'].dt.minute
raw_df['CRS_ARR_HR'] = raw_df['CRS_ARR_TIME'].dt.hour
raw_df['CRS_ARR_MIN'] = raw_df['CRS_ARR_TIME'].dt.minute

# Encode categories
origin_delays_by_year = raw_df.groupby(['ORIGIN', 'FL_YR'])['ARR_DELAY'].mean().reset_index().rename({'ARR_DELAY': 'AVG_ORIGIN_DELAY_LAST_YR'}, axis=1)
origin_delays_by_year['FL_YR'] = origin_delays_by_year['FL_YR'] + 1

dest_delays_by_year = raw_df.groupby(['DEST', 'FL_YR'])['ARR_DELAY'].mean().reset_index().rename({'ARR_DELAY': 'AVG_DEST_DELAY_LAST_YR'}, axis=1)
dest_delays_by_year['FL_YR'] = dest_delays_by_year['FL_YR'] + 1

carrier_delays_by_year = raw_df.groupby(['OP_CARRIER', 'FL_YR'])['ARR_DELAY'].mean().reset_index().rename({'ARR_DELAY': 'AVG_CARRIER_DELAY_LAST_YR'}, axis=1)
carrier_delays_by_year['FL_YR'] = carrier_delays_by_year['FL_YR'] + 1

raw_df = raw_df.merge(origin_delays_by_year, on=['ORIGIN', 'FL_YR'], how='left').set_index(raw_df.index).drop('ORIGIN', axis=1)
raw_df = raw_df.merge(dest_delays_by_year, on=['DEST', 'FL_YR'], how='left').set_index(raw_df.index).drop('DEST', axis=1)
raw_df = raw_df.merge(carrier_delays_by_year, on=['OP_CARRIER', 'FL_YR'], how='left').set_index(raw_df.index).drop('OP_CARRIER', axis=1)

# Build labels
raw_df['DELAY15'] = (raw_df['ARR_DELAY'] >= 15).astype(int)
raw_df['DELAY30'] = (raw_df['ARR_DELAY'] >= 30).astype(int)
raw_df['DELAY60'] = (raw_df['ARR_DELAY'] >= 60).astype(int)
raw_df['DELAY120'] = (raw_df['ARR_DELAY'] >= 120).astype(int)

# Cleanup
raw_df = raw_df.drop(['FL_DATE', 'CRS_DEP_TIME', 'CRS_ARR_TIME', 'OP_CARRIER_FL_NUM', 'DEP_TIME', 'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN', 'ARR_TIME', 'AIR_TIME', 'ACTUAL_ELAPSED_TIME', 'DIVERTED', 'DEP_DELAY', 'CANCELLATION_CODE', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY'], axis=1)
raw_df['ARR_DELAY'] = raw_df['ARR_DELAY'].fillna(0)
raw_df = raw_df.dropna() # This will get rid of 2009 data since it's the first year in our dataset


# RUN
label_columns = ['DELAY15', 'DELAY30', 'DELAY60', 'DELAY120', 'CANCELLED']
curve_fit_params = {}

for label in label_columns:
    # Load
    df = raw_df.rename({label: 'LABEL'}, axis=1) # Standardize label name
    df = df.drop([c for c in label_columns + ['ARR_DELAY'] if c in df.columns], axis=1) # Drop non-feature columns
    # Split
    X_train, X_test, y_train, y_test = stratified_split(df)
    # Feature selection
    feature_scores = get_feature_scores(X_train, y_train)
    # X_train = pare_features(feature_scores, X_train)
    # X_test = pare_features(feature_scores, X_test)
    # Train
    winning_grid = train_models(search_space, X_train, y_train)
    # Predict
    test_predictions = predict(winning_grid, X_test)
    eval_df = bin_predictions(test_predictions.join(y_test), 100)
    # Calibrate
    binned_probs = compute_class_balance_per_bin(eval_df)
    best_fit = fit_curves(binned_probs)
    binned_probs_calibrated = apply_best_curve(binned_probs, best_fit, 'AVG_SCORE')
    # plot_calibration(binned_probs_calibrated)
    eval_df_probs = apply_best_curve(eval_df, best_fit, 'SCORE')
    eval_df_probs['FITTED_PROBABILITY'].min()
    eval_df_probs['FITTED_PROBABILITY'].max()
    # Evaluate
    metrics = evaluate_metrics(eval_df_probs)
    feature_deciles = analyze_top_features(feature_scores, eval_df_probs, X_test)
    # TODO: Save
    joblib.dump(winning_grid['grid'].best_estimator_, '/Users/michaelfirn/Downloads/OntimeVoyager/{}_.pkl'.format(label))
    curve_fit_params[label] = {}
    curve_fit_params[label]['curve'] = best_fit['name']
    curve_fit_params[label]['params'] = best_fit['params'] # NOTE: client will need to post-process distribution of probabilities for different delay thresholds, in order to make sure they decrease monotonically as the length of delay increases


# CURVE FIT PARAMS
# {
#     'DELAY15': {'curve': 'linear_fit', 'params': [1.0200384895733157, -0.0034693419947515736]}, 
#     'DELAY30': {'curve': 'arctan_fit', 'params': [1.0264286544344035, -0.0015754058169079599]}, 
#     'DELAY60': {'curve': 'log_fit', 'params': [1.010452480525234, 0.0019788609483877284]}, 
#     'DELAY120': {'curve': 'log_fit', 'params': [1.0505892352003179, -0.0006082483095044566]}, 
#     'CANCELLED': {'curve': 'linear_fit', 'params': [1.0813583901530917, -0.0009191259304315782]}
# }