# for numerical computing
import numpy as np

# for dataframes
import pandas as pd

# for easier visualization
import seaborn as sns

# for visualization and to display plots
from matplotlib import pyplot as plt

# import color maps
from matplotlib.colors import ListedColormap

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

from math import sqrt

# to split train and test set
from sklearn.model_selection import train_test_split

# to perform hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import cross_val_score

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix

from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection.cross_validate import train_test_split
#from sklearn.model_selection.cross_validate import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
#import xgboost
import os
mingw_path = '/home/eddie/Desktop/python/Liver_disease_Machine_learning'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
from xgboost import XGBClassifier
from xgboost import plot_importance  # to plot feature importance

# to save the final model on disk
from sklearn.externals import joblib

df=pd.read_csv('indian_liver_patient.csv')

print(df.head())

print(df.shape)


#Distribution of Numerical Features

# Plot histogram grid
df.hist(figsize=(15,15), xrot=-45, bins=10) ## Display the labels rotated by 45 degress

# Clear the text "residue"
plt.show()


print(df.describe())

## if score==negative, mark 0 ;else 1 
def partition(x):
    if x == 2:
        return 0
    return 1

df['Dataset'] = df['Dataset'].map(partition)

#Distribution of categorical data

df.describe(include=['object'])


#Bar plots for categorical Features
plt.figure(figsize=(5,5))
sns.countplot(y='Gender', data=df)
plt.show()

print(df[df['Gender'] == 'Male'][['Dataset', 'Gender']].head())

sns.factorplot (x="Age", y="Gender", hue="Dataset", data=df);
plt.show()
#Age seems to be a factor for liver disease for both male and female genders


sns.countplot(data=df, x = 'Gender', label='Count')
plt.show()
M, F = df['Gender'].value_counts()
print('Number of patients that are male: ',M)
print('Number of patients that are female: ',F)

## Label Male as 0 and Female as 1
def partition(x):
    if x =='Male':
        return 0
    return 1

df['Gender'] = df['Gender'].map(partition)

#2-D Scatter Plot


sns.set_style('whitegrid')   ## Background Grid
sns.FacetGrid(df, hue = 'Dataset', size = 5).map(plt.scatter, 'Total_Bilirubin', 'Direct_Bilirubin').add_legend()
plt.savefig('Total_BilirubinVSDirect_Bilirubin')

sns.set_style('whitegrid')   ## Background Grid
sns.FacetGrid(df, hue = 'Dataset', size = 5).map(plt.scatter, 'Total_Bilirubin', 'Albumin').add_legend()
plt.savefig('Total_BilirubinVSAlbumin')

sns.set_style('whitegrid')   ## Background Grid
sns.FacetGrid(df, hue = 'Dataset', size = 5).map(plt.scatter, 'Total_Protiens', 'Albumin_and_Globulin_Ratio').add_legend()
plt.savefig('Total_ProtiensVSAlbumin_and_Globulin_Ratio')

"""

Correlations

    Finally, let's take a look at the relationships between numeric features and other numeric features.
    Correlation is a value between -1 and 1 that represents how closely values for two separate features move in unison.
    Positive correlation means that as one feature increases, the other increases; eg. a child's age and her height.
    Negative correlation means that as one feature increases, the other decreases; eg. hours spent studying and number of parties attended.
    Correlations near -1 or 1 indicate a strong relationship.
    Those closer to 0 indicate a weak relationship.
    0 indicates no relationship.
"""

print(df.corr())

plt.figure(figsize=(10,10))
sns.heatmap(df.corr())
plt.savefig("HeatMap")

mask=np.zeros_like(df.corr())
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(10,10))
with sns.axes_style("white"):
    ax = sns.heatmap(df.corr()*100, mask=mask, fmt='.0f', annot=True, lw=1, cmap=ListedColormap(['green', 'yellow', 'red','blue']))

plt.savefig("Data")


#Data Cleaning

df = df.drop_duplicates()
print( df.shape )

#There were 13 duplicates
#Removing Outliers

sns.boxplot(df.Aspartate_Aminotransferase)
plt.savefig("RemovedOutliers")


df.Aspartate_Aminotransferase.sort_values(ascending=False).head()

df = df[df.Aspartate_Aminotransferase <=2500 ]
print(df.shape)


df.isnull().values.any()


df=df.dropna(how='any')

print(df.shape)

print(df.head())

# Create separate object for target variable
y = df.Dataset

# Create separate object for input features
X = df.drop('Dataset', axis=1)

# Split X and y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=1234,
                                                    stratify=df.Dataset)


# Print number of observations in X_train, X_test, y_train, and y_test
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


"""
Data standardization

    In Data Standardization we perform zero mean centring and unit scaling; i.e. we make the mean of all the features as zero and the standard deviation as 1.
    Thus we use mean and standard deviation of each feature.
    It is very important to save the mean and standard deviation for each of the feature from the training set, because we use the same mean and standard deviation in the test set.
"""


train_mean = X_train.mean()
train_std = X_train.std()



## Standardize the train data set
X_train = (X_train - train_mean) / train_std


## Check for mean and std dev.
print(X_train.describe())


## Note: We use train_mean and train_std_dev to standardize test data set
X_test = (X_test - train_mean) / train_std

## Check for mean and std dev. - not exactly 0 and 1
print(X_test.describe())


#Model-1 Logistic Regression

tuned_params = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'penalty': ['l1', 'l2']}
model = GridSearchCV(LogisticRegression(), tuned_params, scoring = 'roc_auc', n_jobs=-1)
model.fit(X_train, y_train)


model.best_estimator_

## Predict Train set results
y_train_pred = model.predict(X_train)


## Predict Test set results
y_pred = model.predict(X_test)


# Get just the prediction for the positive class (1)
y_pred_proba = model.predict_proba(X_test)[:,1]


# Display first 10 predictions
y_pred_proba[:10]


i=28  ## Change the value of i to get the details of any point (56, 213, etc.)
print('For test point {}, actual class = {}, precited class = {}, predicted probability = {}'.
      format(i, y_test.iloc[i], y_pred[i], y_pred_proba[i]))


confusion_matrix(y_test, y_pred).T

# Calculate ROC curve from y_test and pred
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)


# Plot the ROC curve
fig = plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')

# Plot ROC curve
plt.plot(fpr, tpr, label='l1')
plt.legend(loc='lower right')

# Diagonal 45 degree line
plt.plot([0,1],[0,1],'k--')

# Axes limits and labels
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# Calculate AUC for Train set
print(roc_auc_score(y_train, y_train_pred))

#0.581751737359

# Calculate AUC for Test set
print(auc(fpr, tpr))

#0.710262345679

#Feature Importance

## Building the model again with the best hyperparameters
model = LogisticRegression(C=1, penalty = 'l2')
model.fit(X_train, y_train)

indices = np.argsort(-abs(model.coef_[0,:]))
print("The features in order of importance are:")
print(50*'-')
for feature in X.columns[indices]:
    print(feature)
"""
The features in order of importance are:
--------------------------------------------------
Alamine_Aminotransferase
Direct_Bilirubin
Aspartate_Aminotransferase
Albumin
Total_Protiens
Total_Bilirubin
Alkaline_Phosphotase
Age
Gender
Albumin_and_Globulin_Ratio   
""" 



"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Second Model for Random Forest"

tuned_params = {'n_estimators': [100, 200, 300, 400, 500], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
model = RandomizedSearchCV(RandomForestClassifier(), tuned_params, n_iter=15, scoring = 'roc_auc', n_jobs=-1)
model.fit(X_train, y_train)

model.best_estimator_

y_train_pred = model.predict(X_train)

y_pred = model.predict(X_test)

# Get just the prediction for the positive class (1)
y_pred_proba = model.predict_proba(X_test)[:,1]



# Display first 10 predictions
y_pred_proba[:10]


confusion_matrix(y_test, y_pred).T

# Calculate ROC curve from y_test and pred
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)


# Plot the ROC curve
fig = plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')

# Plot ROC curve
plt.plot(fpr, tpr, label='l1')
plt.legend(loc='lower right')

# Diagonal 45 degree line
plt.plot([0,1],[0,1],'k--')

# Axes limits and labels
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("TruePositiveVsFalsePositiveRandomForest")
#plt.show()

#Calculate AUC for Train set
roc_auc_score(y_train, y_train_pred)

#0.88846153846153852


# Calculate AUC for Test set
print(auc(fpr, tpr))

#0.726466049383


#Feature Importance

## Building the model again with the best hyperparameters
model = RandomForestClassifier(n_estimators=500, min_samples_split=2, min_samples_leaf=4)
model.fit(X_train, y_train)

indices = np.argsort(-model.feature_importances_)
print("The features in order of importance are:")
print(50*'-')
for feature in X.columns[indices]:
    print(feature)
"""
he features in order of importance are:
--------------------------------------------------
Alkaline_Phosphotase
Total_Bilirubin
Aspartate_Aminotransferase
Alamine_Aminotransferase
Direct_Bilirubin
Age
Albumin
Albumin_and_Globulin_Ratio
Total_Protiens
Gender"""


#Model-3 XGBoost

tuned_params = {'max_depth': [1, 2, 3, 4, 5], 'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [100, 200, 300, 400, 500], 'reg_lambda': [0.001, 0.1, 1.0, 10.0, 100.0]}
model = RandomizedSearchCV(XGBClassifier(), tuned_params, n_iter=15, scoring = 'roc_auc', n_jobs=-1)
model.fit(X_train, y_train)

model.best_estimator_

y_train_pred = model.predict(X_train)

y_pred = model.predict(X_test)

# Get just the prediction for the positive class (1)
y_pred_proba = model.predict_proba(X_test)[:,1]


# Display first 10 predictions
y_pred_proba[:10]


confusion_matrix(y_test, y_pred).T

# Calculate ROC curve from y_test and pred
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)


# Plot the ROC curve
fig = plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')

# Plot ROC curve
plt.plot(fpr, tpr, label='l1')
plt.legend(loc='lower right')

# Diagonal 45 degree line
plt.plot([0,1],[0,1],'k--')

# Axes limits and labels
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("TruePositiveVsFalsePositive_Model-3_XGBoost")
#plt.show()

# Calculate AUC for Train
roc_auc_score(y_train, y_train_pred)

#0.68413611310807565

# Calculate AUC for Test
print(auc(fpr, tpr))

0.715470679012

#Feature Importance

model = XGBClassifier(max_depth=1,learning_rate=0.05,n_estimators=500, reg_lambda=1)
model.fit(X_train, y_train)

def my_plot_importance(booster, figsize, **kwargs): 
    from matplotlib import pyplot as plt
    from xgboost import plot_importance
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax, **kwargs)


my_plot_importance(model, (10,10))
plt.savefig("FScoreVSFeatures")

#Model4 - KNN

# creating odd list of K for KNN
neighbors = list(range(1,20,2))
# empty list that will hold cv scores
cv_scores = []

#  10-fold cross validation , 9 datapoints will be considered for training and 1 for cross validation (turn by turn) to determine value of k
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())   

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print('\nThe optimal number of neighbors is %d.' % optimal_k)

#The optimal number of neighbors is 17.


MSE.index(min(MSE))

#plot misclassification error vs k 
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.savefig("Number of Neighbors K_VS_Misclassification Error")
#plt.show()

classifier = KNeighborsClassifier(n_neighbors = optimal_k)
classifier.fit(X_train, y_train)



KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=17, p=2,
           weights='uniform')



y_pred = classifier.predict(X_test)



y_train_pred = classifier.predict(X_train)



acc = accuracy_score(y_test, y_pred, normalize=True) * float(100)  ## get the accuracy on testing data
acc



cnf=confusion_matrix(y_test,y_pred).T
cnf

# Get just the prediction for the positive class (1)
y_pred_proba = classifier.predict_proba(X_test)[:,1]


# Display first 10 predictions
y_pred_proba[:10]



# Calculate ROC curve from y_test and pred
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Plot the ROC curve
fig = plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')

# Plot ROC curve
plt.plot(fpr, tpr, label='l1')
plt.legend(loc='lower right')

# Diagonal 45 degree line
plt.plot([0,1],[0,1],'k--')

# Axes limits and labels
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("TruePositiveVSFalsePosotive_KNN")
#plt.show()

# Calculate AUC for Train
roc_auc_score(y_train, y_train_pred)

# Calculate AUC for Test
print(auc(fpr, tpr))


#Model-5 Descision Trees

tuned_params = {'min_samples_split': [2, 3, 4, 5, 7], 'min_samples_leaf': [1, 2, 3, 4, 6], 'max_depth': [2, 3, 4, 5, 6, 7]}
model = RandomizedSearchCV(DecisionTreeClassifier(), tuned_params, n_iter=15, scoring = 'roc_auc', n_jobs=-1)
model.fit(X_train, y_train)

model.best_estimator_


y_train_pred = model.predict(X_train)

y_pred = model.predict(X_test)

y_pred_proba = model.predict_proba(X_test)[:,1]

y_pred_proba[:10]

confusion_matrix(y_test, y_pred).T

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)


# Plot the ROC curve
fig = plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')

# Plot ROC curve
plt.plot(fpr, tpr, label='l1')
plt.legend(loc='lower right')

# Diagonal 45 degree line
plt.plot([0,1],[0,1],'k--')

# Axes limits and labels
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("TruePositveVsFalsePositive_Model-5_DescisionTrees")
#plt.show()


# Calculate AUC for Train
roc_auc_score(y_train, y_train_pred)

print(auc(fpr, tpr))


#Feature Importance

## Building the model again with the best hyperparameters
model = DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=6, max_depth=4)
model.fit(X_train, y_train)

indices = np.argsort(-model.feature_importances_)
print("The features in order of importance are:")
print(50*'-')
for feature in X.columns[indices]:
    print(feature)

"""
The features in order of importance are:
--------------------------------------------------
Total_Bilirubin
Alkaline_Phosphotase
Age
Albumin_and_Globulin_Ratio
Albumin
Aspartate_Aminotransferase
Gender
Direct_Bilirubin
Alamine_Aminotransferase
Total_Protiens
"""

#Model-6 SVC

from sklearn import svm
def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X_train, y_train)
    grid_search.best_params_
    return grid_search.best_params_


svClassifier=SVC(kernel='rbf',probability=True)
svClassifier.fit(X_train,y_train)


svc_param_selection(X_train,y_train,5)


###### Building the model again with the best hyperparameters
model = SVC(C=1, gamma=1,probability=True)
model.fit(X_train, y_train)


## Predict Train results
y_train_pred = model.predict(X_train)


## Predict Test results
y_pred = model.predict(X_test)


confusion_matrix(y_test, y_pred).T


y_pred_proba = model.predict_proba(X_test)[:,1]


# Calculate ROC curve from y_test and pred
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)


# Plot the ROC curve
fig = plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')

# Plot ROC curve
plt.plot(fpr, tpr, label='l1')
plt.legend(loc='lower right')

# Diagonal 45 degree line
plt.plot([0,1],[0,1],'k--')

# Axes limits and labels
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("FalsePositiveVSTruePositive_SVC")
#plt.show()

#Calculate AUC for Train
roc_auc_score(y_train, y_train_pred)

print(auc(fpr, tpr))


#Model-7 Gradient Boosting

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier


#Import Library
from sklearn.ensemble import GradientBoostingClassifier
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create Gradient Boosting Classifier object
gbclass = GradientBoostingClassifier(
                    random_state = 1000,
                    verbose = 0,
                    n_estimators = 10,
                    learning_rate = 0.9,
                    loss = 'deviance',
                    max_depth = 3
                   )
#gbclass = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
# Train the model using the training sets and check score
gbclass.fit(X_train, y_train)
#Predict Output
predicted= gbclass.predict(X_test)

gbclass_score = round(gbclass.score(X_train, y_train) * 100, 2)
gbclass_score_test = round(gbclass.score(X_test, y_test) * 100, 2)
print('Score: \n', gbclass_score)
print('Test Score: \n', gbclass_score_test)
print('Accuracy: \n', accuracy_score(y_test,predicted))
print(confusion_matrix(predicted,y_test))
print(classification_report(y_test,predicted))
"""
Score: 
 90.02
Test Score: 
 69.03
Accuracy: 
 0.690265486726
[[13 16]
 [19 65]]
             precision    recall  f1-score   support

          0       0.45      0.41      0.43        32
          1       0.77      0.80      0.79        81

avg / total       0.68      0.69      0.69       113
"""

## Predict Train results
y_train_pred = gbclass.predict(X_train)


## Predict Test results
y_pred = gbclass.predict(X_test)


y_pred_proba = gbclass.predict_proba(X_test)[:,1]


# Calculate ROC curve from y_test and pred
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)


# Plot the ROC curve
fig = plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')

# Plot ROC curve
plt.plot(fpr, tpr, label='l1')
plt.legend(loc='lower right')

# Diagonal 45 degree line
plt.plot([0,1],[0,1],'k--')

# Axes limits and labels
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("FalsePOsitiveVsTruePositive_Gradient_Bossting")


roc_auc_score(y_train,y_train_pred )


# Calculate AUC for Test
print(auc(fpr, tpr))

0.707175925926

#Neural Networks

# Neural Networks# Neural 
neural = MLPClassifier(hidden_layer_sizes=40,
                     activation='relu',
                     solver='adam',
                     alpha=0.001,
                     batch_size='auto',
                     max_iter=200,
                     random_state=137,
                     tol=0.0001,
                     early_stopping=False,
                     validation_fraction=0.1,
                     beta_1=0.9,
                     beta_2=0.999,
                     epsilon=1e-08,
                     learning_rate='constant',
                     power_t=0.5,
                     momentum=0.8,
                     nesterovs_momentum=True,
                     shuffle=True,
                     learning_rate_init=0.001)
neural.fit(X_train, y_train)
#Predict Output
predicted = neural.predict(X_test)

neural_score = round(neural.score(X_train, y_train) * 100, 2)
neural_score_test = round(neural.score(X_test, y_test) * 100, 2)
print('Neural Score: \n', neural_score)
print('Neural Test Score: \n', neural_score_test)
print('Accuracy: \n', accuracy_score(y_test, predicted))
print(confusion_matrix(predicted,y_test))
print(classification_report(y_test,predicted))

"""
Neural Score: 
 77.83
Neural Test Score: 
 69.91
Accuracy: 
 0.699115044248
[[ 5  7]
 [27 74]]
             precision    recall  f1-score   support

          0       0.42      0.16      0.23        32
          1       0.73      0.91      0.81        81

avg / total       0.64      0.70      0.65       113

"""
## Predict Train results
y_train_pred = neural.predict(X_train)


## Predict Test results
y_pred = neural.predict(X_test)


y_pred_proba = neural.predict_proba(X_test)[:,1]


# Calculate ROC curve from y_test and pred
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)


# Plot the ROC curve
fig = plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')

# Plot ROC curve
plt.plot(fpr, tpr, label='l1')
plt.legend(loc='lower right')

# Diagonal 45 degree line
plt.plot([0,1],[0,1],'k--')

# Axes limits and labels
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("TruePositiveVsFalsePositive_NN")


roc_auc_score(y_train,y_train_pred )

# Calculate AUC for Test
print(auc(fpr, tpr))


win_model = XGBClassifier(max_depth=1,learning_rate=0.05,n_estimators=500, reg_lambda=1)
win_model.fit(X_train, y_train)
with open('LiverDisease.pkl', 'wb') as pickle_file:
      joblib.dump(win_model, 'LiverDisease.pkl')

