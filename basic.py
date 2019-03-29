import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Courier_weekly_data = pd.read_csv("data/Courier_weekly_data.csv")
Courier_lifetime_data = pd.read_csv("data/Courier_lifetime_data.csv")

# Note: number of MVs
print(Courier_weekly_data.isnull().sum(axis=0))
print(Courier_lifetime_data.isnull().sum(axis=0))

Courier_weekly_data.info()

'''
Courier_weekly_data['feature_17'].hist(bins=40,log=True)
plt.show()

corr = Courier_weekly_data.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)

# Draw a graph with pandas and keep what's returned
ax = Courier_weekly_data.plot(kind='scatter', x='week', y='feature_7')
# Set the x scale because otherwise it goes into weird negative numbers
ax.set_xlim((-1,12))
# Set the x-axis label
ax.set_xlabel("Week")
# Set the y-axis label
ax.set_ylabel("Feature")
plt.show()

import seaborn
seaborn.pairplot(Courier_weekly_data,vars=['feature_1','feature_2','feature_3'],hue='week',palette="husl",diag_kind = 'kde')
plt.show()
'''

# Check the weather of categorical features, if some of them are needed to be assumed as continue or not.
print(Courier_weekly_data.describe())
print(Courier_weekly_data.dtypes)
cat_features = ['feature_1',
                'feature_2',
                'feature_3',
                'feature_11',
                'feature_16',
                'feature_17']
print(Courier_weekly_data.loc[:, cat_features].describe())

for feature_item in cat_features:
    print('The feature <{}> level size is {}'.format(feature_item, len(set(Courier_weekly_data.loc[:, feature_item]))))
    print(set(Courier_weekly_data.loc[:, feature_item]))

# Note: I decided to eliminate the {'feature_11','feature_16','feature_17'}.
cat_dropped_features = ['feature_11', 'feature_16', 'feature_17']
Courier_weekly_data = Courier_weekly_data.drop(columns=cat_dropped_features)

'''
Note: 
I treat the {'feature_1','feature_2','feature_3'} as continues.I think the {'feature_1','feature_2','feature_3'} are 
related to the number of the packages that are delivered by any person. 'feature_1' has to be 'feature_2' normalized.
So, i just keep the 'feature_1'.
'''
Courier_weekly_data = Courier_weekly_data.drop(columns=['feature_2'])

# Note: other categorical features are treated as float.
Courier_weekly_data = Courier_weekly_data.astype({"feature_1": float, "feature_3": float})

# Indexing the panel data
weeks = Courier_weekly_data['week']
couriers = Courier_weekly_data['courier']
Courier_weekly_data = Courier_weekly_data.drop(columns=['courier', 'week'])

# Outlier handling
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Define the PCA object
pca = PCA()

# Run PCA on scaled data and obtain the scores array
T = pca.fit_transform(StandardScaler().fit_transform(Courier_weekly_data))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

# Score plot of the first 2 PC
fig = plt.figure(figsize=(8, 6))
with plt.style.context(('ggplot')):
    plt.scatter(T[:, 0], T[:, 1], edgecolors='k', cmap='jet')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Score Plot')
plt.show()

# Mahalanobis distance for evaluating the overall distance of the outliers.
from sklearn.covariance import MinCovDet
import statistics

# fit a Minimum Covariance Determinant (MCD) robust estimator to data
robust_cov = MinCovDet().fit(T[:, :6])

# Get the Mahalanobis distance
m = robust_cov.mahalanobis(T[:, :6])
plt.hist(m, density=True, bins=50)
plt.ylabel('Probability')
plt.show()

colors = [plt.cm.jet(float(i) / max(m)) for i in m]
fig = plt.figure(figsize=(8, 6))
with plt.style.context(('ggplot')):
    plt.scatter(T[:, 0], T[:, 1], c=colors, edgecolors='k', s=40)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Score Plot')
plt.show()

# Note: there are some outliers obviously in the plot of the first two PCA components. They are needed to be clean.
print("m of sample is % s " % (statistics.mean(m)))
print("Standard Deviation of sample is % s " % (statistics.stdev(m)))
# Note: two standard deviation off the data is keep (from left).
filter_bond = statistics.stdev(m) * 2 + statistics.mean(m)
old_df_sahpe = Courier_weekly_data.shape

Courier_weekly_data['week'] = weeks
Courier_weekly_data['courier'] = couriers
Courier_weekly_data['Mahalanobis_distance'] = m
Courier_weekly_data = Courier_weekly_data.sort_values(by=['Mahalanobis_distance'])
Courier_weekly_data = Courier_weekly_data.loc[Courier_weekly_data['Mahalanobis_distance'] < filter_bond]
new_df_sahpe = Courier_weekly_data.shape
print('{} reacord <%{}> are deleted.'.format(old_df_sahpe[0] - new_df_sahpe[0], 100 * (old_df_sahpe[0] - new_df_sahpe[
    0]) / old_df_sahpe[0]))

# Indexing the panel data
week = Courier_weekly_data['week']
courier = Courier_weekly_data['courier']
Courier_weekly_data = Courier_weekly_data.drop(columns=['Mahalanobis_distance'])
Courier_weekly_data_clean = Courier_weekly_data
Courier_weekly_data = Courier_weekly_data.set_index(['courier', 'week']).sort_index()

# missing values
# Note: join tow table. The lifetime data is completed by the grouped mean of weekly data fro any courier.
Courier_weekly_data_mean = Courier_weekly_data.mean(level='courier', axis=0)
merged_df = pd.merge(Courier_lifetime_data, Courier_weekly_data_mean, how='left', on='courier')
merged_df = merged_df.set_index(['courier']).sort_index()
print(merged_df.shape)

print('=' * 40)
print('the number of missing values is:', merged_df[merged_df[
    'feature_2_life'].isnull()].shape[0])
print('=' * 40)
"""
There are two groups of incomplete instances in merged data set. 
    1- The ones have information about the courier in weekly table:
        in this case i use the regression for filling the missing value.
    2- The ones do not have information about the courier in weekly table: 
        The missing value is filled by mean of groups {a,b,c}
"""
# Note: slice the data in two mentioned groups.
# with information
merged_df_with_weekly_info = merged_df.loc[list(set(courier))]
print(merged_df_with_weekly_info.shape)

# check the response variable distribution.
plt.hist(merged_df['feature_2_life'].dropna(), bins=100)
plt.ylabel('Probability')
plt.xlim((-10, 100))
plt.ylim((0, 3000))
# Todo: plt.show()
# so it is close to poisson.

merged_df_with_weekly_info_feature_1_life = merged_df_with_weekly_info['feature_1_life']
merged_df_with_weekly_info = merged_df_with_weekly_info.drop(columns=['feature_1_life'])

merged_df_with_weekly_info_pred = merged_df_with_weekly_info[merged_df_with_weekly_info[
    'feature_2_life'].isnull()]
merged_df_with_weekly_info_train = merged_df_with_weekly_info.dropna(subset=['feature_2_life'])

print('-' * 50)
print('The Train data for missing value predication shape is:', merged_df_with_weekly_info_train.shape)
print('The predicable data for missing value predication shape is:', merged_df_with_weekly_info_pred.shape)
print('-' * 50)

# Note: The feature subset ['feature_1','feature_5','feature_9'] is omitted based on Z-test p_value (>0.05)
import statsmodels.api as sm

data_endog = merged_df_with_weekly_info_train['feature_2_life']
data_exog = merged_df_with_weekly_info_train.drop(columns=['feature_2_life', 'feature_1', 'feature_5', 'feature_9'])

pred_endog = []
pred_exog = merged_df_with_weekly_info_pred.drop(columns=['feature_2_life', 'feature_1', 'feature_5', 'feature_9'])

# Instantiate a gamma family model with the default link function.
Poisson_model = sm.GLM(data_endog, data_exog, family=sm.families.Poisson())
Poisson_results = Poisson_model.fit()
print('-' * 50)
print(Poisson_results.summary())
print(Poisson_results)
print('-' * 50)

print('-' * 50)
Poisson_model_pred = round(Poisson_results.predict(pred_exog))
merged_df_with_weekly_info_pred['feature_2_life'] = Poisson_model_pred

# Reconstruct the merged_df_with_weekly_info
merged_df_with_weekly_info = pd.concat([merged_df_with_weekly_info_pred,
                                        merged_df_with_weekly_info_train])
merged_df_with_weekly_info['feature_1_life'] = merged_df_with_weekly_info_feature_1_life
print(merged_df_with_weekly_info.shape)
print('-' * 50)

# without information
merged_df_without_weekly_info = merged_df.drop(list(set(courier)))
lifetime_imputed = pd.concat([merged_df_without_weekly_info[['feature_1_life', 'feature_2_life']],
                              merged_df_with_weekly_info[['feature_1_life', 'feature_2_life']]])

print('=' * 40)
print('The number of missing values is:', lifetime_imputed[lifetime_imputed[
    'feature_2_life'].isnull()].shape[0])
print('Equal to: %', 100 * lifetime_imputed[lifetime_imputed[
    'feature_2_life'].isnull()].shape[0] / lifetime_imputed.shape[0])
print('=' * 40)

# Note: the rest of them are imputed by the mean of any group

print(lifetime_imputed.groupby(['feature_1_life']).mean())
print(lifetime_imputed.groupby(['feature_1_life']).std())

'''
Note: The mean of different group (a,b,c,d) is very close nad the confidence interval is to wide.
      I prefer not to impute by mean of categories. It is not a good approach. 
      On the same hand, The data set is big enough for eliminating 11% of the data.
'''
lifetime_imputed = lifetime_imputed.dropna(subset=['feature_2_life'])

print('=' * 40)
print('The final shape of the lifetime data set is:', lifetime_imputed.shape)
print('=' * 40)

Courier_weekly_data_clean = pd.merge(Courier_weekly_data_clean, lifetime_imputed, how='left', on='courier')

# feature relation
merged_weekly_with_label = pd.merge(Courier_weekly_data, Courier_lifetime_data, how='left', on='courier')[
    'feature_1_life']

columns_item = Courier_weekly_data_clean.drop(columns=['feature_1_life'], axis=1).set_index(['courier', 'week']).columns

merged_weekly_with_label = np.where(merged_weekly_with_label == 'a', 'w', merged_weekly_with_label)
merged_weekly_with_label = np.where(merged_weekly_with_label == 'b', 'green', merged_weekly_with_label)
merged_weekly_with_label = np.where(merged_weekly_with_label == 'c', 'blue', merged_weekly_with_label)
merged_weekly_with_label = np.where(merged_weekly_with_label == 'd', 'yellow', merged_weekly_with_label)

pca = PCA()
T = pca.fit_transform(StandardScaler().fit_transform(
    Courier_weekly_data_clean.drop(columns=['feature_1_life'], axis=1).set_index(['courier', 'week']).sort_index()))


# Score plot of the first 2 PC


def myplot(score, coeff, labels=None, colored=True):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    fig = plt.figure(figsize=(8, 6))
    with plt.style.context(('ggplot')):
        if colored:
            plt.scatter(xs * scalex, ys * scaley, c=merged_weekly_with_label, edgecolors='k', cmap='jet')
            for i in range(n):
                plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='black', alpha=0.5)
                if labels is None:
                    plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i + 1), color='black', ha='center',
                             va='center')
                else:
                    plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='black', ha='center', va='center',
                             fontsize=14)
        else:
            plt.scatter(xs * scalex, ys * scaley, edgecolors='k', cmap='jet')
            for i in range(n):
                plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='black', alpha=0.5)
                if labels is None:
                    plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i + 1), color='black', ha='center',
                             va='center')
                else:
                    plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='black', ha='center', va='center')


plt.xlim(-0.75, 0.75)
plt.ylim(-0.75, 0.75)
plt.xlabel("PC{}".format(1))
plt.ylabel("PC{}".format(2))
plt.grid()
# Call the function. Use only the 2 PCs.
myplot(T[:, 0:2], np.transpose(pca.components_[0:2, :]), labels=columns_item)
plt.show()

# psc for summary data
merged_weekly_with_label = pd.merge(Courier_weekly_data_mean, lifetime_imputed, how='left', on='courier')[
    'feature_1_life']

merged_weekly_with_label = np.where(merged_weekly_with_label == 'a', 'red', merged_weekly_with_label)
merged_weekly_with_label = np.where(merged_weekly_with_label == 'b', 'green', merged_weekly_with_label)
merged_weekly_with_label = np.where(merged_weekly_with_label == 'c', 'blue', merged_weekly_with_label)
merged_weekly_with_label = np.where(merged_weekly_with_label == 'd', 'yellow', merged_weekly_with_label)

pca = PCA()
T = pca.fit_transform(StandardScaler().fit_transform(Courier_weekly_data_mean))
# Score plot of the first 2 PC

plt.xlim(-0.75, 0.75)
plt.ylim(-0.75, 0.75)
plt.xlabel("PC{}".format(1))
plt.ylabel("PC{}".format(2))
# Call the function. Use only the 2 PCs.
myplot(T[:, 0:2], np.transpose(pca.components_[0:2, :]), labels=Courier_weekly_data.columns, colored=True)
plt.show()

# ------------------------------- Modeling --------------------------
# prepare the data set

# Note: extract the '0/1' labels.
df_labs = Courier_weekly_data_clean[(Courier_weekly_data_clean.week > 8) & (Courier_weekly_data_clean.week < 12)][
    ['courier']]
df_labs = list(set(df_labs['courier']))

Courier_weekly_data_clean['label'] = 0
Courier_weekly_data_clean.loc[Courier_weekly_data_clean['courier'].isin(df_labs), 'label'] = 1

# Delete the 8,9,10,11
Courier_weekly_data_clean = Courier_weekly_data_clean[
    (Courier_weekly_data_clean.week < 7) | (Courier_weekly_data_clean.week > 12)]

print(Courier_weekly_data_clean.groupby(['label']).size())

from sklearn.utils import resample

# Separate majority and minority classes
df_majority = Courier_weekly_data_clean[Courier_weekly_data_clean.label == 1]
df_minority = Courier_weekly_data_clean[Courier_weekly_data_clean.label == 0]

# Up sample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,  # sample with replacement
                                 n_samples=1694,  # to match majority class
                                 random_state=4)  # reproducible results

# Combine majority class with up sampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

Courier_weekly_data_clean = df_upsampled.copy()

print('new class counts:')
print(print(df_upsampled.groupby(['label']).size()))

# ------------------------Modeling ---------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, mean_squared_error, average_precision_score, recall_score, \
    classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.formula.api as smf
import statsmodels.api as sm

# ------------------------------------ interaction investigation ---------------------------------
'''
from statsmodels.graphics.factorplots import interaction_plot

fig = interaction_plot(Courier_weekly_data_clean['week'],
                       Courier_weekly_data_clean['feature_1_life'],
                       Courier_weekly_data_clean['feature_2_life'],
                       ms=10)

plt.show()
'''
# ----------------------- ----------- panel data ------------------------------------------------

train, test = train_test_split(Courier_weekly_data_clean, test_size=0.25)

normal_ols = smf.ols(formula='label ~ feature_1+feature_3+feature_5+feature_6+feature_7+feature_8+'
                             'feature_10+feature_12+feature_13+feature_14+feature_15+C(week)+'
                             'C(feature_1_life)+feature_2_life',
                     data=train).fit()
print(normal_ols.summary())

print('*' * 50)
predictions = round(normal_ols.predict(test))
errors = abs(predictions - test['label'])

# result_accuracy.at[name, fs_item] = accuracy_score(predictions, y_test)
print('*' * 50)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
print('Accuracy:', accuracy_score(predictions, test['label']))
print('Normalized Accuracy:', accuracy_score(predictions, test['label'], normalize=False))
print('Mean squared error:', mean_squared_error(predictions, test['label']))
print('r:', classification_report(predictions, test['label']))
print('*' * 50)

# ----------------------- ----------- Logistic Reg. ------------------------------------------------

formula = 'label ~ feature_1+feature_3+feature_4+feature_5+feature_6+feature_7+feature_8+feature_12+' \
          'feature_14+feature_15+C(week)+C(feature_1_life)+feature_2_life'

# Instantiate a gamma family model with the default link function.
Poisson_model = smf.glm(formula=formula, data=train, family=sm.families.Binomial())
Poisson_results = Poisson_model.fit()
print('*' * 50)
print(Poisson_results.summary())
print('*' * 50)
predictions = round(Poisson_results.predict(test))
errors = abs(predictions - test['label'])

# result_accuracy.at[name, fs_item] = accuracy_score(predictions, y_test)
print('*' * 50)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
print('Accuracy:', accuracy_score(predictions, test['label']))
print('Normalized Accuracy:', accuracy_score(predictions, test['label'], normalize=False))
print('Mean squared error:', mean_squared_error(predictions, test['label']))
print('r:', classification_report(predictions, test['label']))
print('*' * 50)

input()
# ----------------------------------- model test ------------------------------------------------
from sklearn.kernel_ridge import KernelRidge

Courier_weekly_data_clean_ = Courier_weekly_data_clean.set_index(['courier', 'week']).sort_index()

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA", "LogisticRegression", "Kernel Ridge"]

classifiers = [
    KNeighborsClassifier(weights='distance', n_neighbors=50, algorithm='auto'),
    SVC(kernel="linear", C=0.025, class_weight='balanced', probability=True),
    SVC(class_weight='balanced', probability=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=10, n_estimators=30, max_features=5),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(solver='liblinear'),
    KernelRidge(alpha=3)]

df = Courier_weekly_data_clean

labs = df['label']
abcd = df['feature_1_life']

df = df.drop(['label', 'feature_1_life'], axis=1)

df_ordinal = df.copy(deep=True)

# -----------------------------------------------------------------------------------------------------------------
# use for classification
# -----------------------------------------------------------------------------------------------------------------
cols = df.columns

df = StandardScaler().fit_transform(df)
df = pd.DataFrame(data=df, columns=cols)

# Saving feature names for later use
feature_list = list(df.columns)

# Convert to numpy array
df = np.array(df)

# add inter action to model
poly = PolynomialFeatures(interaction_only=True, include_bias=True)
df = poly.fit_transform(df)

# Labels are the values we want to predict
labels = np.array(labs)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.25,
                                                    random_state=42)

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Validating Features Shape:', X_test.shape)
print('Validating Labels Shape:', y_test.shape)

from sklearn.model_selection import GridSearchCV

mlp = MLPClassifier(max_iter=20)
parameter_space = {
    'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
    'activation': ['tanh', 'relu', 'logistic'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive']}
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(X_train, y_train)

# Best hyper parameter set
print('Best parameters found:\n', clf.best_params_)

print('%'*80)

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 20, cv = 5, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model
rf_random.fit(X_train, y_train)
print(rf_random.best_params_)

input()

clf = RandomForestClassifier(max_depth=10, n_estimators=30, max_features=5)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
predictions = clf.predict(X_test)
predictions = np.around(np.array(predictions))

# iterate over classifiers
for name, clf in zip(names, classifiers):
    print('===============', name, '==================')
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    # ----------------------------validation---------------------
    predictions = clf.predict(X_test)
    predictions = np.around(np.array(predictions))

    # Calculate the absolute errors
    errors = abs(predictions - y_test)

    # result_accuracy.at[name, fs_item] = accuracy_score(predictions, y_test)
    print('-' * 50)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    print('Accuracy:', accuracy_score(predictions, y_test))
    print('Normalized Accuracy:', accuracy_score(predictions, y_test, normalize=False))
    print('Mean squared error:', mean_squared_error(predictions, y_test))
    print('average_precision:', average_precision_score(predictions, y_test))
    print('recall_score:', recall_score(predictions, y_test))
    print('recall_score:', classification_report(predictions, y_test))
    print('-' * 50)

# Normalizing continuous variables
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

# tuning
clf = LogisticRegression()
model_res = clf.fit(X_train_res, y_train_res)
from sklearn.model_selection import GridSearchCV

# Create regularization penalty space
penalty = ['l1', 'l2']

# Create regularization hyperparameter space
C = np.logspace(0, 4, 10)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

# Create grid search using 5-fold cross validation
clf = GridSearchCV(clf, hyperparameters, cv=5, verbose=0)

# Fit grid search
best_model = clf.fit(X_train, y_train)
# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])

# ----------------------------validation---------------------
clf = LogisticRegression(solver='liblinear', penalty='l1', C=1.0).fit(X_train, y_train)
score = clf.score(X_test, y_test)

predictions = clf.predict(X_test)
predictions = np.around(np.array(predictions))

# Calculate the absolute errors
errors = abs(predictions - y_test)

# result_accuracy.at[name, fs_item] = accuracy_score(predictions, y_test)
print('-' * 50)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
print('Accuracy:', accuracy_score(predictions, y_test))
print('Normalized Accuracy:', accuracy_score(predictions, y_test, normalize=False))
print('Mean squared error:', mean_squared_error(predictions, y_test))
print('average_precision:', average_precision_score(predictions, y_test))
print('recall_score:', recall_score(predictions, y_test))
print('recall_score:', classification_report(predictions, y_test))
print('-' * 50)
