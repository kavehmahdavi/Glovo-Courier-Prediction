# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import MinCovDet
import statistics
import statsmodels.api as sm
import matplotlib.patches as mpatches

# %%
Courier_weekly_data = pd.read_csv("data/Courier_weekly_data.csv")
Courier_lifetime_data = pd.read_csv("data/Courier_lifetime_data.csv")
# %%
# Check the weather of categorical features, if some of them are needed to be assumed as continue or not.
cat_features = ['feature_1',
                'feature_2',
                'feature_3',
                'feature_11',
                'feature_16',
                'feature_17']
print(Courier_weekly_data.loc[:, cat_features].describe())

for feature_item in cat_features:
    print('The feature <{}> level size is {}'.format(feature_item, len(set(Courier_weekly_data.loc[:, feature_item]))))

# Note: I decided to eliminate the {'feature_11','feature_16','feature_17'}.
cat_dropped_features = ['feature_11', 'feature_16', 'feature_17']
Courier_weekly_data = Courier_weekly_data.drop(columns=cat_dropped_features)

# %%
Courier_weekly_data = Courier_weekly_data.drop(columns=['feature_2'])

# Note: other categorical features are treated as float.
Courier_weekly_data = Courier_weekly_data.astype({"feature_1": float, "feature_3": float})

# Indexing the panel data
weeks = Courier_weekly_data['week']
couriers = Courier_weekly_data['courier']
Courier_weekly_data = Courier_weekly_data.drop(columns=['courier', 'week'])

# %% md
# Outlier handling
# %%
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
print("The first 8 component are describing the 90% of variance among the data.")

# Mahalanobis distance for evaluating the overall distance of the outliers.
# fit a Minimum Covariance Determinant (MCD) robust estimator to data
robust_cov = MinCovDet().fit(T[:, :7])

# Get the Mahalanobis distance
m = robust_cov.mahalanobis(T[:, :7])
plt.hist(m, density=True, bins=50)
plt.ylabel('Probability')
plt.xlabel('Mahalanobis distance')
plt.title('Mahalanobis distance histogram')
plt.xlim(-1, 700)
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
print("Mean of sample is: % s " % (statistics.mean(m)))
print("Standard Deviation of sample is: % s " % (statistics.stdev(m)))
print('two standard deviation off the data is keep.')

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
# %%
# Indexing the panel data
week = Courier_weekly_data['week']
courier = Courier_weekly_data['courier']
Courier_weekly_data = Courier_weekly_data.drop(columns=['Mahalanobis_distance'])
Courier_weekly_data_clean = Courier_weekly_data
Courier_weekly_data = Courier_weekly_data.set_index(['courier', 'week']).sort_index()

# %% md
## Missing values handling
# %%
Courier_weekly_data_mean = Courier_weekly_data.mean(level='courier', axis=0)
merged_df = pd.merge(Courier_lifetime_data, Courier_weekly_data_mean, how='left', on='courier')
merged_df = merged_df.set_index(['courier']).sort_index()

print('=' * 80)
print('The number of missing values in Courier_lifetime is:', merged_df[merged_df[
    'feature_2_life'].isnull()].shape[0])
print('=' * 80)

print('Note: slice the data in two mentioned groups (without/with information)')

# with information
merged_df_with_weekly_info = merged_df.loc[list(set(courier))]
print(merged_df_with_weekly_info.shape)

print('check the response variable (feature_2_lifetime) distribution.')
plt.hist(merged_df['feature_2_life'].dropna(), bins=100)
plt.ylabel('Probability')
plt.xlim((-10, 100))
plt.ylim((0, 3000))
Todo: plt.show()
print('So it is close to Poisson.')

merged_df_with_weekly_info_feature_1_life = merged_df_with_weekly_info['feature_1_life']
merged_df_with_weekly_info = merged_df_with_weekly_info.drop(columns=['feature_1_life'])

merged_df_with_weekly_info_pred = merged_df_with_weekly_info[merged_df_with_weekly_info[
    'feature_2_life'].isnull()]
merged_df_with_weekly_info_train = merged_df_with_weekly_info.dropna(subset=['feature_2_life'])

print('-' * 80)
print('The Train data for missing value predication shape is:', merged_df_with_weekly_info_train.shape)
print('The predicable data for missing value predication shape is:', merged_df_with_weekly_info_pred.shape)
print('-' * 80)

print("Note: The feature subset ['feature_1','feature_5','feature_9'] is omitted based on Z-test p_value (>0.05)")

data_endog = merged_df_with_weekly_info_train['feature_2_life']
data_exog = merged_df_with_weekly_info_train.drop(columns=['feature_2_life', 'feature_1', 'feature_5', 'feature_9'])

pred_endog = []
pred_exog = merged_df_with_weekly_info_pred.drop(columns=['feature_2_life', 'feature_1', 'feature_5', 'feature_9'])

# Instantiate a gamma family model with the default link function.
Poisson_model = sm.GLM(data_endog, data_exog, family=sm.families.Poisson())
Poisson_results = Poisson_model.fit()
print('-' * 80)
print(Poisson_results.summary())
print('-' * 80)

Poisson_model_pred = round(Poisson_results.predict(pred_exog))
merged_df_with_weekly_info_pred['feature_2_life'] = Poisson_model_pred

# Reconstruct the merged_df_with_weekly_info
merged_df_with_weekly_info = pd.concat([merged_df_with_weekly_info_pred,
                                        merged_df_with_weekly_info_train])
merged_df_with_weekly_info['feature_1_life'] = merged_df_with_weekly_info_feature_1_life

# without information
merged_df_without_weekly_info = merged_df.drop(list(set(courier)))
lifetime_imputed = pd.concat([merged_df_without_weekly_info[['feature_1_life', 'feature_2_life']],
                              merged_df_with_weekly_info[['feature_1_life', 'feature_2_life']]])

print('=' * 80)
print('The number of missing values is:', lifetime_imputed[lifetime_imputed[
    'feature_2_life'].isnull()].shape[0])
print('Equal to: %', 100 * lifetime_imputed[lifetime_imputed[
    'feature_2_life'].isnull()].shape[0] / lifetime_imputed.shape[0])
print('Note: There is interaction between the feature_1_life and feature_2_life '
      '(overlap between confidence intervals)then it is imposable to impute missing'
      ' values in feature_2_life with assuming the feature_1_life.'
      'So, the rest of recodes with missing values are needed to be deleted.')
print('=' * 80)

# Note: the rest of them are imputed by the mean of any group

print(lifetime_imputed.groupby(['feature_1_life']).mean())
print(lifetime_imputed.groupby(['feature_1_life']).std())
# %%
'''
Note: The mean of different group (a,b,c,d) is very close nad the confidence interval is to wide.
      I prefer not to impute by mean of categories. It is not a good approach. 
      On the same hand, The data set is big enough for eliminating 11% of the data.
'''
lifetime_imputed = lifetime_imputed.dropna(subset=['feature_2_life'])

print('=' * 80)
print('The final shape of the lifetime data set is:', lifetime_imputed.shape)

Courier_weekly_data_clean = pd.merge(Courier_weekly_data_clean, lifetime_imputed, how='left', on='courier')

print('The final shape of the weekly data set is:', Courier_weekly_data_clean.shape)
print('=' * 80)
# %% md
# Feature Relation

# %%
import seaborn as sns

fig = plt.figure(figsize=(16, 12))
corr = Courier_weekly_data_clean.drop(['courier', 'week'], axis=1).corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True)
plt.show()

# %%
Courier_weekly_data_mean = Courier_weekly_data_clean.drop(columns=['feature_1_life'], axis=1).set_index(
    ['courier', 'week']).mean(level='courier', axis=0)
merged_weekly_with_label = pd.merge(Courier_weekly_data, Courier_lifetime_data, how='left', on='courier')[
    'feature_1_life']

columns_item = Courier_weekly_data_clean.drop(columns=['feature_1_life'], axis=1).set_index(['courier', 'week']).columns

merged_weekly_with_label = np.where(merged_weekly_with_label == 'a', 'w', merged_weekly_with_label)
merged_weekly_with_label = np.where(merged_weekly_with_label == 'b', 'green', merged_weekly_with_label)
merged_weekly_with_label = np.where(merged_weekly_with_label == 'c', 'blue', merged_weekly_with_label)
merged_weekly_with_label = np.where(merged_weekly_with_label == 'd', 'red', merged_weekly_with_label)

pca = PCA()
T = pca.fit_transform(StandardScaler().fit_transform(
    Courier_weekly_data_clean.drop(columns=['feature_1_life'], axis=1).set_index(['courier', 'week']).sort_index()))


# Score plot of the first 2 PC


def myplot(score, coeff, labels=None, colored=True, cc=merged_weekly_with_label):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    with plt.style.context(('ggplot')):
        if colored:
            plt.scatter(xs * scalex, ys * scaley, c=cc, edgecolors='k', cmap='jet')
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


fig = plt.figure(figsize=(16, 12))
plt.xlim(-0.75, 0.75)
plt.ylim(-0.75, 0.75)
plt.xlabel("PC{}".format(1))
plt.ylabel("PC{}".format(2))
plt.grid()

classes = ['A', 'B', 'C', 'D']
class_colours = ['w', 'g', 'b', 'r']
recs = []
for i in range(0, len(class_colours)):
    recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
plt.legend(recs, classes, loc=4)

# Call the function. Use only the 2 PCs.
myplot(T[:, 0:2], np.transpose(pca.components_[0:2, :]), labels=columns_item, cc=merged_weekly_with_label)
plt.show()

# psc for summary data
merged_weekly_with_label = pd.merge(Courier_weekly_data_mean, lifetime_imputed, how='left', on='courier')[
    'feature_1_life']

merged_weekly_with_label = np.where(merged_weekly_with_label == 'a', 'w', merged_weekly_with_label)
merged_weekly_with_label = np.where(merged_weekly_with_label == 'b', 'green', merged_weekly_with_label)
merged_weekly_with_label = np.where(merged_weekly_with_label == 'c', 'blue', merged_weekly_with_label)
merged_weekly_with_label = np.where(merged_weekly_with_label == 'd', 'red', merged_weekly_with_label)

pca = PCA()
T = pca.fit_transform(StandardScaler().fit_transform(Courier_weekly_data_mean))
# Score plot of the first 2 PC

fig = plt.figure(figsize=(16, 12))
plt.xlim(-0.75, 0.75)
plt.ylim(-0.75, 0.75)
plt.xlabel("PC{}".format(1))
plt.ylabel("PC{}".format(2))
plt.grid()

classes = ['A', 'B', 'C', 'D']
class_colours = ['w', 'g', 'b', 'r']
recs = []
for i in range(0, len(class_colours)):
    recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
plt.legend(recs, classes, loc=4)

# Call the function. Use only the 2 PCs.
myplot(T[:, 0:2], np.transpose(pca.components_[0:2, :]), labels=columns_item, cc=merged_weekly_with_label)
plt.show()
# %% md
# Modeling
# %%
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
# %%
# psc for summary data
merged_weekly_with_label = Courier_weekly_data_clean['label']

merged_weekly_with_label = np.where(merged_weekly_with_label == 0, 'r', merged_weekly_with_label)
merged_weekly_with_label = np.where(merged_weekly_with_label == '1', 'b', merged_weekly_with_label)

pca = PCA()
T = pca.fit_transform(StandardScaler().fit_transform(
    Courier_weekly_data_clean.drop(columns=['feature_1_life', 'week', 'courier', 'label'])))

# Score plot of the first 2 PC

fig = plt.figure(figsize=(16, 12))
plt.xlim(-0.75, 0.75)
plt.ylim(-0.75, 0.75)
plt.xlabel("PC{}".format(1))
plt.ylabel("PC{}".format(2))
plt.grid()

classes = ['1', '0']
class_colours = ['b', 'r']
recs = []
for i in range(0, len(class_colours)):
    recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
plt.legend(recs, classes, loc=4)

# Call the function. Use only the 2 PCs.
myplot(T[:, 0:2], np.transpose(pca.components_[0:2, :]), labels=Courier_weekly_data_clean.columns,
       cc=merged_weekly_with_label)
plt.show()

# %% md
#  Imbalanced Classes

# %%
print(Courier_weekly_data_clean.groupby(['label']).size())

# %%


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
print(Courier_weekly_data_clean.groupby(['label']).size())

# %%
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
import statsmodels.api as sm
import statsmodels.formula.api as smf


# %%
def validation_report(predictions, y_test):
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


# %%
Courier_weekly_data_clean_ = Courier_weekly_data_clean.set_index(['courier', 'week']).sort_index()
df = Courier_weekly_data_clean.copy()
labs = df['label']
abcd = df['feature_1_life']
df = df.drop(['label', 'feature_1_life'], axis=1)
df_ordinal = df.copy(deep=True)
cols = df.columns
df = StandardScaler().fit_transform(df)
df = pd.DataFrame(data=df, columns=cols)
# Saving feature names for later use
feature_list = list(df.columns)
# Convert to numpy array
df = np.array(df)

# %%
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

# %% md
### <span style="color:red">Warning: it is not a good idea to run it on Jupiter.</span>

# %%
from sklearn.kernel_ridge import KernelRidge

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA", "LogisticRegression", "Kernel Ridge"]

classifiers = [
    KNeighborsClassifier(weights='distance', n_neighbors=50, algorithm='auto'),
    SVC(kernel="linear", C=0.025),
    SVC(),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=10, n_estimators=30, max_features=5),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(solver='liblinear'),
    KernelRidge(alpha=3)]

# iterate over classifiers
for name, clf in zip(names, classifiers):
    print('===============', name, '==================')
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    # ----------------------------validation---------------------
    predictions = clf.predict(X_test)
    predictions = np.around(np.array(predictions))
    validation_report(predictions, y_test)

# %%
clf = RandomForestClassifier(max_depth=10, n_estimators=30, max_features=5)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
predictions = clf.predict(X_test)
predictions = np.around(np.array(predictions))
validation_report(predictions, y_test)

base_accuracy_RF = accuracy_score(predictions, y_test)

# %%
clf = MLPClassifier(alpha=1)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
predictions = clf.predict(X_test)
predictions = np.around(np.array(predictions))
validation_report(predictions, y_test)
base_accuracy_ANN = accuracy_score(predictions, y_test)

# %%
# Tune two models one by one.
X_train_rf = X_train.copy()
X_test_rf = X_test.copy()

X_train_ann = X_train.copy()
X_test_ann = X_test.copy()

# %% md
# Tuning the hyper parameters of model Grid Search
# %% md
### Random forest model tuning

# %% md
### <span style="color:red">Warning: it is not a good idea to run it on Jupiter.</span>
# %%
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
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
# Use the random grid to search for best hyper parameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf,
                               param_distributions=random_grid,
                               n_iter=20, cv=5, verbose=2,
                               random_state=42, n_jobs=-1)

# Fit the random search model
rf_random.fit(X_train, y_train)
print(rf_random.best_params_)

# %%
clf = RandomForestClassifier(max_depth=20,
                             n_estimators=800,
                             max_features='sqrt',
                             min_samples_split=2,
                             bootstrap=False,
                             min_samples_leaf=1)
clf.fit(X_train_rf, y_train)
score = clf.score(X_test_rf, y_test)
predictions = clf.predict(X_test_rf)
predictions = np.around(np.array(predictions))
validation_report(predictions, y_test)
final_accuracy_RF = accuracy_score(predictions, y_test)
print('Random Forest model: Improvement of {:0.2f}%.'.format(
    100 * (final_accuracy_RF - base_accuracy_RF) / base_accuracy_RF))

# %% md
## ANN model tuning
# %% md
### <span style="color:red">Warning: it is not a good idea to run it on Jupiter.</span>
# %%
from sklearn.model_selection import GridSearchCV

mlp = MLPClassifier(max_iter=100)
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

# %%
clf = MLPClassifier(activation='relu',
                    alpha=0.9,
                    hidden_layer_sizes=(100,),
                    learning_rate='adaptive',
                    solver='adam')

clf.fit(X_train_ann, y_train)
score = clf.score(X_test_ann, y_test)
predictions = clf.predict(X_test_ann)
predictions = np.around(np.array(predictions))
validation_report(predictions, y_test)
final_accuracy_ANN = accuracy_score(predictions, y_test)
print('ANN model: Improvement of {:0.2f}%.'.format(100 * (final_accuracy_ANN - base_accuracy_ANN) / base_accuracy_ANN))

# %%
# Logistic regression
train, test = train_test_split(Courier_weekly_data_clean, test_size=0.25)
formula = 'label ~ feature_1+feature_3+feature_4+feature_5+feature_6+feature_7+feature_8+feature_12+' \
          'feature_14+feature_15+feature_2_life'

Poisson_model = smf.glm(formula=formula, data=train, family=sm.families.Binomial())
Poisson_results = Poisson_model.fit()
print('*' * 50)
print(Poisson_results.summary())
print('*' * 50)
predictions = round(Poisson_results.predict(test))
validation_report(predictions, test['label'])
