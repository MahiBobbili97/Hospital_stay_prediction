#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv('data.csv')


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


def fill_with_mode(col):
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)


# In[8]:


fill_with_mode('Bed Grade')


# In[9]:


fill_with_mode('City_Code_Patient')


# In[10]:


def drop_column(col):
    df.drop([col], axis=1, inplace=True)


# In[11]:


drop_column('case_id')


# In[12]:


drop_column('patientid')


# In[13]:


categorical_cols = []
numerical_cols = []


# In[14]:


def find_categorical_cols(df):
    for col in df.columns:
        if df[col].dtypes == 'object':
            categorical_cols.append(col)


# In[15]:


def find_numerical_cols(df):
    for col in df.columns:
        if df[col].dtypes != 'object':
            numerical_cols.append(col)


# In[16]:


find_categorical_cols(df)
find_numerical_cols(df)


# In[17]:


print("Categorical Features:", categorical_cols)


# In[18]:


print("Numerical Features:", numerical_cols)


# In[19]:


i = 1
plt.figure(figsize=(15, 20))
for col in categorical_cols:
    plt.subplot(5, 2, i)
    sns.countplot(df[col])
    i = i + 1
plt.show()


# In[20]:


i = 1
plt.figure(figsize=(15, 20))
for col in numerical_cols:
    plt.subplot(4, 2, i)
    sns.distplot(df[col])
    i = i + 1

plt.show()


# In[21]:


def find_unique_in_cols(df):
    for col in df.select_dtypes(include='object').columns:
        print(col)
        print(df[col].unique())


# In[22]:


find_unique_in_cols(df)


# In[23]:


plt.figure(figsize=(10, 5))
g = sns.countplot(df['Stay'], order=df['Stay'].value_counts().index)
g.set_xticklabels(g.get_xticklabels(), rotation=45)
plt.show()


# In[24]:


le = preprocessing.LabelEncoder()
for column in categorical_cols:
    df[column] = le.fit_transform(df[column])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)


# In[25]:


corr = df.corr()
sns.heatmap(corr, cmap="viridis")


# In[26]:


i = 1
plt.figure(figsize=(15, 20))
for col in categorical_cols:
    if col not in ['Stay']:
        plt.subplot(5, 2, i)
        sns.barplot(x=df[col], y="Stay", data=df)
        i = i + 1
plt.show()


# In[27]:


sns.barplot(y=df["Visitors with Patient"], x=df["Stay"], data=df)


# In[28]:


print(df.info())


# In[29]:


for column in df:
    df[column] = df[column].astype(np.int64)


# In[30]:


X = df.loc[:, df.columns != 'Stay']
Y = df['Stay']


# In[31]:

counter = Counter(Y)
for k, v in counter.items():
    dist = v / len(Y) * 100
    print(f"Class={k}, n={v} ({dist}%)")


# In[32]:


plt.figure(1, figsize=(16, 4))
plt.bar(counter.keys(), counter.values())


# In[33]:


oversample = SMOTE()
X, Y = oversample.fit_resample(X, Y)


# In[38]:

def model_classifier(model, X, Y):
    cv = KFold(n_splits=10, shuffle=True, random_state=4)
    scores = []
    for train_index, test_index in cv.split(X, Y):
        X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                            test_size=0.2,
                                                            random_state=42,
                                                            stratify=Y)
        model_obj = model.fit(X_train, y_train)
        y_pred = model_obj.predict(X_test)
        Accuracy = []
        Accuracy.append(get_accuracy(y_test, y_pred))
    print("The Best Accuracy is: ")
    print(max(Accuracy))

# In[39]:


def get_classification_report(ytest, ypred):
    print(classification_report(ytest, ypred))


# In[40]:


def get_accuracy(ytest, ypred):
    accuracy = accuracy_score(ytest, ypred)
    return (accuracy * 100.0)


# In[41]:


model = DecisionTreeClassifier()
model_classifier(model, X, Y)


# In[42]:


forestVC = RandomForestClassifier()
model_classifier(forestVC, X, Y)


# In[43]:


model = CatBoostClassifier()
model_classifier(model, X, Y)


# In[44]:


X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=Y)


# In[45]:


param_grid = {
    'n_estimators': [100, 150],
    'max_features': ['auto', 'log2'],
    'max_depth': [None, 4, 5],
    'criterion': ['gini', 'entropy']
}


# In[46]:

forest = RandomForestClassifier()
CV_rfc = GridSearchCV(estimator=forest, param_grid=param_grid, cv=2)
CV_rfc.fit(X_train, y_train)


# In[47]:


CV_rfc.best_params_


# In[48]:


forestVC = RandomForestClassifier(n_estimators=100, max_features='auto',
                                  max_depth=None, criterion='gini')
model_classifier(forestVC, X, Y)


# In[49]:


params = {'depth': [3, 1, 2, 6, 4, 5, 7, 8, 9, 10],
          'iterations': [250, 100, 500, 1000],
          'learning_rate': [0.03, 0.001, 0.01, 0.1],
          'l2_leaf_reg': [3, 1, 5, 10, 100],
          'border_count': [32, 5, 13, 23, 58, 133, 254]}


# In[50]:


model = CatBoostClassifier()
CV_rfc = GridSearchCV(estimator=model, param_grid=params, cv=2)
CV_rfc.fit(X_train, y_train)


# In[55]:


CV_rfc.best_params_


# In[56]:


model = CatBoostClassifier(iterations=1000, learning_rate=0.1,
                           loss_function='MultiClass')
model_classifier(model, X, Y)


# In[57]:


param_dict = {
                "criterion": ['gini', 'entropy'],
                "max_depth": [None, 2, 3],
                "min_samples_split": range(1, 3),
                "min_samples_leaf": range(1, 3)
}

model = DecisionTreeClassifier()
clf = GridSearchCV(model, param_grid=param_dict, cv=2)
clf.fit(X_train, y_train)


# In[58]:


clf.best_params_


# In[60]:


model = DecisionTreeClassifier(criterion='gini', max_depth=None,
                               min_samples_split=2, min_samples_leaf=1)
model_obj = model.fit(X_train, y_train)
y_pred = model_obj.predict(X_test)
print(get_accuracy(y_test, y_pred))


# In[ ]:
