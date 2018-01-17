#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 18:43:00 2018

@author: aashish2096
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

data_train.info()
data_train.isnull().sum()
data_test.isnull().sum()


sns.heatmap(data_train[['Survived','Age','SibSp','Parch','Fare']].corr(),fmt = '.2f',cmap ='coolwarm',annot =True)
        
g = sns.factorplot(x ='SibSp',y='Survived',data =data_train, kind="bar",palette="muted",size =6) 
g.despine(left = True)
g = g.set_ylabels('Survival')       
        
g = sns.FacetGrid(data_train,col = 'Survived')
g = g.map(sns.distplot,'Age')

g = sns.kdeplot(data_train['Age'][(data_train['Survived'] == 0) & (data_train['Age'].notnull())],color ="Red")
g = sns.kdeplot(data_train['Age'][(data_train['Survived'] == 1) & (data_train['Age'].notnull())],color ="Blue")
g.set_xlabel("Age")
g.set_ylabel('Survival')
g = g.legend(['Survived','Not Survived'])


data_train.Fare.isnull().sum()


g = sns.barplot(x = 'Sex',y ="Survived",data = data_train)
g= g.set_ylabel('Survival')

g = sns.factorplot(x = 'Pclass',y = 'Survived',data = data_train,kind= 'bar',size =4,palette ="muted")
g.despine(left = True)
g= g.set_ylabels('Survival Rate')



data_train['Embarked'].isnull().sum()

g = sns.factorplot(x = 'Embarked',y = 'Survived',data = data_train,kind= 'bar',size =4,palette ="muted")
g.despine(left = True)
g= g.set_ylabels('Boarding Place')

sns.countplot(x= 'Embarked',data = data_train)
plt.show()

data_train['Embarked'].fillna('S',inplace = True)
data_train['Travel_freq'] = data_train['SibSp'] + data_train['Parch']
data_test['Travel_freq'] = data_test['SibSp'] + data_test['Parch']
sns.barplot(x  = 'Travel_freq', y = 'Survived',data = data_train)
plt.show()


data_train['Travel_alone'] = np.where(data_train['Travel_freq'] > 0 , 0 , 1)
data_train.drop('Travel_freq',axis = 1,inplace = True)

data_test['Travel_alone'] = np.where(data_test['Travel_freq'] > 0 , 0 , 1)
data_test.drop('Travel_freq',axis = 1,inplace = True)


#data_train.drop('Parch',axis = 1,inplace = True)

data_train.info()
data_test.info()
#data_train.drop('Cabin',axis = 1,inplace = True)
#data_train.info()

sns.kdeplot(data_train['Age'][data_train.Survived == 1],color = 'yellow')
sns.kdeplot(data_train['Age'][data_train.Survived == 0],color = 'green',shade = True)
plt.lebels(['Age','Survived'])
plt.show()

sns.barplot(x  = 'Ticket', y = 'Survived',data = data_train)
plt.show()
data_train.info()


data_train['Cabin_presence'] = data_train['Cabin'].notnull().astype('int')
sns.barplot(x="Cabin_presence", y="Survived", data=data_train)
plt.show()

data_test['Cabin_presence'] = data_test['Cabin'].notnull().astype('int')

data_train.info()

data_train.drop('Cabin',axis = 1, inplace = True)
data_test.drop('Cabin',axis = 1, inplace = True)

data_train['Title'] = data_train.Name.str.extract('([A-Za-z]+)\.',expand = False)
data_test['Title'] = data_test.Name.str.extract('([A-Za-z]+)\.',expand = False)
pd.crosstab(data_train['Title'],data_train['Survived'])
pd.crosstab(data_test['Title'],data_train['Survived'])


data_train['Title'] = data_train['Title'].replace(['Capt','Col','Countless','Don','Dr','Jonkheer','Lady','Major','Rev'],'Misc')
data_train['Title'].unique()
data_train['Title'] = data_train['Title'].replace(['Mne','Mlle','Ms'],'Miss')
data_train['Title'] = data_train['Title'].replace(['Sir'],'Mr')

data_train['Title'].unique()

data_train['Title'] = data_train['Title'].replace(['Mme','Countess'],'Miss')
data_train['Title'].unique()

data_train[['Title','Survived']].groupby(['Title'], as_index = False).mean()

title_map ={'Master': 1 , 'Miss' : 2, 'Mr' : 3 ,'Mrs' : 4 ,'Misc': 5}
data_train['Title'] =  data_train['Title'].map(title_map)



data_test['Title'] = data_test['Title'].replace(['Col','Dona','Dr','Rev'],'Misc')
data_test['Title'] = data_test['Title'].replace(['Ms'],'Mrs')
data_test['Title'].unique()

title_map ={'Master': 1 , 'Miss' : 2, 'Mr' : 3 ,'Mrs' : 4 ,'Misc': 5}
data_test['Title'] =  data_test['Title'].map(title_map)


sex_mapping = {"male": 0, "female": 1}
data_train['Sex'] = data_train['Sex'].map(sex_mapping)
data_test['Sex'] = data_test['Sex'].map(sex_mapping)

place_map ={'S':1,'C':2,'Q':3}
data_train['Embarked'] = data_train['Embarked'].map(place_map)
data_test['Embarked'] = data_test['Embarked'].map(place_map)

for x in range(len(data_test["Fare"])):
    if pd.isnull(data_test["Fare"][x]):
        pclass = data_test["Pclass"][x] #Pclass = 3
        data_test["Fare"][x] = round(data_train[data_train["Pclass"] == pclass]["Fare"].mean(), 4)
        

for x in range(len(data_train['Age'])):
    
    age_1 = round(data_train[data_train['Title']== 1]['Age'].mean())  
    age_2 = round(data_train[data_train['Title']== 2]['Age'].mean())
    age_3 = round(data_train[data_train['Title']== 3]['Age'].mean())  
    age_4 = round(data_train[data_train['Title']== 4]['Age'].mean())
    age_5 = round(data_train[data_train['Title']== 5]['Age'].mean())
    
    
age_df = [age_1,age_2,age_3,age_4,age_5]
   

data_train['Age'] = data_train['Age'].fillna('k')
data_test['Age'] = data_test['Age'].fillna('k')


for x in range(len(data_train['Age'] )):
    if(data_train['Age'][x] == 'k'):
        if (data_train['Title'][x] == 1):
            data_train['Age'][x] = age_df[0]
        elif(data_train['Title'][x] == 2):
            data_train['Age'][x] = age_df[1]       
        elif(data_train['Title'][x] == 3):
            data_train['Age'][x] = age_df[2]  
        elif(data_train['Title'][x] == 4):
            data_train['Age'][x] = age_df[3]  
        elif(data_train['Title'][x] == 5):
            data_train['Age'][x] = age_df[4]      


for x in range(len(data_test['Age'] )):
    if(data_test['Age'][x] == 'k'):
        if (data_test['Title'][x] == 1):
            data_test['Age'][x] = age_df[0]
        elif(data_test['Title'][x] == 2):
            data_test['Age'][x] = age_df[1]       
        elif(data_test['Title'][x] == 3):
            data_test['Age'][x] = age_df[2]  
        elif(data_test['Title'][x] == 4):
            data_test['Age'][x] = age_df[3]  
        elif(data_test['Title'][x] == 5):
            data_test['Age'][x] = age_df[4]      


data_train.info()
data_test.info()
)


ide = data_test['PassengerId']
y_train = data_train['Survived']

x_train  = data_train.drop('Survived',axis = 1, inplace = True)
data_train.drop('PassengerId',axis =1 ,inplace = True)
data_test.drop('PassengerId',axis =1 ,inplace = True)
rem = ['SibSp','Parch','Name','Ticket']
data_train.drop(rem,axis = 1, inplace = True)
data_test.drop(rem,axis = 1, inplace = True)
x_train = data_train
x_test = data_test


from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

x_train ,x_test ,y_train,y_test = train_test_split(data_train , y_train ,test_size = 0.25,random_state = 0)


gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_test)
acc_gaussian = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_gaussian)

l_reg = LogisticRegression()
l_reg.fit(x_train,y_train)
y_pred = l_reg.predict(x_test)
acc_l_reg = round(accuracy_score(y_pred,y_test),2)
print(acc_l_reg)

random_for = RandomForestClassifier()
random_for.fit(x_train,y_train)
y_pred = random_for.predict(x_test)
acc_random_for = round(accuracy_score(y_pred,y_test),2)
print(acc_random_for)

knn_class = KNeighborsClassifier()
knn_class.fit(x_train,y_train)
y_pred = knn_class.predict(x_test)
acc_knn_class = round(accuracy_score(y_pred,y_test),2)
print(acc_knn_class)

svc_class = SVC()
svc_class.fit(x_train,y_train)
y_pred = svc_class.predict(x_test)
acc_svc_class = round(accuracy_score(y_pred,y_test),2)
print(acc_svc_class)

des_class = DecisionTreeClassifier()
des_class.fit(x_train,y_train)
y_pred = des_class.predict(x_test)
acc_des_class = round(accuracy_score(y_pred,y_test),2)
print(acc_des_class)

gb_class = GradientBoostingClassifier()
gb_class.fit(x_train,y_train)
y_pred = gb_class.predict(x_test)
acc_gb_class = round(accuracy_score(y_pred,y_test),2)
print(acc_gb_class)


predictions  = random_for.predict(data_test)
output = pd.DataFrame({ 'PassengerId' : ide, 'Survived': predictions })
output.to_csv('submission.csv', index=False)

















































        
        
        