# Import libraries
import pandas as pd 
import numpy as np
import pickle
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score

# Functions
def remover_outliers(nombre_columna, nombre_dataframe,umbral = 1.5):
    # IQR
    Q1 = np.percentile(nombre_dataframe[nombre_columna], 25,
                       interpolation = 'midpoint')
    Q3 = np.percentile(nombre_dataframe[nombre_columna], 75,
                       interpolation = 'midpoint')
    IQR = Q3 - Q1
    # Upper bound
    upper = np.where(nombre_dataframe[nombre_columna] >= (Q3+1.5*IQR))
    # Lower bound
    lower = np.where(nombre_dataframe[nombre_columna] <= (Q1-1.5*IQR))
    ''' Removing the Outliers '''
    nombre_dataframe = nombre_dataframe.drop(upper[0])
    nombre_dataframe = nombre_dataframe.drop(lower[0]).reset_index(drop = True)
    return nombre_dataframe

# Load data
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv')

# Removing outliers
remover_outliers('Pregnancies', df_raw,umbral = 1.5)
remover_outliers('Glucose', df_raw,umbral = 1.5)
remover_outliers('Insulin', df_raw,umbral = 1.5)
remover_outliers('BMI', df_raw,umbral = 1.5)
remover_outliers('DiabetesPedigreeFunction', df_raw,umbral = 1.5)
remover_outliers('Age', df_raw,umbral = 1.5)

# Separate labels from features
X = df_raw.drop(['Outcome'], axis = 1)
y = df_raw['Outcome']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)

# Train the model
tree_para = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150],'min_samples_split': [2, 3, 4]}
clf = GridSearchCV(DecisionTreeClassifier(), tree_para, cv=5)
clf.fit(X_train,y_train)
clf = DecisionTreeClassifier(criterion='entropy',max_depth=5,min_samples_split=4,random_state=0)
clf.fit(X_train, y_train)

# Save the model
filename = '../models/modelo_decision_tree.sav'
pickle.dump(clf, open(filename, 'wb'))

# Evaluate the model
print(clf.score(X_test, y_test))
print(classification_report(y_test, clf.predict(X_test)))
