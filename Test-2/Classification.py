#import essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_score
from sklearn.pipeline import  Pipeline

import psycopg2
import redis
from multiprocessing import Process, Queue
import time

#initialise db
def db():
    conn = psycopg2.connect(database="pokemon", user="apoorv", password="1234", host="localhost", port=5433)
    curr = conn.cursor()
    return conn, curr

#read from database
def load_the_database(conn, curr):
    query = 'SELECT * from pokemon'
    df = pd.read_sql(query, conn, index_col='_id')
    return df

#preprocessing of data for legendary pokemon classification
def preprocess_data(df):
    df['legendary'] = df['legendary'].astype('int64')
    X = df.drop('legendary', axis=1)
    Y = df['legendary']
    return X, Y

#define the Logistic regression model
def model():
    pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', LogisticRegression())])
    param_grid = {'classifier__C': np.logspace(-3, -1, 10),}
    model = GridSearchCV(pipeline, param_grid, cv=10, n_jobs=-1, verbose=2, refit=True, scoring='f1')
    return model

#split the training and testing data into 75/25
def split_data(X, Y):
    return train_test_split(X, Y, random_state=0, test_size=0.25, stratify=y)

#training and validation of model
def training_and_validation(model, Xtrain, Xtest, Ytrain, Ytest):
    model.fit(Xtrain, Ytrain)
    Yprob = model.predict_proba(Xtest)[:, 1]
    fpr, tpr = roc_curve(Ytest, Yprob)
    for group in list(zip(fpr, tpr)):
        print(group)


    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Negative Rate')
    plt.title('ROC AUC score: {}'.format(roc_auc_score(Ytest, Yprob)))
    plt.show()

    best_tpr = tpr[7]
    best_fpr = fpr[7]
    best_thres = thres[7]
    Ypred = np.zeros(Yprob.shape[0])
    Ypred[Yprob > best_threshold] = 1
    precision = precision_score(Ytest, Ypred)
    print('Precision: {}'.format(precision))
    return best_tpr, best_fpr, precision

# store values in redis
def store_values(tpr, fpr, precision):
    rd = redis.Redis(host='localhost')
    rd.set('tpr', tpr)
    rd.set('fpr', fpr)
    rd.set('precision', precision)
    print(rd.get('tpr'))
    print(rd.get('fpr'))
    print(rd.get('precision'))

#process 1 reading values from postgresql
def proc1(conn, curr, q):
    df = load_the_database(conn, curr)
    q.put(df)

#process 2 processing and analyzing values
def proc2(q):
    df = q.get()
    X, Y = preprocess_data(df)
    model = model()
    Xtrain, Xtest, Ytrain, Ytest = split(X, Y)
    tpr, fpr, precision = training_and_validation(model, Xtrain, Xtest, Ytrain, Ytest)
    store_values(tpr, fpr, precision)


if __name__ == '__main__':
    conn, curr = db()
    q = Queue()
    p1 = Process(target=proc1, args=(conn, curr, q))
    p2 = Process(target=proc2, args=(q))
    #start P1
    p1.start() 
    #delay between processes of 15 seconds
    time.sleep(15)
    #start P2
    p2.start()
    p2.join()

