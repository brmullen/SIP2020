#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:56:02 2020

@author: saathvikdirisala
"""


import pandas as pd
from collections import Counter
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.decomposition import PCA
from statistics import stdev
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.svm import SVR, SVC

def read_tsv(file, index_col):
    tsv_file = open(file)
    data = pd.read_csv(tsv_file, delimiter = "\t", index_col = index_col)
    tsv_file.close()
    return data

for i in range(14):
    i = i+1
    if(i==1):
        data = read_tsv("/Users/saathvikdirisala/Desktop/python/PostNatalData/P"+str(i)+"_IC_metrics.tsv", index_col = "exp_ic")
    else:
        df = read_tsv("/Users/saathvikdirisala/Desktop/python/PostNatalData/P"+str(i)+"_IC_metrics.tsv", index_col = "exp_ic")
        data = data.append(df)

def missing_val_fill(data, col, filler):
    if len(str(filler))<=10:
        if filler == "random":
           for i in range(len(col)):
                data[col[i]][np.isnan(data[col[i]])==True] = list(np.random.normal(np.mean(data[col[i]][np.isnan(data[col[i]])==False]),
                                                                                   stdev(data[col[i]][np.isnan(data[col[i]])==False]),
                                                                                   len(data[col[i]][np.isnan(data[col[i]])==True])))
        else:
           data[col] = data[col].fillna(filler)
    else:
        data[col] = data[col].fillna(filler)
    return data

def FeatureScaler(df, num_cols, ScalerType):
    data = df[num_cols]
    if ScalerType == "Standard" or ScalerType == "standard" or ScalerType == "StandardScaler":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(data)
        data = scaler.transform(data)
    elif ScalerType == "MinMax" or ScalerType == "minmax" or ScalerType == "MinMaxScaler":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler().fit(data)
        data = scaler.transform(data)
    elif ScalerType == "Robust" or ScalerType == "robust" or ScalerType == "RobustScaler":
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler().fit(data)
        data = scaler.transform(data)
    data = pd.DataFrame(data, index = df.index)
    data.columns = num_cols
    for i in num_cols:
        df[i] = data[i]
    return df

def Encoder(data, col):
    import collections
    a = collections.Counter(data[col])
    if(len(a)>2):
        lb = LabelBinarizer()
        cat = lb.fit_transform(data[col])
        cat = pd.DataFrame(cat, index = data.index)
        cat.columns = [list(lb.classes_)]
        data = data.drop(labels = [col], axis = 1)
        data[list(lb.classes_)] = cat
    else:
        lb = preprocessing.LabelEncoder()
        lb.fit(data[col])
        data[col] = lb.transform(data[col])
    return data

def cat_df(data, feature):
    unq = list(Counter(data[feature]))
    length = len(unq)
    data1 = list(np.zeros(length))
    for i in range(length):
        TF = data[feature]==unq[i]#.between(i*(length/split),(i+1)*(length/split))
        d = data[TF]
        col = list(d.columns)
        col.remove(feature) 
        data1[i] = d[col]
    return data1

def split(data, conditions):
    df = [0]
    for i in range(len(conditions)):
        df[i] = data[conditions[i]]
        df.append(0)
    df = df[:-1]
    return df

def test_train_split(X,y, n_splits, test_size):
    Split = StratifiedShuffleSplit(n_splits = n_splits, test_size = test_size)
    for train_index, test_index in Split.split(X,y):
        X_test, X_train = X.iloc[test_index],X.iloc[train_index]
        y_test, y_train = y.iloc[test_index],y.iloc[train_index]
    return X_train, X_test, y_train, y_test

num_cols = list(data.columns)
num_cols.remove("artifact")
num_cols.remove("signal")
num_cols.remove("age")
num_cols.remove("anml")

col = [0]
q = 0

col = list(data.columns)
'''for e in data.columns:
    if(len(np.unique(np.isnan(data[e])))==2):
        col[q] = e
        col.append(0)
        q+=1
col = col[:-1]'''
col.remove("threshold.area")
col.remove("threshold.perc")

data[data["signal"]==1] = missing_val_fill(data[data["signal"]==1], col, data[data["signal"]==1].median())
data[data["signal"]==0] = missing_val_fill(data[data["signal"]==0], col, data[data["signal"]==0].median())

print(Counter(np.isnan(data["threshold.area"])))

data = (data.pipe(missing_val_fill, col = ["threshold.area", "threshold.perc"], filler = 0)
            .pipe(cat_df, "age"))
            #.pipe(missing_val_fill, col = col, filler = data.median())

print(len(data[0][data[0]["threshold.area"]==0]))

'''import sweetviz
Domain = data[0][data[0]["threshold.area"]!=0]
NoDomain = data[0][data[0]["threshold.area"]==0]
report = sweetviz.compare([Domain, "Domain"], [NoDomain, "NoDomain"], "signal")'''

for i in range(len(data)):
    data[i] = (data[i].pipe(FeatureScaler, num_cols = num_cols, ScalerType = "standard")
                      .pipe(Encoder, col = "artifact")
                      .pipe(Encoder, col = "signal"))
    


ages = 5
X = list(np.zeros(ages))
y = list(np.zeros(ages))
for i in range(ages):
    X[i] = data[i][num_cols]
    y[i] = data[i]["signal"]
    X[i].columns = num_cols
    y[i].columns = ["signal"]

X_train = list(np.zeros(ages))
X_test = list(np.zeros(ages))
y_train = list(np.zeros(ages))
y_test = list(np.zeros(ages))



'''ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units = 37, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 37, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
#output layer activation = 'soft max' for non-binary output variable

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#categorical_crossentropy for non-binary output variable

log = LogisticRegression(solver = "liblinear")

y_pred = list(np.zeros(len(data)))
for i in range(len(data)):
    #log.fit(X_train[i], y_train[i])
    ann.fit(X_train[i], y_train[i], batch_size = 32, epochs = 150)
    y_pred[i] = ann.predict(X_test[i])
    y_pred[i] = (y_pred[i] > 0.5)
    y_test[i] = np.array(y_test[i])
    #print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test[0].reshape(len(y_test[0]),1)),1))    

for i in range(len(data)):
    cm = confusion_matrix(y_test[i], y_pred[i])
    print(cm)
    acc = accuracy_score(y_test[i], y_pred[i])
    print("accuracy:",acc)
    pre = precision_score(y_test[i], y_pred[i])
    print("precision:",pre)
    rec = recall_score(y_test[i], y_pred[i])
    print("recall:",rec)'''

'''activation = ["relu", "softmax", "leakyrelu", "prelu", "elu", "thresholdrelu"]
optimizer = ["sgd", "rmsprop", "adam", "adadelta", "adagrad", "adamax", "nadam", "ftrl"]
units = [5, 10, 15, 20, 25, 30, 35, 37]
batch_size = [20, 30, 32, 40, 50]

param_grid = dict(activation = activation,
                 optimizer = optimizer,
                 units = units,
                 batch_size = batch_size)

grid = GridSearchCV(estimator = ann,
                    param_grid = param_grid,
                    scoring = "roc_auc",
                    verbose = 1,
                    n_jobs = -1)

grid_result = grid.fit(X_train[0], y_train[0])

print("Best Score:", grid_result.best_score_)
print("Best Parameters:", grid_result.best_params_)
'''

'''for w in range(len(X)):
    X[w] = X[w][X[w]["threshold.area"]!=0]'''


ranking_dfs = list(np.zeros(ages))
fig = list(np.zeros(ages))
for u in range(len(fig)):
    fig[u] = list(np.zeros(len(X[0].columns)))
for j in range(ages):
    a = 0
    shape_diff = list(np.zeros(len(X[j].columns)))
    median_diff = list(np.zeros(len(X[j].columns)))
    modal_diff = list(np.zeros(len(X[j].columns)))
    skew_diff = list(np.zeros(len(X[j].columns)))
    cluster_diff = list(np.zeros(len(X[j].columns)))
    for i in X[j].columns:
        signal_indices = y[j][y[j]==1].index
        artifact_indices = y[j][y[j]==0].index
        a+=1
        sig = X[j][i][signal_indices]
        art = X[j][i][artifact_indices]
        median_diff[a-1] = abs(sig.median()-art.median())
        if(stdev(sig)!=0 and stdev(art)!=0):
            skew_diff[a-1] = abs(3*(sig.mean()-sig.median())/(stdev(sig)) - 3*(art.mean()-art.median())/(stdev(art)))
        else:
            skew_diff[a-1] = abs(3*(sig.mean()-sig.median()) - 3*(art.mean()-art.median()))
  
        plt.figure(100)
        n = plt.hist([sig,art], bins = 20)
        plt.close(fig=100)
        '''plt.figure(a)
        plt.title(str(i) + "  " + str(median_diff[a-1]))
        sb.distplot(sig, bins = 20, hist = False, rug = True, color = "green", kde_kws = {'bw':0.1})
        sb.distplot(art, bins = 20, hist = False, rug = True, color = "red", kde_kws = {'bw':0.1})'''
        fig[j][a-1] , ax1 = plt.subplots()
        ax1.set_title("P"+str(j+1)+"  "+str(i))
        ax1.hist(sig, bins = 50, alpha = 0.5, label ='artifact', color = 'r')
        ax1.grid(False)         
        ax1.set_xlabel('Values')
        ax1.set_ylabel('# of instances', color = 'red')
        
        ax2 = ax1.twinx()
        ax2.hist(art, bins = 50, alpha = 0.5, label ='signal', color = 'g')
        ax2.set_ylabel('# of instances', color = 'green')
        ax2.grid(False)
        
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.show()
        
        temp_df = pd.DataFrame(n[0][0], columns = ["signal"])
        temp_df["artifact"] = n[0][1]
        loc = n[1]
        temp_df = FeatureScaler(temp_df, ["signal", "artifact"], "minmax")
        shape_diff[a-1] = list(np.array(temp_df["signal"])-np.array(temp_df["artifact"]))
        modal_diff[a-1] = abs(loc[list(temp_df["signal"]).index(max(temp_df["signal"]))]-loc[list(temp_df["artifact"]).index(max(temp_df["artifact"]))])
        cluster_diff[a-1] = modal_diff[a-1]*(1/stdev(sig))*(1/stdev(art))
        
    
    def diff_gauge(data):
        stdev = list(np.zeros(len(data)))
        for i in range(len(data)):
            stdev[i] = np.sqrt(sum(np.array(data[i])**2)/(len(data[i])-1))
        return stdev
    
    diffg = diff_gauge(shape_diff)
    
    features = pd.DataFrame(X[j].columns, columns = ["feature"], index = np.arange(1,(len(X[j].columns)+1)))
    features["median_diff"] = median_diff
    features["shape_diff"] = diffg
    features["modal_diff"] = modal_diff
    features["skew_diff"] = skew_diff
    features["cluster_diff"] = cluster_diff
    features = FeatureScaler(features, ["median_diff", "shape_diff", "modal_diff", "skew_diff", "cluster_diff"], "minmax")
    #features["combined_gauge1"] = np.array(features["median_diff"])+np.array(features["shape_diff"])+np.array(features["modal_diff"])
    #features["combined_gauge2"] = (np.array(features["spread_diff"]))+np.array(features["median_diff"])
    
    ranking_dfs[j] = features
    
r = len(ranking_dfs[0])
rank = list(np.zeros(len(ranking_dfs)))
for h in range(len(ranking_dfs)):
    ranking_dfs[h]["Combo_Gauge"] = 3*np.array(ranking_dfs[h]["cluster_diff"]) + 1*np.array(ranking_dfs[h]["median_diff"]) + 0*np.array(ranking_dfs[h]["shape_diff"]) + 2.3*np.array(ranking_dfs[h]["modal_diff"]) + 0*np.array(ranking_dfs[h]["skew_diff"])
    ranking_dfs[h] = (ranking_dfs[h].sort_values(by = ["cluster_diff"], ascending = False ,ignore_index = True))
    rank[h] = ranking_dfs[h][:r]

print(rank)

def figure(data, feature):
    index = list(data.columns).index(feature)
    return index

a = [0]
b = 0
for i in range(len(rank)):
    for j in (rank[i]["feature"]):
        idx = figure(X[0], j)
        a[b] = fig[i][idx]
        b+=1
        a.append(0)
a = a[:-1]

sampsize = 25
lst = list(np.zeros(ages))
for t in range(ages):
    lst[t] = list(np.zeros(sampsize))

scores = pd.DataFrame(list(range(1,(ages+1))), columns = ["Age"])
scores["Accuracy"] = lst
scores["Precision"] = lst
scores["Recall"] = lst
_params = pd.DataFrame(list(range(1,(ages+1))), columns = ["Age"])
_params["params"] = lst

y_pred = [0]
X_train_mod = list(np.zeros(len(X_train)))
X_test_mod = list(np.zeros(len(X_test)))

for i in range(ages):
    imp_features = list(rank[i]["feature"])
    scores = pd.DataFrame(list(np.zeros(sampsize)), columns = ["Accuracy"])
    scores["Precision"] = list(np.zeros(sampsize))
    scores["Recall"] = list(np.zeros(sampsize))
    for h in range(sampsize):
            X_train[i], X_test[i], y_train[i], y_test[i] = test_train_split(X[i],y[i], 10, 0.3)
            #print(i+1,i+1,i+1,i+1,i+1,i+1,i+1,i+1,i+1,i+1,i+1,i+1)
            X_train_mod[i] = X_train[i][imp_features]
            X_test_mod[i] = X_test[i][imp_features]
            
            '''ann = tf.keras.models.Sequential()
            ann.add(tf.keras.layers.Dense(units = 7, activation = 'relu'))
            ann.add(tf.keras.layers.Dense(units = 7, activation = 'relu'))
            ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
            ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
            ann.fit(X_train_mod[i], y_train[i], batch_size = 20, epochs = 200)
            y_pred[i] = ann.predict(X_test_mod[i])
            y_pred[i] = (y_pred[i]>0.5)'''
            
            '''log = LogisticRegression(solver = "lbfgs")
            log.fit(X_train_mod[i], y_train[i])
            y_pred[i] = log.predict(X_test_mod[i])'''
            
            '''gnb = GaussianNB(var_smoothing = 0.0001)
            gnb.fit(X_train_mod[i], y_train[i])
            y_pred[i] = gnb.predict(X_test_mod[i])'''
            
            '''rnf = RandomForestClassifier(n_estimators = 15, max_features = 2, class_weight = {0:1,1:1})
            
            n_estimators = [12,15,17,20]
            max_features = [5,6,7,8,9,10,11]
            class_weight = [{0:3,1:1}, {0:4,1:1}, {0:3.5,1:1}, {0:3.711,1:1}, {0:1,1:1}, {0:2,1:1}]
            #criterion = ["gini","entropy"]

            param_grid = dict(n_estimators = n_estimators,
                              max_features = max_features,
                              class_weight = class_weight)
                              #criterion = criterion)
            
            grid = GridSearchCV(estimator = rnf,
                                param_grid = param_grid,
                                scoring = "f1",
                                verbose = 1,
                                n_jobs = -1)
            
            grid_result = grid.fit(X_train_mod[i], y_train[i])
            
            print("Best Score:", grid_result.best_score_)
            print("Best Parameters:", grid_result.best_params_)
            
            _params["params"][i][h] = grid_result.best_params_
            
        
            rnf = RandomForestClassifier(n_estimators = _params["params"][i][h]["n_estimators"], max_features = _params["params"][i][h]["max_features"], class_weight = _params["params"][i][h]["class_weight"])
            rnf.fit(X_train_mod[i], y_train[i])
            y_pred[i] = rnf.predict(X_test_mod[i])'''
            
            svr = SVC(kernel = "linear")
            svr.fit(X_train_mod[i], y_train[i])
            y_pred = svr.predict(X_test_mod[i])
            
            #print(confusion_matrix(y_test[i], y_pred))
            scores["Accuracy"][h] = accuracy_score(y_test[i], y_pred)
            scores["Precision"][h] = precision_score(y_test[i], y_pred)
            scores["Recall"][h] = recall_score(y_test[i], y_pred)
            
            print(i,h, sep = " ")
    
    scrdata = {"Accuracy":scores["Accuracy"], "Precision":scores["Precision"], "Recall":scores["Recall"]}
    
    fig1, ax1 = plt.subplots()
    ax1.boxplot(scrdata.values())
    ax1.set_xticklabels(scrdata.keys())
    plt.title("Age:"+str(i+1))
    
    '''for j in range(ages):
        print(cm[j])
        print("accuracy:",acc[j])
        print("precision:",pre[j])
        print("recall:",rec[j])'''
    



for z in range(ages):
    scrdata = {"Accuracy":scores["Accuracy"][z], "Precision":scores["Precision"][z], "Recall":scores["Recall"][z]}
    
    fig1, ax1 = plt.subplots()
    ax1.boxplot(scrdata.values())
    ax1.set_xticklabels(scrdata.keys())
    plt.title("Age:"+str(z+1))


'''[          feature  median_diff  shape_diff  modal_diff  skew_diff  Combo_Gauge
0    freq.rangesz     0.814445    0.741052    1.000000   0.478007     1.741052
1     mass.region     0.969968    0.926299    0.762152   1.000000     1.688452
2  region.minaxis     1.000000    1.000000    0.638011   0.812527     1.638011
3      mass.total     0.766561    0.820709    0.751364   0.403153     1.572073
4       mass.perc     0.901272    0.872639    0.538630   0.724833     1.411269,             feature  median_diff  ...  skew_diff  Combo_Gauge
0  freq.maxsnr.freq     0.962928  ...   0.983994     1.815520
1    region.minaxis     0.987435  ...   0.742868     1.677627
2      freq.rangesz     0.770176  ...   0.602764     1.644630
3        mass.total     0.771140  ...   0.212618     1.582666
4       mass.region     0.921899  ...   0.813385     1.483573

[5 rows x 6 columns],             feature  median_diff  ...  skew_diff  Combo_Gauge
0   freq.range.high     0.000000  ...   0.547887     1.762179
1      freq.rangesz     0.943275  ...   0.990878     1.758056
2  freq.maxsnr.freq     0.911059  ...   0.508924     1.613941
3            length     0.975367  ...   0.374369     1.503718
4    region.minaxis     0.794270  ...   0.379932     1.480601

[5 rows x 6 columns],             feature  median_diff  ...  skew_diff  Combo_Gauge
0    freq.integrate     0.589518  ...   0.214563     1.628883
1  freq.maxsnr.freq     0.770477  ...   0.213025     1.608313
2  spatial.COMdom.x     0.109838  ...   0.062520     1.579010
3      freq.rangesz     0.604201  ...   0.169924     1.530849
4    region.minaxis     0.758634  ...   0.162129     1.450145

[5 rows x 6 columns],              feature  median_diff  ...  skew_diff  Combo_Gauge
0   spatial.COMdom.x     0.868895  ...   0.635241     1.883492
1   freq.maxsnr.freq     0.469039  ...   0.082533     1.794414
2     freq.integrate     0.857970  ...   0.959057     1.611752
3  region.centroid.0     0.797297  ...   0.578798     1.585994
4       freq.rangesz     1.000000  ...   0.943171     1.432800

[5 rows x 6 columns],             feature  median_diff  ...  skew_diff  Combo_Gauge
0  spatial.COMdom.x     0.252135  ...   0.095225     1.891724
1  freq.maxsnr.freq     0.883431  ...   0.822676     1.667038
2            length     1.000000  ...   0.841877     1.627217
3    freq.integrate     0.924385  ...   1.000000     1.613235
4      freq.rangesz     0.713727  ...   0.556998     1.597500

[5 rows x 6 columns],           feature  median_diff  shape_diff  modal_diff  skew_diff  Combo_Gauge
0    freq.rangesz     1.000000    0.877837    1.000000   0.680484     1.877837
1          length     0.773611    0.764847    0.976365   0.330084     1.741212
2  region.minaxis     0.887897    0.850161    0.723358   0.663091     1.573519
3  freq.integrate     0.670595    0.912162    0.606556   0.988949     1.518718
4      mass.total     0.651503    0.798627    0.707622   0.635051     1.506250,           feature  median_diff  shape_diff  modal_diff  skew_diff  Combo_Gauge
0          length     0.832827    0.913714    1.000000   0.191714     1.913714
1  region.minaxis     0.718326    0.747252    0.873710   0.191117     1.620962
2    freq.rangesz     0.817057    0.748278    0.767944   0.161746     1.516222
3      mass.total     0.654430    0.658379    0.745256   0.266573     1.403635
4  freq.integrate     0.591622    0.714607    0.652985   0.241570     1.367592,           feature  median_diff  shape_diff  modal_diff  skew_diff  Combo_Gauge
0  region.minaxis     0.864644    0.716616    1.000000   0.712458     1.716616
1          length     0.729078    0.928955    0.768293   0.165880     1.697247
2  freq.integrate     0.660456    0.742349    0.935762   0.847222     1.678111
3     spatial.min     1.000000    1.000000    0.652954   0.418160     1.652954
4    temporal.max     0.819060    0.783161    0.787429   0.797549     1.570591,              feature  median_diff  ...  skew_diff  Combo_Gauge
0   spatial.COMdom.x     0.759274  ...   0.762256     1.696225
1   freq.maxsnr.freq     0.649562  ...   0.289148     1.686750
2     freq.integrate     0.737376  ...   0.845927     1.630762
3             length     0.771035  ...   0.225609     1.622965
4  region.centroid.0     0.710071  ...   0.646816     1.585837

[5 rows x 6 columns],             feature  median_diff  ...  skew_diff  Combo_Gauge
0            length     0.925949  ...   0.830819     1.699899
1  freq.maxsnr.freq     0.727417  ...   0.322221     1.669480
2      freq.rangesz     1.000000  ...   0.938038     1.647300
3  spatial.COMdom.y     0.465801  ...   0.523554     1.645972
4    region.minaxis     0.879812  ...   0.723196     1.448039

[5 rows x 6 columns],             feature  median_diff  ...  skew_diff  Combo_Gauge
0  spatial.COMdom.y     0.160467  ...   0.370685     1.672098
1  freq.maxsnr.freq     0.742637  ...   0.015158     1.600411
2            length     1.000000  ...   0.360719     1.594176
3         mass.perc     0.737205  ...   0.553563     1.520799
4        mass.total     0.556116  ...   0.746250     1.473080

[5 rows x 6 columns],              feature  median_diff  ...  skew_diff  Combo_Gauge
0   spatial.COMdom.y     0.503864  ...   0.239235     1.819970
1  region.centroid.1     0.414033  ...   0.129722     1.677365
2    freq.range.high     0.101481  ...   0.379614     1.624229
3     region.minaxis     1.000000  ...   0.395013     1.596375
4   freq.maxsnr.freq     0.758986  ...   0.284013     1.574601

[5 rows x 6 columns],              feature  median_diff  ...  skew_diff  Combo_Gauge
0  region.centroid.0     0.942076  ...   1.000000     1.837924
1   spatial.COMdom.x     0.946551  ...   0.980619     1.788063
2        mass.region     0.946325  ...   0.436245     1.584345
3   freq.maxsnr.freq     0.485444  ...   0.191280     1.535820
4             length     0.981187  ...   0.856022     1.535002

[5 rows x 6 columns]]'''
    
    
