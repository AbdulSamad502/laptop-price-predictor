import pandas as pd
import math
import optuna
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

MODEL_PATH = "model_pipeline.pkl"
DATA_PATH = "clean_data.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH):

    print("Model already trained. Loading saved files...")

    pipe = pickle.load(open(MODEL_PATH,"rb"))
    df = pickle.load(open(DATA_PATH,"rb"))

else:

    print("No trained model found. Training model now...")

    df=pd.read_csv("laptop_data.csv")

    df=df.drop(columns=['Unnamed: 0'],axis=1)
    df['Ram']=df['Ram'].str.replace('GB','').astype('Int32')
    df['Weight']=df['Weight'].str.replace('kg','').astype(float)

    df['Touchscreen']=df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
    df['Ips']=df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)
    df['X_res']=df['ScreenResolution'].str.split('x').str.get(-2).str.split(' ').str.get(-1).astype('Int32')
    df['Y_res']=df['ScreenResolution'].str.split('x').str.get(-1).astype('Int32')

    df['ppi']=(((df['X_res']**2)+(df['Y_res']**2))**0.5/df['Inches']).astype(float)
    df=df.drop(columns=['ScreenResolution','Inches','X_res','Y_res'])

    df['Cpu Name']=df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))

    def preprocee(text):
        if text=='Intel Core i5' or text=='Intel Core i7' or text=='Intel Core i3':
            return text
        else:
            if text.split()[0]=="Intel":
                return 'Other Intel Processor'
            else:
                return 'AMD Processor'

    df['Cpu Brand']=df['Cpu Name'].apply(preprocee)
    df.drop(columns=['Cpu','Cpu Name'],inplace=True)

    def preprocess_memory(df):

        df['Memory'] = df['Memory'].astype(str).replace(r'\.0', '', regex=True)
        df["Memory"] = df["Memory"].str.replace('GB', '', regex=False)
        df["Memory"] = df["Memory"].str.replace('TB', '000', regex=False)

        new = df["Memory"].str.split("+", n=1, expand=True)

        df["first"] = new[0].str.strip()
        df["second"] = new[1]

        df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
        df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
        df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
        df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

        df['first'] = df['first'].str.replace(r'\D', '', regex=True)

        df["second"] = df["second"].fillna("0")

        df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
        df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
        df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
        df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

        df['second'] = df['second'].str.replace(r'\D', '', regex=True)

        df["first"] = df["first"].astype(int)
        df["second"] = df["second"].astype(int)

        df["HDD"] = df["first"]*df["Layer1HDD"] + df["second"]*df["Layer2HDD"]
        df["SSD"] = df["first"]*df["Layer1SSD"] + df["second"]*df["Layer2SSD"]
        df["Hybrid"] = df["first"]*df["Layer1Hybrid"] + df["second"]*df["Layer2Hybrid"]
        df["Flash_Storage"] = df["first"]*df["Layer1Flash_Storage"] + df["second"]*df["Layer2Flash_Storage"]

        df.drop(columns=[
            'first','second',
            'Layer1HDD','Layer1SSD','Layer1Hybrid','Layer1Flash_Storage',
            'Layer2HDD','Layer2SSD','Layer2Hybrid','Layer2Flash_Storage'
        ], inplace=True)

    preprocess_memory(df)

    df.drop(columns=['Memory','Hybrid','Flash_Storage'],inplace=True)

    df['Gpu Brand']=df['Gpu'].apply(lambda x:x.split()[0])
    df=df[df['Gpu Brand']!='ARM']

    df.drop(columns=['Gpu'],inplace=True)

    def cat_os(text):

        if text=='Windows 10' or text=='Windows 7' or text=='Windows 10 S':
            return 'Windows'
        else:
            if text=='Mac OS X' or text=='macOS':
                return 'Mac'
            else:
                return 'Others/No OS/Linux'

    df['Os']=df['OpSys'].apply(cat_os)

    df.drop(columns=['OpSys'],inplace=True)

    x=df.drop(columns=['Price'])
    y=np.log(df['Price'])

    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.15,random_state=2)

    trf1=ColumnTransformer(transformers=[
        ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
    ],remainder='passthrough')

    best_model = XGBRegressor(
        n_estimators=260,
        learning_rate=0.08260003146834576,
        max_depth=10,
        min_child_weight=7,
        subsample=0.6628251630266014,
        colsample_bytree=0.6957367288453935,
        gamma=0.0027218854758875415,
        reg_alpha=0.15284675655491556,
        reg_lambda=2.841736847189042,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("transform", trf1),
        ("model", best_model)
    ])

    pipe.fit(x_train, y_train)

    y_pred = pipe.predict(x_test)

    scores = cross_val_score(
        pipe,
        x_train,
        y_train,
        cv=5,
        scoring="r2"
    )

    print("Cross Validation Scores:",scores)
    print("Mean R2 Score:",np.mean(scores))
    print("Std Dev:",np.std(scores))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)

    with open(DATA_PATH, "wb") as f:
        pickle.dump(df, f)

    print("Model training completed and files saved.")