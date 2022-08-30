import boto3
import datetime as datetime
from io import BytesIO
import joblib
import lightgbm as lgb
import numpy as np
import os
import pandas as pd
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from xgboost import XGBClassifier

import config
from utils.connect_to_s3 import s3


def seed_everything(seed=2022):
    """
    standardises random seed
    :param seed: sets random seed for entire script
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = str(seed)
    tf.random.set_seed(seed)


def load_gbif():
    """
    Loads GBIF Data from S3 Database
    """
    obj = s3.Bucket('adsi-aws-bucket').Object('data/rubus/combined.csv').get()
    df = pd.read_csv(obj['Body'], index_col=0)
    return df


def load_asdi():
    """
    Loads ASDI Data from S3 Database
    """
    obj = s3.Bucket('adsi-aws-bucket').Object('data/gn/aggregate.csv').get()
    temp = pd.read_csv(obj['Body'], index_col=0)
    return temp


def load_nasa():
    """
    Loads NASA Data from S3 Database
    """
    obj = s3.Bucket('adsi-aws-bucket').Object('data/nasa/nasa.csv').get()
    nasa_df = pd.read_csv(obj['Body'], index_col=0)
    return nasa_df


# Mapping of US States
states = {
    'AK': 'Alaska',
    'AL': 'Alabama',
    'AR': 'Arkansas',
    'AZ': 'Arizona',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DC': 'District of Columbia',
    'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia',
    'HI': 'Hawaii',
    'IA': 'Iowa',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'MA': 'Massachusetts',
    'MD': 'Maryland',
    'ME': 'Maine',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MO': 'Missouri',
    'MS': 'Mississippi',
    'MT': 'Montana',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'NE': 'Nebraska',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NV': 'Nevada',
    'NY': 'New York',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VA': 'Virginia',
    'VT': 'Vermont',
    'WA': 'Washington',
    'WI': 'Wisconsin',
    'WV': 'West Virginia',
    'WY': 'Wyoming'
}

us_state_to_abbrev = {
        "Alabama": "AL",
        "Alaska": "AK",
        "Arizona": "AZ",
        "Arkansas": "AR",
        "California": "CA",
        "Colorado": "CO",
        "Connecticut": "CT",
        "Delaware": "DE",
        "Florida": "FL",
        "Georgia": "GA",
        "Hawaii": "HI",
        "Idaho": "ID",
        "Illinois": "IL",
        "Indiana": "IN",
        "Iowa": "IA",
        "Kansas": "KS",
        "Kentucky": "KY",
        "Louisiana": "LA",
        "Maine": "ME",
        "Maryland": "MD",
        "Massachusetts": "MA",
        "Michigan": "MI",
        "Minnesota": "MN",
        "Mississippi": "MS",
        "Missouri": "MO",
        "Montana": "MT",
        "Nebraska": "NE",
        "Nevada": "NV",
        "New Hampshire": "NH",
        "New Jersey": "NJ",
        "New Mexico": "NM",
        "New York": "NY",
        "North Carolina": "NC",
        "North Dakota": "ND",
        "Ohio": "OH",
        "Oklahoma": "OK",
        "Oregon": "OR",
        "Pennsylvania": "PA",
        "Rhode Island": "RI",
        "South Carolina": "SC",
        "South Dakota": "SD",
        "Tennessee": "TN",
        "Texas": "TX",
        "Utah": "UT",
        "Vermont": "VT",
        "Virginia": "VA",
        "Washington": "WA",
        "West Virginia": "WV",
        "Wisconsin": "WI",
        "Wyoming": "WY",
        "District of Columbia": "DC",
        "American Samoa": "AS",
        "Guam": "GU",
        "Northern Mariana Islands": "MP",
        "Puerto Rico": "PR",
        "United States Minor Outlying Islands": "UM",
        "U.S. Virgin Islands": "VI",
    }

def process_and_merge(df, temp, states):
    """
    Merges GBIF Target(df) with ASDI Temperature and Precipitation Data (temp)
    """

    # Drop data without states after running through script 1 to map Lat Lon to State and drop absence data
    df = df[(df["newstateProvince"].notna()) & (df["occurrenceStatus"]=="PRESENT")] 

    # Finish Loading Fullstate Name and getting year/month/day
    temp["fullstate_name"] = temp["state"].apply(lambda x: states[x])

    temp["year"] = temp["date"].apply(lambda x: str(x)[:4])
    temp["month"] = temp["date"].apply(lambda x: str(x)[4:6])
    temp["day"] = temp["date"].apply(lambda x: str(x)[6:])

    temp["year"] = temp["year"].astype(int)
    temp["month"] = temp["month"].astype(int)
    temp["day"] = temp["day"].astype(int)


    # Select Useful Columns and merge
    temp = temp[["date", "year", 'month', 'day', 'PRCP', 'TAVG', 'TMAX', "TMIN", 'fullstate_name']]
    df = df[["year", "month", "day", "newstateProvince", 'scientificName']]

    testing_df = df.merge(temp, how='left', left_on=["year", 'month', 'day', 'newstateProvince'], right_on=["year", 'month', 'day', "fullstate_name"])
    
    # Dropna and take 2009 + Data
    testing_df = testing_df[testing_df["year"]>=2009].dropna(subset=["TMAX", "TMIN", "PRCP"])
    print(testing_df.shape)
    return testing_df




def generate_absence_data(states):
    """
    Generate Absence Data to combine with presence data in previous function
    """
    # Generating Absence Data from ASDI Temp
    absence_list = [i for i in list(states.values()) if i not in testing_df["newstateProvince"].unique().tolist()]

    for i in testing_df["newstateProvince"].unique().tolist():
        if len(testing_df[testing_df["newstateProvince"] == i]) <= 5:
            if i not in absence_list:
                absence_list.append(i)

    addon_df = temp[(temp["fullstate_name"].isin(absence_list))].dropna(subset=["TMAX", "TMIN", "PRCP"])
    temp_df = addon_df.groupby("fullstate_name").apply(lambda x: x.sample(frac=0.05, random_state=2022))

    temp_df.drop(columns={"fullstate_name"}, inplace=True)
    temp_df = temp_df.reset_index()
    print(temp_df.shape)
    return temp_df


def merge_absence_and_presence(testing_df, temp_df):
    """
    Processes the Data (Clean Row Names and Adding Target Variables) &
    Merges Absence data with the Presence Data to obtain balanced dataset
    """
    # Align Columns to Concatenate
    updated_cols = ["newstateProvince", "date", "year", "month", "day", "PRCP", "TAVG", "TMAX", "TMIN"]

    testing_df.drop(columns={"scientificName", "fullstate_name"}, inplace=True)
    temp_df.drop(columns={"level_1"}, inplace=True)
    temp_df.rename(columns={"fullstate_name":"newstateProvince"}, inplace=True)

    testing_df = testing_df[updated_cols]
    # Groupby Date and State
    testing_df = testing_df.groupby(["year", "month", "day", "newstateProvince"]).agg("mean").reset_index()
    temp_df = temp_df.groupby(["year", "month", "day", "newstateProvince"]).agg("mean").reset_index()

    # Add target labels
    testing_df["target"] = 1
    temp_df["target"] = 0
    # Concat DF
    merged_df = testing_df.append(temp_df)
    return merged_df


def convert_DOY(year, doy):
    """
    Converts Nasa df's DOY format to date
    
    :param year: year number
    :param day: number of days into the year
    """
    
    year = int(str(datetime.datetime(year, 1, 1) + datetime.timedelta(doy - 1)).split(" ")[0].split("-")[0])
    month = int(str(datetime.datetime(year, 1, 1) + datetime.timedelta(doy - 1)).split(" ")[0].split("-")[1])
    day = int(str(datetime.datetime(year, 1, 1) + datetime.timedelta(doy - 1)).split(" ")[0].split("-")[2])
    return year, month, day


def process_nasa_data(nasa_df, states):
    """
    Processes data from the NASA Power Project

    :param nasa_df: NASA Dataframe
    :param states: States mapping converting from State code to state name
    """
    nasa_df = nasa_df[nasa_df["YEAR"]>=2009]
    nasa_df = nasa_df.reset_index()
    nasa_df["newstateProvince"] = nasa_df["STATE"].apply(lambda x: states[x])

    # Convert DOY to Date
    nasa_df["year"] = nasa_df.apply(lambda x: convert_DOY(x.YEAR, x.DOY)[0], axis=1)
    nasa_df["month"] = nasa_df.apply(lambda x: convert_DOY(x.YEAR, x.DOY)[1], axis=1)
    nasa_df["day"] = nasa_df.apply(lambda x: convert_DOY(x.YEAR, x.DOY)[2], axis=1)

    nasa_df.drop(columns={"YEAR", "STATE", "DOY", 'index'}, inplace=True)
    return nasa_df


def merge_nasa(merged_df, nasa_df):
    """
    merges GBIF + ASDI Dataframe with the NASA dataframe

    :param merged_df: GBIF + ASDI Dataframe
    :param nasa_df: NASA Dataframe
    """

    merged_df = merged_df.merge(nasa_df, how='left', on=['newstateProvince', 'year', 'month', 'day'])
    merged_df.dropna(subset=["PS"], inplace=True)
    return merged_df





#! Modelling
def feat_eng1(merged_df):
    """
    Fixes Feature format and generates 1 additional features : Temperature Range
    
    :param merged_df: Processed Dataframe without feature engineering
    """
    merged_df["TMAX"] = merged_df["TMAX"] / 10
    merged_df["TMIN"] = merged_df["TMIN"] / 10
    merged_df["TRANGE"] = merged_df["TMAX"] - merged_df["TMIN"]
    merged_df['TAVG'].fillna((merged_df["TMAX"] + merged_df['TMIN']) / 2, inplace=True)
    merged_df = merged_df.sort_values(["year", "month", "day"])

    # Eemove the -999 values (Nan values presumably to save neural network performance)
    merged_df = merged_df[~(merged_df["ALLSKY_SFC_PAR_TOT"]==-999)]
    return merged_df




def get_train_test_and_feature(merged_df):
    """
    Gets the train and test dataset, splittling by year

    :param merged_df: Processed Dataframe
    """
    # Separating Train and test df and get feature columns
    test_df = merged_df[(merged_df["date"]>=20210601)]
    merged_df = merged_df[~((merged_df["date"]>=20210601))]
    feature_cols = [i for i in merged_df if i not in ["newstateProvince", "date", 'year', 'month', 'day', 'target']]
    return merged_df, test_df, feature_cols




def display_statistics(oof_preds, test_preds):
    """
    Displays CV and Test Scores of Model

    :param oof_preds: Out of fold predictions
    :param test_preds: Test predictions
    """
    print("CV Predictions >= 0.5 :", len([i for i in oof_preds if i >= 0.5]))
    print("CV Predictions < 0.5 :", len([i for i in oof_preds if i < 0.5]))
    print("="*50)
    print("Actual CV Predictions >= 0.5 :", len([i for i in merged_df["target"] if i >= 0.5]))
    print("Actual CV Predictions < 0.5 :", len([i for i in merged_df["target"] if i < 0.5]))
    print("="*50)
    print("CV AUC Score :", roc_auc_score(merged_df["target"], oof_preds))
    print()
    print("-"*100)
    print()
    print("Test Predictions >= 0.5 :", len([i for i in test_preds if i >= 0.5]))
    print("Test Predictions < 0.5 :", len([i for i in test_preds if i < 0.5]))
    print("="*50)
    print("Actual Test Predictions >= 0.5 :", len([i for i in test_df["target"] if i >= 0.5]))
    print("Actual Test Predictions < 0.5 :", len([i for i in test_df["target"] if i < 0.5]))
    print("="*50)
    print("Test AUC Score :", roc_auc_score(test_df["target"], test_preds))


def train_lgb():
    test_predictions_lgb = np.zeros(len(test_df))
    oof_predictions_lgb = np.zeros(len(merged_df))


    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
    for fold, (train_index, test_index) in enumerate(skf.split(merged_df[feature_cols], merged_df['target'])):
        print(f"Fold {fold}")
        X_train, X_test = merged_df[feature_cols].iloc[train_index], merged_df[feature_cols].iloc[test_index]
        y_train, y_test = merged_df["target"].iloc[train_index], merged_df["target"].iloc[test_index]
        params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'seed': 2022,
            }
        
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_test, y_test)
        
        model = lgb.train(
                params = params,
                train_set = lgb_train,
                num_boost_round = 10000,
                valid_sets = [lgb_valid],
                early_stopping_rounds = 100,
                verbose_eval = 100,
    #             feval = lgb_metric
                )
        
        joblib.dump(model, f'model_output/lgbm_fold{fold}_seed2022.pkl')
        val_pred = model.predict(X_test)
        oof_predictions_lgb[test_index] = val_pred
        
        test_pred = model.predict(test_df[feature_cols])
        test_predictions_lgb += test_pred / 5
        print("="*50)

    return oof_predictions_lgb, test_predictions_lgb







def train_xgb():
    test_predictions_xgb = np.zeros(len(test_df))
    oof_predictions_xgb = np.zeros(len(merged_df))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
    for fold, (train_index, test_index) in enumerate(skf.split(merged_df[feature_cols], merged_df['target'])):
        print(f"Fold {fold}")
        X_train, X_test = merged_df[feature_cols].iloc[train_index], merged_df[feature_cols].iloc[test_index]
        y_train, y_test = merged_df["target"].iloc[train_index], merged_df["target"].iloc[test_index]
        
        eval_set = [(X_test, y_test)]
        
        xgb_params = { 
    #         'learning_rate':0.05, 
    #         'subsample':0.8,
    #         'colsample_bytree':0.6, 
            'eval_metric':'logloss',
            'random_state':2022,
            'n_estimators':1000,
            'seed':2022
            
        }
        
        model = XGBClassifier(**xgb_params) 
        
        
        
        model.fit(X_train, y_train,
                    eval_set=[(X_test, y_test)], 
                    early_stopping_rounds=50,
                    verbose=False
                    )
        
        joblib.dump(model, f'model_output/xgb_fold{fold}_seed2022.pkl')
        val_pred = model.predict(X_test)
        oof_predictions_xgb[test_index] = val_pred
        
        test_pred = model.predict(test_df[feature_cols])
        test_predictions_xgb += test_pred / 5
        print("="*50)
    return oof_predictions_lgb, test_predictions_lgb







def train_nn():
    test_predictions_nn = np.zeros(len(test_df))
    oof_predictions_nn = np.zeros(len(merged_df))


    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
    for fold, (train_index, test_index) in enumerate(skf.split(merged_df[feature_cols], merged_df['target'])):
        print(f"Fold {fold}")
        X_train, X_test = merged_df[feature_cols].iloc[train_index], merged_df[feature_cols].iloc[test_index]
        y_train, y_test = merged_df["target"].iloc[train_index], merged_df["target"].iloc[test_index]
        
        
        # Data normalization using Minmaxscaler
        mm_scaler = MinMaxScaler()
        X_train_mm = mm_scaler.fit_transform(X_train)
        X_test_mm = mm_scaler.transform(X_test)
        
        # Model checkpoint
        checkpoint = ModelCheckpoint(filepath=f'model_output/NN_Fold{fold}_seed2022.pkl',
                                    monitor='val_loss',verbose=0,save_weights_only=True,save_best_only=True,mode='min')

        # model architecture for ANN model
        model = Sequential()

        model.add(Dense(16, activation="relu"))
        model.add(Dense(8, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))


        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3, decay=1e-3 / 200))
        model.compile(optimizer = "adam", loss = 'binary_crossentropy')

        # Patient early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        print("Training Model ...")
        # Fitting the model on train data
        history = model.fit(X_train_mm, y_train, validation_data=(X_test_mm, y_test), epochs=200, batch_size=128,
                            verbose=1, callbacks=[es,checkpoint])
        print("Training Complete ...")
        
        val_pred = model.predict(X_test_mm)

        oof_predictions_nn[test_index] = [i[0] for i in val_pred]
        
        
        mm_scaler = MinMaxScaler()
        test_mm = mm_scaler.fit_transform(test_df[feature_cols])
        test_pred = model.predict(test_mm)
        test_predictions_nn += [i[0]/5 for i in test_pred]
        print("="*50)
    return oof_predictions_nn, test_predictions_nn




def generate_preds_post_process(test_predictions_lgb, test_predictions_xgb, test_predictions_nn):
    """
    Generates Ensemble Predictions

    :param test_predictions_lgb: LGB Test Predictions
    :param test_predictions_xgb: XGB Test Predictions
    :param test_predictions_nn: NN Test Predictions
    """
    en_df = pd.DataFrame()
    en_df["lgb"] = test_predictions_lgb
    en_df["xgb"] = test_predictions_xgb
    en_df["nn"] = test_predictions_nn

    en_df["ensemble_all_3"] = (en_df["nn"] + en_df["xgb"] + en_df["lgb"]) / 3
    en_df["ensemble_nn_lgb"] = (en_df["nn"] + en_df["lgb"]) / 2



    test_df["ensemble_all_3_preds"] = en_df["ensemble_all_3"].tolist()
    test_df["ensemble_nn_lgb"] = en_df["ensemble_nn_lgb"].tolist()
    test_df["nn_preds"] = en_df["nn"].tolist()
    test_df["xgb_preds"] = en_df["xgb"].tolist()
    test_df["lgb_preds"] = en_df["lgb"].tolist()
    return test_df


def update_s3_db(upload_df, us_state_to_abbrev):
    """
    :params upload_df: Dataframe with state and final ensemble predictions
    :params us_state_to_abbrev: Map State name back to state code
    """
    upload_df["stateCode"] = upload_df["newstateProvince"].apply(lambda k: us_state_to_abbrev[k])
    csv_buffer = BytesIO()
    upload_df.to_csv(csv_buffer)
    output_file = "model-output.csv"
    s3.Object('adsi-aws-bucket', output_file).put(Body=csv_buffer.getvalue(), Key=output_file)


if __name__ == '__main__':    
    #* Set Random Seed for kernel
    seed_everything(seed=2022)
    #* Loads GBIF, ASDI and NASA data
    # #! Load locally
    # df = pd.read_csv(r"data\out_postprocess.csv")
    #! Load Using S3
    df = load_gbif()
    # #! load locally
    # temp = pd.read_csv(r"data\agg.csv")
    #! Load Using S3
    temp = load_asdi()
    # #! Load locally 
    # nasa_df = pd.read_csv(r"data\nasa.csv")
    #! Load Using S3
    nasa_df = load_nasa()
    #* Merges ASDI data with GBIF Data
    testing_df = process_and_merge(df, temp, states)
    #* Generate Absence Data
    temp_df = generate_absence_data(states)
    #* Combines (ASDI + GBIF Presence Data) with the Absence Data
    merged_df = merge_absence_and_presence(testing_df, temp_df)
    #* Processes NASA dataset and merge it with the above dataframe
    nasa_df = process_nasa_data(nasa_df, states)
    merged_df = merge_nasa(merged_df, nasa_df)
    #* Clean and generate simple feature(s), then do a train test split and extract feature columns
    merged_df = feat_eng1(merged_df)
    merged_df, test_df, feature_cols = get_train_test_and_feature(merged_df)
    #* Obtain Model Predictions from LGB, XGB and NN respectively and display the prediction statistics
    oof_predictions_lgb, test_predictions_lgb = train_lgb()
    display_statistics(oof_predictions_lgb, test_predictions_lgb)
    oof_predictions_xgb, test_predictions_xgb = train_xgb()
    display_statistics(oof_predictions_xgb, test_predictions_xgb)
    oof_predictions_nn, test_predictions_nn = train_nn()
    display_statistics(oof_predictions_nn, test_predictions_nn)
    #* Ensemble the Predictions and average them by each State
    test_df = generate_preds_post_process(test_predictions_lgb, test_predictions_xgb, test_predictions_nn)
    upload_df = test_df.groupby("newstateProvince").agg("mean").reset_index()[['newstateProvince', 'ensemble_all_3_preds',]]
    #* Upload the data to Amazon S3
    # #! Save Predictions Locally
    # upload_df.to_csv(r"model_output/updated_model_predsv2.csv", index=False)
    #! Save Predictions to S3 DB
    update_s3_db(upload_df, us_state_to_abbrev)






