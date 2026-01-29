import os

# Custom modules
from src.utils.exception import CustomException
from src.utils.logger import logging
from src.utils.main_utils import save_df
import sys
from datetime import datetime

# Data manipulation libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline



file_path = os.getenv("MONGO_INGESTED_DATA_LOCATION")

# Importing the data
def import_data(file_path: str) -> pd.DataFrame:
    """
    Import data from a PARQUET file and return it as a pandas DataFrame.
    """
    try:
        df = pd.read_parquet(file_path)
        logging.info(f"Data **imported** from {file_path} successfully.")
        return df
    except Exception as e:
        logging.error(f"Error occurred while **importing** data from {file_path}: {e}")
        raise CustomException(e, sys)


"""
Handeling null values
Handeling duplicate values
Removing ['Z_CostContact', 'Z_Revenue', 'ID', '_id'-> MongoId] columns
"""

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.dropna()
        df = df.drop_duplicates()
        df = df.drop(columns=['Z_CostContact', 'Z_Revenue', 'ID', '_id'], axis=1)
        logging.info("Data **transformed** successfully.")
        return df
    except Exception as e:
        logging.error(f"Error occurred while **transforming** data: {e}")
        raise CustomException(e, sys)


def featureCreation_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # create new column for feature
        ##  creating a new field to store the Age of the customer
        df['Age']=2026 - df['Year_Birth']   

        ###  recoding the customer's education level to numeric form (0: high-school, 1: diploma, 2: bachelors, 3: masters, and 4: doctorates)
        df["Education"] = df["Education"].map({
            "Basic": 0,
            "2n Cycle": 1,
            "Graduation": 2,
            "Master": 3,
            "PhD": 4
        }).astype("int64")

        ###  recoding the customer's marital status to binary form (1: married or together, 0: others)
        df["Marital_Status"] = df["Marital_Status"].map({
            "Married": 1,
            "Together": 1,
            "Absurd": 0,
            "Widow": 0,
            "YOLO": 0,
            "Divorced": 0,
            "Single": 0,
            "Alone": 0
        }).astype("int64")


        #  creating a new field to store the number of children in the household
        df['Children']=df['Kidhome']+df['Teenhome']

        #creating Family_Size
        df['Family_Size']=df['Marital_Status']+df['Children']+1

        #  creating a new field to store the total spending of the customer
        df['Total_Spending']=df["MntWines"]+ df["MntFruits"]+ df["MntMeatProducts"]+ df["MntFishProducts"]+ df["MntSweetProducts"]+ df["MntGoldProds"]
        df["Total Promo"] =  df["AcceptedCmp1"]+ df["AcceptedCmp2"]+ df["AcceptedCmp3"]+ df["AcceptedCmp4"]+ df["AcceptedCmp5"]

        ## The following code works out how long the customer has been with the company and store the total number of promotions the customers responded to
        df['Dt_Customer']=pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
        today=datetime.today()
        df['Days_as_Customer']=(today-df['Dt_Customer']).dt.days
        df['Offers_Responded_To']=df['AcceptedCmp1']+df['AcceptedCmp2']+df['AcceptedCmp3']+df['AcceptedCmp4']+df['AcceptedCmp5']+df['Response']
        df["Parental Status"] = np.where(df["Children"] > 0, 1, 0)

        #dropping columns which are already used to create new features
        columns_to_drop = ['Year_Birth','Kidhome','Teenhome']
        df.drop(columns = columns_to_drop, axis = 1, inplace=True)
        df.rename(columns={"Marital_Status": "Marital Status","MntWines": "Wines","MntFruits":"Fruits",
                        "MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets",
                        "MntGoldProds":"Gold","NumWebPurchases": "Web","NumCatalogPurchases":"Catalog",
                        "NumStorePurchases":"Store","NumDealsPurchases":"Discount Purchases"},
                inplace = True)

        df = df[["Age","Education","Marital Status","Parental Status","Children","Income","Total_Spending","Days_as_Customer","Recency","Wines","Fruits","Meat","Fish","Sweets","Gold","Web","Catalog","Store","Discount Purchases","Total Promo","NumWebVisitsMonth"]]
        logging.info("Feature creation completed successfully.")
        return df
    except Exception as e:
        logging.error(f"Error occurred during feature creation: {e}")
        raise CustomException(e, sys)
    

# Outlier capping using IQR method
def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    try:
        for feature in df.select_dtypes(include=['int64', 'float64']).columns:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)

            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[feature] = df[feature].astype(float)
            df.loc[df[feature] < lower_bound, feature] = lower_bound
            df.loc[df[feature] > upper_bound, feature] = upper_bound

        logging.info("Outliers handled successfully.")
        return df
    except Exception as e:
        logging.error(f"Error occurred while handling outliers: {e}")
        raise CustomException(e, sys)


# Feature Scaling
def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
        outlier_features = ['Wines', 'Fruits', 'Meat', 'Fish','Sweets', 'Gold', 'Total_Spending', "Age"]
        numeric_features= [feature for feature in numeric_features if feature not in outlier_features]

        numeric_pipeline = Pipeline(
            steps=[
                ("Imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("StandardScaler", StandardScaler()),
            ])

        outlier_features_pipeline = Pipeline(
            steps=[
                ("Imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("transformer", PowerTransformer(standardize=True)),
            ])

        preprocessor = ColumnTransformer(
            [
                ("numeric pipeline", numeric_pipeline, numeric_features),
                ("outlier feature pipeline", outlier_features_pipeline, outlier_features),
            ],
            remainder="passthrough",)

        columns = df.columns.tolist()
        df = preprocessor.fit_transform(df)
        df = pd.DataFrame(data=df, columns=columns)
        logging.info("Feature scaling completed successfully.")
        return df
    except Exception as e:
        logging.error(f"Error occurred during feature scaling: {e}")
        raise CustomException(e, sys)


if __name__ == '__main__':
    df = import_data(file_path)
    df = transform_data(df)
    df = featureCreation_data(df)
    df = handle_outliers(df)
    df = scale_features(df)
    
    processed_file_path = os.getenv("PROCESSED_DATA_LOCATION")
    os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
    save_df(df, processed_file_path)
    print(f"\n\nData transformation completed and saved to {processed_file_path}.")

