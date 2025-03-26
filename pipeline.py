import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.stats as stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
import joblib

class RegressionPipeline:
    def __init__(self, data_path, num_features, cat_features):
        self.data_path = data_path
        self.num_features = num_features
        self.cat_features = cat_features
        self.model_pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self._load_data()
        self._split_data()
        self._build_pipeline()

    def _load_data(self):
        self.data = pd.read_csv(self.data_path)

    def _split_data(self):
        X = self.data[self.num_features + self.cat_features]
        y = self.data['redshift']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    @staticmethod
    def transform_mjd(X):
        mjd = X[:, 0]
        min_mjd = np.min(mjd)
        sequential_days = mjd - min_mjd
        mjd_epoch = pd.Timestamp('1858-11-17')
        dates = pd.to_datetime(mjd, unit='D', origin=mjd_epoch)
        months = dates.month
        month_sin = np.sin(2 * np.pi * months / 12)
        month_cos = np.cos(2 * np.pi * months / 12)
        return np.column_stack([sequential_days, month_sin, month_cos])

    @staticmethod
    def convert_ra_dec(X):
        ra_rad = np.radians(X[:, 0])
        dec_rad = np.radians(X[:, 1])
        x = np.cos(dec_rad) * np.cos(ra_rad)
        y = np.cos(dec_rad) * np.sin(ra_rad)
        z = np.sin(dec_rad)
        return np.column_stack([x, y, z])

    @staticmethod
    def clean_class_column(X):
        X = pd.DataFrame(X, columns=["class"])
        mapping = {"G": "GALAXY", "S": "STAR", "QSO": "QUASAR"}
        X["class"] = X["class"].replace(mapping)
        return X.to_numpy()

    def _build_pipeline(self):
        mjd_transformer = FunctionTransformer(self.transform_mjd, validate=False)
        ra_dec_transformer = FunctionTransformer(self.convert_ra_dec, validate=False)
        class_cleaner = FunctionTransformer(self.clean_class_column, validate=False)

        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('mjd_transform', mjd_transformer),
            ('ra_dec_transform', ra_dec_transformer),
            ('scaler', StandardScaler())
        ])

        cat_transformer = Pipeline(steps=[
            ("clean_class", class_cleaner),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', num_transformer, self.num_features),
            ('cat', cat_transformer, self.cat_features)
        ])

        self.model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])

    def train(self):
        self.model_pipeline.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model_pipeline.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return {"MAE": mae, "MSE": mse, "R2": r2}

    def predict(self, X):
        return self.model_pipeline.predict(X)

    def save_model(self, filename):
        joblib.dump(self.model_pipeline, filename, compress=3)



"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge  
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

class RegressionPipeline:
    def __init__(self, data_path, num_features, cat_features):
        self.data_path = data_path
        self.num_features = num_features
        self.cat_features = cat_features
        self.model_pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self._load_data()
        self._split_data()
        self._build_pipeline()

    def _load_data(self):
        self.data = pd.read_csv(self.data_path)

    def _split_data(self):
        X = self.data[self.num_features + self.cat_features]
        y = self.data['redshift']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    @staticmethod
    def transform_mjd(X):
        mjd = X[:, 0]
        min_mjd = np.min(mjd)
        sequential_days = mjd - min_mjd
        mjd_epoch = pd.Timestamp('1858-11-17')
        dates = pd.to_datetime(mjd, unit='D', origin=mjd_epoch)
        months = dates.month
        month_sin = np.sin(2 * np.pi * months / 12)
        month_cos = np.cos(2 * np.pi * months / 12)
        return np.column_stack([sequential_days, month_sin, month_cos])

    @staticmethod
    def convert_ra_dec(X):
        ra_rad = np.radians(X[:, 0])
        dec_rad = np.radians(X[:, 1])
        x = np.cos(dec_rad) * np.cos(ra_rad)
        y = np.cos(dec_rad) * np.sin(ra_rad)
        z = np.sin(dec_rad)
        return np.column_stack([x, y, z])

    @staticmethod
    def clean_class_column(X):
        X = pd.DataFrame(X, columns=["class"])
        mapping = {"G": "GALAXY", "S": "STAR", "QSO": "QUASAR"}
        X["class"] = X["class"].replace(mapping)
        return X.to_numpy()

    def _build_pipeline(self):
        mjd_transformer = FunctionTransformer(self.transform_mjd, validate=False)
        ra_dec_transformer = FunctionTransformer(self.convert_ra_dec, validate=False)
        class_cleaner = FunctionTransformer(self.clean_class_column, validate=False)

        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('mjd_transform', mjd_transformer),
            ('ra_dec_transform', ra_dec_transformer),
            ('scaler', StandardScaler())
        ])

        cat_transformer = Pipeline(steps=[
            ("clean_class", class_cleaner),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', num_transformer, self.num_features),
            ('cat', cat_transformer, self.cat_features)
        ])

        self.model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', BayesianRidge()) 
        ])

    def train(self):
        self.model_pipeline.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model_pipeline.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return {"MAE": mae, "MSE": mse, "R2": r2}

    def predict(self, X):
        return self.model_pipeline.predict(X)

    def save_model(self, filename):
        joblib.dump(self.model_pipeline, filename, compress=3)
"""