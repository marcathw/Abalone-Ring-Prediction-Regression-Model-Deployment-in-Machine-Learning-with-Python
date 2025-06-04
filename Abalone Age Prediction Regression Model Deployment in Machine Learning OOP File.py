import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, r2_score as r2
import pickle

class Preprocessor:
    def __init__(self, filepath = "train.csv"):
        self.filepath = filepath
        self.data = None
        self.ohe_encoder = OneHotEncoder(sparse_output = False)
        self.catcols = ["Sex"]
        self.ohe_cols = ["Sex"]

    def read_data(self):
        self.data = pd.read_csv(self.filepath)
        print("Data successfully read.")

    def drop_identifier(self):
        self.data = self.data.drop(columns = ["id"])
        print("Identifier column dropped.")

    def divide(self):
        self.numcols = [col for col in self.data.columns if col not in self.catcols]
        print("Categorical and numerical column divided.")

    def check_duplicates(self):
        print("Total duplicates: ", self.data.duplicated().sum())

    def check_missing_values(self):
        missing = pd.DataFrame({
            "column name": self.data.columns.tolist(),
            "missing values": self.data.isnull().sum().tolist(),
            "percentage": ((self.data.isnull().sum()) * 100 / len(self.data)).round(2).tolist()
        })
        print("Total missing values:")
        print(missing)
    
    def encode(self):
        ohe_df = self.ohe_encoder.fit_transform(self.data[self.ohe_cols])
        ohe_df = pd.DataFrame(ohe_df, columns = self.ohe_encoder.get_feature_names_out(self.ohe_cols))
        self.data = pd.concat([self.data.drop(columns = self.ohe_cols), ohe_df], axis = 1)
        print("Encoding performed.")

    def define_x_y(self):
        x = self.data.drop(columns = ["Rings"])
        y = self.data["Rings"]
        return x, y, self.numcols

class Modelling:
    def __init__(self, x, y, numcols, test_size = 0.3, random_state = 11):
        self.x = x
        self.y = y
        self.numcols = numcols
        self.numcols.remove("Rings")
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size = test_size, random_state = random_state)
        self.scaler = StandardScaler()
        self.model = LinearRegression()

    def scale(self):
        self.x_train[self.numcols] = self.scaler.fit_transform(self.x_train[self.numcols])
        self.x_test[self.numcols] = self.scaler.transform(self.x_test[self.numcols])
        print("Scaling performed.")

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def evaluate(self):
        self.y_pred = self.model.predict(self.x_test)
        res_dict = {
            'mae': mae(self.y_test, self.y_pred),
            'mse': mse(self.y_test, self.y_pred),
            'rmse': np.sqrt(mse(self.y_test, self.y_pred)),
            'r2': r2(self.y_test, self.y_pred)
        }
        print(res_dict)

    def model_save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

# ----------------------------------------------------
preprocessor = Preprocessor()
preprocessor.read_data()
preprocessor.drop_identifier()
preprocessor.divide()
preprocessor.check_duplicates()
preprocessor.check_missing_values()
preprocessor.encode()
x, y, numcols = preprocessor.define_x_y()

# ----------------------------------------------------
modelling = Modelling(x, y, numcols)
modelling.scale()
modelling.train()
modelling.evaluate()
modelling.model_save("Abalone Age Prediction Regression Model Deployment in Machine Learning Pickle File.pkl")