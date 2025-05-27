import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class Model:
    def __init__(self,test_size=0.2):
        # Load data         
        self.x, self.y = self.load_data()
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=test_size)
        self.model =  self.train_model(x_train, y_train)
        # Evaluate
        self.accuracy =  self.evaluate_model(x_test, y_test)

    def load_data(self):
        df = pd.read_csv("data.csv")#change data path accordingly
    
        # Drop customerID (non-informative) and rows with missing TotalCharges
        df = df.dropna()
        df = df.drop("customerID", axis=1)

        # Convert target to binary
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

        # One-hot encode categorical variables
        df = pd.get_dummies(df)
        # print("Data during training:")
        # print(df)
        x = df.drop("Churn", axis=1)
        y = df["Churn"]
        return x, y

    def train_model(self,x_train, y_train, n_estimators=100):
        model = RandomForestClassifier(n_estimators=n_estimators)
        model.fit(x_train, y_train)
        return model

    def evaluate_model(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    
    def predict(self,data):
        processed_data = self.preprocess_input_csv(data)
        return self.model.predict(processed_data)

    def log_feature_importance(self, feature_names):
        importance = self.model.feature_importances_
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
        importance_df = importance_df.sort_values(by="Importance", ascending=False)
        print("\nTop Important Features:\n", importance_df.head(10))  # <-- new
        plt.barh(feature_names, importance)
        plt.title("Feature Importance")
        plt.savefig("feature_importance.png")
        plt.close()

    def preprocess_input_csv(self, data):
        df = None
        try:
            if isinstance( data, pd.DataFrame ):#Here we assume that data is incomming from mcp_server
                data = data.iloc[:-1]
                # print("Data that we received: ")
                # print(data)
                df = data
            else:#Here data is actually the data path when we manually run the model.py file
                df = pd.read_csv(data)

            df = df.drop("customerID", axis=1)
            #Convert the Encodical values
            df = pd.get_dummies(df)
            
            # Align with training features
            df = df.reindex(columns=self.x.columns, fill_value=0)
            # print("Data frame from model:")
            # print(df)
            # if df.columns is self.x.columns:
            #     print("Parsed Data Frame correctly")
            return df
        except Exception as e:
            print(f"Error occured during preprocessing : {e}")

    
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--predict_data_path", type=str, default="predict.csv")
    args = parser.parse_args()
    model = Model(args.test_size)
    model.log_feature_importance(model.x.columns)
    accuracy_score = model.accuracy
    print(f"Model is trained successfully with accuracy score: {accuracy_score}")
    # print(args.predict_data_path)
    predicted = model.predict(args.predict_data_path)
    print(f"Predicted : {predicted}")

if __name__ == "__main__":
    run()