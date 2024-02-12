import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVR

def linear_model(data, categorical_features, numeric_features, target):
    X = data[categorical_features + numeric_features]
    y = data[target]

    # Encoding delle variabili categoriche
    encoder = OrdinalEncoder()
    X.loc[:, categorical_features] = encoder.fit_transform(X[categorical_features])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    linear_model = LinearRegression() 
    linear_model.fit(X_train, y_train)

    # Model predictions
    y_pred = linear_model.predict(X_test)
    print("--------------------- Logistic Regression ---------------------\n")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred, squared=False))
    print("R^2:",r2_score(y_test, y_pred))
    print("\n---------------------------------------------------------")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test, alpha=0.5, label="Real values")
    plt.scatter(y_test, y_pred, alpha=0.5, label="Predictions")
    plt.xlabel("Real values")
    plt.ylabel("Predictions")
    plt.title("Linear regression results")
    plt.legend()
    plt.savefig("images/graph/linear_model.png")
    return linear_model

