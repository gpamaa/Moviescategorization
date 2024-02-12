import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import GradientBoostingRegressor
def gradient_boosting(data, categorical_features, numeric_features, target):
    X = data[categorical_features + numeric_features]
    y = data[target]

    # Encoding delle variabili categoriche
    encoder = OrdinalEncoder()
    X.loc[:, categorical_features] = encoder.fit_transform(X[categorical_features])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    gradient_boosting_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gradient_boosting_model.fit(X_train, y_train)
    gradient_boosting_y_pred = gradient_boosting_model.predict(X_test)
    print("--------------------- Gradient Boosting Regressor ---------------------\n")
    print("MAE:", mean_absolute_error(y_test, gradient_boosting_y_pred))
    print("MSE:", mean_squared_error(y_test, gradient_boosting_y_pred, squared=False))
    print("R^2:",r2_score(y_test, gradient_boosting_y_pred))
    print("\n---------------------------------------------------------")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test, alpha=0.5, label="Real values")
    plt.scatter(y_test, gradient_boosting_y_pred, alpha=0.5, label="Predictions")
    plt.xlabel("Real values")
    plt.ylabel("Predictions")
    plt.title("gradient boosting regression results")
    plt.legend()
    plt.savefig("images/graph/gb_graph.png")
    return gradient_boosting_model
