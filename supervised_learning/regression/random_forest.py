import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import OrdinalEncoder
def randomForest(data, categorical_features, numeric_features, target):
    X = data[categorical_features + numeric_features]
    y = data[target]

    # Encoding delle variabili categoriche
    encoder = OrdinalEncoder()
    X.loc[:, categorical_features] = encoder.fit_transform(X[categorical_features])

# Feature extraction
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    random_forest_model = RandomForestRegressor(n_estimators=35, random_state=42)#non metto il criterio perche cosi adotta il mean squared error che è il criterio standard
    random_forest_model.fit(X_train, y_train)

    # Model predictions
    random_forest_y_pred = random_forest_model.predict(X_test)
    print("--------------------- Random Forest Regressor ---------------------\n")
    print("MAE:", mean_absolute_error(y_test, random_forest_y_pred))
    print("MSE:", mean_squared_error(y_test, random_forest_y_pred, squared=False))
    print("R^2:",r2_score(y_test, random_forest_y_pred))
    print("MAPE:", mean_absolute_percentage_error(y_test, random_forest_y_pred))
    print("\n---------------------------------------------------------")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test, alpha=0.5, label="Real values")
    plt.scatter(y_test, random_forest_y_pred, alpha=0.5, label="Predictions")
    plt.xlabel("Real values")
    plt.ylabel("Predictions")
    plt.title("Random forest regression results")
    plt.legend()
    plt.savefig("images/graph/rf_graph.png")
    return random_forest_model