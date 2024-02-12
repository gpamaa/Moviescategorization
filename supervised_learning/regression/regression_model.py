from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from random_forest import randomForest
from gradient_boosting import gradient_boosting
from linear_model import linear_model
#Calcolo delle feature importance
def printFeatureRanking(clf, X, path):
    if isinstance(clf, LinearRegression):
        importances = abs(clf.coef_) #il regressore lineare utilizza il coefficente per valutare l'importanza delle feature
    else:
        importances = clf.feature_importances_ #feature importances è invece usato per valutare l'importanza delle feature in modelli che usano alberi decisionali come il random forest e il gradient boosting
    indices = np.argsort(importances)[::-1] #ordina le feature in base all'importanza
    print("\nFeature ranking:")
    for f in range(0, 5): 
        print("%d. %s (%f)" % (f+1, X.columns[indices[f]], importances[indices[f]]))# stampa le prime 5 feature piu importanti

    # creazione del grafico
    plt.clf()
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices], color="orange", align="center")
    plt.xticks(range(X.shape[1]), [categorical_features[i] if i<len(categorical_features) else numeric_features[i-len(categorical_features)] for i in indices], rotation=90, fontsize=8)
    plt.xlim([-1, X.shape[1]])
    plt.subplots_adjust(bottom=0.4)

    plt.savefig(path)
data = pd.read_csv("dataset/generated_dataset/generated_dataset.csv")


# Selezionare le feature e la variabile target
categorical_features = ["genre","director","writer","star","country","company","month"]
numeric_features = ["year","score","votes","runtime","budget"]
target = "Earn"
X = data[categorical_features + numeric_features]



print("\n---RANDOM FOREST---")
clf = randomForest(data, categorical_features, numeric_features, target)
printFeatureRanking(clf, X, "images/features/regression/rf_feature.png" )

print("\n---GRADIENT BOOSTING---")
clf = gradient_boosting(data, categorical_features, numeric_features, target)
printFeatureRanking(clf, X, "images/features/regression/gb_feature.png" )

print("\n---LINEAR REGRESSOR---")
clf = linear_model(data, categorical_features, numeric_features, target)
printFeatureRanking(clf, X, "images/features/regression/linear_feature.png" )

