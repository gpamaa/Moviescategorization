from matplotlib import pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import networkx as nx
import itertools
import numpy as np
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
data = pd.read_csv("dataset/generated_dataset/generated_dataset.csv")


# Selezionare le feature e la variabile target


data['director_encoded'] = pd.factorize(data['director'])[0]
data['star_encoded'] = pd.factorize(data['star'])[0]
data['category_encoded'] = pd.factorize(data['Category'])[0]
data['genre_encoded'] = pd.factorize(data['genre'])[0]
score_bins = [0.0, 5.5, 7.5, 10.0]
score_labels = [1,2,3]
data['score_encoded'] = pd.cut(data['score'], bins=score_bins, labels=score_labels)
votes_bins = [0, np.mean(data["votes"]), data['votes'].max()]
votes_labels = [1,2]
data['votes_encoded'] = pd.cut(data['votes'], bins=votes_bins, labels=votes_labels)
print(data['genre_encoded'].unique())
print(data['score_encoded'].unique())
categorical_features = ["director_encoded","category_encoded"]
numeric_features = ["score_encoded","votes_encoded"]

X = data[categorical_features + numeric_features]
model = BayesianNetwork([
    ('director_encoded', 'category_encoded'),
    ('star_encoded', 'category_encoded'),
    ('score_encoded', 'category_encoded'),
    ('director_encoded', 'votes_encoded')
])
variable_card = {'director_encoded': len(data['director_encoded'].unique()), 'category_encoded': len(data['category_encoded'].unique()), 'star_encoded': len(data['star_encoded'].unique()),'votes_encoded': len(data['votes_encoded'].unique())}
# Definisci le CPD per ogni variabile
cpds = []

estimator = MaximumLikelihoodEstimator(model,data)
cpd_director=estimator.estimate_cpd('director_encoded')
cpd_star=estimator.estimate_cpd('star_encoded')
cpd_votes=estimator.estimate_cpd('votes_encoded')
cpd_score=estimator.estimate_cpd('score_encoded')
cpd_cat=estimator.estimate_cpd('category_encoded')
# Aggiungi le CPD alla rete bayesiana
model.add_cpds(cpd_score,cpd_cat,cpd_director,cpd_star,cpd_votes)
inference = VariableElimination(model)
result = inference.query(variables=['category_encoded'], evidence={'score_encoded': 1, 'star_encoded': 3,'director_encoded':5,'votes_encoded': 2})

print(result)
if model.check_model():
    print("\n La rete è valida \n")

    # Creazione del grafo
    G = nx.DiGraph()
    G.add_nodes_from(model.nodes())
    G.add_edges_from(model.edges())

    # Calcolo della posizione dei nodi
    pos = nx.spring_layout(G, k=1500, iterations=10000)

    # Disegno del grafo
    plt.figure(figsize=(8, 6))

    # Disegno del grafo
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='grey', font_weight='bold')

    # Regola la posizione dei nodi per aggiungere un margine intorno ai nodi
    x_vals, y_vals = zip(*pos.values())
    x_max, x_min = max(x_vals), min(x_vals)
    y_max, y_min = max(y_vals), min(y_vals)
    x_margin = (x_max - x_min) * 0.2
    y_margin = (y_max - y_min) * 0.2
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)

    # Aggiunge margini a sinistra e a destra
    x_size = x_max - x_min
    y_size = y_max - y_min
    max_size = max(x_size, y_size)
    left_margin = (max_size - x_size) / 2
    right_margin = max_size - x_size - left_margin
    plt.xlim(x_min - left_margin - x_margin, x_max + right_margin + x_margin)
    

    path = "images/belief_network.png" 
    plt.savefig(path)
    infer = VariableElimination(model)

   


    

else:
    print("\n La rete non è valida \n")
    