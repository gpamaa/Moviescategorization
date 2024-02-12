from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.cm as cm

data = pd.read_csv("dataset/generated_dataset/generated_dataset.csv")


# Selezionare le feature e la variabile target
categorical_features = ["genre","director","writer","star","country","company","month"]
numeric_features = ["year","score","votes","runtime"]

X = data[categorical_features + numeric_features]

# Encoding delle variabili categoriche
encoder = OrdinalEncoder()
X.loc[:, categorical_features] = encoder.fit_transform(X[categorical_features])

# Calcola l'inerzia (somma dei quadrati delle distanze tra ogni osservazione e il centroide del suo cluster) per diversi valori di k
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0,  n_init=10).fit(X)
    inertias.append(kmeans.inertia_)
    
    # stampa nel numero di esempi di ogni cluster al variare di k
    
    print("con K = ", k)
    # Ottieni le etichette di cluster assegnate a ciascun esempio
    cluster_labels = kmeans.labels_

    # Calcola il numero di esempi in ogni cluster
    cluster_counts = np.bincount(cluster_labels)

    # Stampa il numero di esempi in ogni cluster
    for cluster, count in enumerate(cluster_counts):
        print(f"Cluster {cluster}: {count} esempi")
    

# Traccia la curva di elbow
plt.plot(range(1, 11), inertias)
plt.title('Curva di elbow')
plt.xlabel('Numero di cluster')
plt.ylabel('Inerzia')
path = "images/clustering/elbow.png" 
plt.savefig(path)

# Esegui il clustering con l'algoritmo k-means
kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)

kmeans.fit(X)

# Aggiungi i cluster al dataset
data["Cluster"] = kmeans.predict(X)


# Seleziona le feature che determinano la gradevolezza di un film 
feature1 = "votes"
feature2 = "score"

#il grafico scatter mostra eventuali aree della città dove si verificano incidenti stradali con caratteristiche simili, 
# come ad esempio un alto numero di persone ferite o uccise.

# Crea il grafico scatter colorato in base all'etichetta di cluster
#cmap = matplotlib.cm.get_cmap('viridis', len(data['Cluster'].unique()))
num_cluster_unique = len(data['Cluster'].unique())
cmap = cm.get_cmap('viridis', num_cluster_unique)
plt.title('Clustering')
plt.scatter(data[feature1], data[feature2], c=data['Cluster'], cmap=cmap)
plt.xlabel(feature1)
plt.ylabel(feature2)
# Imposta gli intervalli sull'asse x e sull'asse y
plt.xlim(min(data[feature1]), max(data[feature1]) )
plt.ylim(min(data[feature2]),max(data[feature2]) )


# Aggiungi la legenda dei cluster
cluster_labels = sorted(data['Cluster'].unique())
for label in cluster_labels:
    plt.scatter([], [], color=cmap(label), alpha=0.5, label='Cluster {}'.format(label))
plt.legend(title="Cluster", loc="upper right", markerscale=1, fontsize=10)

path = "images/clustering/clustering.png" 
plt.savefig(path)

# Calcola l'indice di silhouette (valutazione) per il clustering
silhouette_avg = silhouette_score(X, kmeans.labels_)

# Stampa l'indice di silhouette medio
print("\n Indice di silhouette medio:", silhouette_avg)
