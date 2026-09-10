# Clustering
Clustering groups unlabeled points by similarity. K-means minimizes within-cluster sum of squares; choose K via elbow or silhouette.
Hierarchical clustering builds dendrograms; DBSCAN finds density-based clusters and marks noise.
Evaluation: silhouette score, Davies–Bouldin, or external labels (ARI, NMI) when available.
Scale features before distance-based clustering. Curse of dimensionality hurts Euclidean metrics.
