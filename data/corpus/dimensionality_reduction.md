# Dimensionality Reduction
PCA finds orthogonal directions of maximum variance; useful for compression, visualization, and noise reduction.
t-SNE and UMAP preserve local neighborhood structure for 2D/3D plots but distort global distances.
Autoencoders learn nonlinear compressed representations. Feature selection (filter/wrapper/embedded) is an alternative to projection.
Always fit reducers on training data only to avoid leakage.
