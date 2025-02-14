import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull


def classify():
    st.write("# Classification & Clustering")
    
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset in the Home section.")
        return
    
    df = st.session_state.df.copy()
    
    st.write("## Data Preparation")
    st.write("### Feature Selection")
    
    features = st.multiselect("Select Feature Columns", df.columns)
    
    if not features:
        st.warning("⚠️ Please select at least one feature column.")
        return
    
    X = df[features]
    
    # Handle categorical and numerical features
    categorical_columns = X.select_dtypes(include=['object']).columns
    numeric_columns = X.select_dtypes(exclude=['object']).columns
    
    if len(numeric_columns) <2:
        st.warning("⚠️ Need at least two numeric features for clustering with PCA.")
        return
    
    # Process and transform the data
    st.write("## Data Preprocessing")
    
    preprocess_option = st.radio(
        "Preprocessing method:",
        ["Standard Scaling", "Min-Max Scaling", "No Scaling"]
    )
    
    transformers = []
    if len(numeric_columns) > 0:
        if preprocess_option == "Standard Scaling":
            numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        elif preprocess_option == "Min-Max Scaling":
            from sklearn.preprocessing import MinMaxScaler
            numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
        else:
            numeric_transformer = Pipeline(steps=[('passthrough', 'passthrough')])
        transformers.append(('num', numeric_transformer, numeric_columns))
    
    if len(categorical_columns) > 0:
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        transformers.append(('cat', categorical_transformer, categorical_columns))
    
    # Apply the preprocessing
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    preprocessed_data = preprocessor.fit_transform(X)
    
    # Convert to dense if sparse
    if hasattr(preprocessed_data, "toarray"):
        preprocessed_data = preprocessed_data.toarray()
    
    # PCA section
    st.write("## Dimensionality Reduction with PCA")
    n_components = st.slider("Select number of PCA components", 2, min(10, preprocessed_data.shape[1]), 2)
    
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(preprocessed_data)
    
    # Show explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()
    
    st.write(f"Total variance explained by {n_components} components: {cumulative_variance[-1]:.2%}")
    
    # Create DataFrame for PCA visualization
    pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(n_components)])
    
    # K-means Clustering section
    st.write("## K-means Clustering")
    
    # Simple K selection (since optimal K is already calculated)
    k = st.number_input("Select number of clusters (K)", min_value=2, value=5)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(pca_data)
    
    # Add cluster labels to PCA dataframe
    pca_df['Cluster'] = clusters
    
    # Visualization of clusters
    st.write("## Cluster Visualization")
    
    vis_dims = st.selectbox(
        "Choose dimensions for visualization",
        options=["PC1 vs PC2", "PC1 vs PC3", "PC2 vs PC3"] if n_components >= 3 else ["PC1 vs PC2"],
        index=0
    )
    
    # Extract the dimensions for plotting
    if vis_dims == "PC1 vs PC2":
        x_dim, y_dim = 'PC1', 'PC2'
    elif vis_dims == "PC1 vs PC3":
        x_dim, y_dim = 'PC1', 'PC3'
    else:
        x_dim, y_dim = 'PC2', 'PC3'
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each cluster with convex hull
    for i in range(k):
        cluster_data = pca_df[pca_df['Cluster'] == i]
        ax.scatter(
            cluster_data[x_dim], 
            cluster_data[y_dim], 
            label=f'Cluster {i}',
            alpha=0.7
        )
        
        # Calculate and plot the convex hull for each cluster
        if len(cluster_data) > 2:  # ConvexHull needs at least 3 points
            points = cluster_data[[x_dim, y_dim]].values
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                ax.plot(points[simplex, 0], points[simplex, 1], 'k-', linewidth=1)
    
    # Plot centroids
    centroids_2d = kmeans.cluster_centers_[:, [pca_df.columns.get_loc(x_dim), pca_df.columns.get_loc(y_dim)]]
    ax.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        marker='x',
        s=200,
        c='black',
        label='Centroids'
    )
    
    ax.set_title(f'K-means Clustering with PCA ({vis_dims})')
    ax.set_xlabel(x_dim)
    ax.set_ylabel(y_dim)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
    # Statistics about clusters
    st.write("## Cluster Statistics")
    
    for i in range(k):
        cluster_data = pca_df[pca_df['Cluster'] == i]
        st.write(f"### Cluster {i}")
        st.write(f"Number of samples: {len(cluster_data)}")
        
        if len(cluster_data) > 0:
            st.write("Cluster center (in PCA space):")
            center = kmeans.cluster_centers_[i]
            center_df = pd.DataFrame({
                'Component': [f'PC{j+1}' for j in range(n_components)],
                'Value': center
            })
            st.dataframe(center_df)
    
    # Add download button for results
    st.write("## Download Results")
    
    # Prepare the full results dataframe
    results_df = df.copy()
    results_df['Cluster'] = clusters
    
    # Add PCA components
    for i in range(n_components):
        results_df[f'PC{i+1}'] = pca_data[:, i]
