import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def plot_clusters(X, clusters, features):
    if len(features) >= 2:
        fig = px.scatter(
            x=X[:, 0],
            y=X[:, 1],
            color=clusters,
            labels={'x': features[0], 'y': features[1]},
            title='Cluster Visualization'
        )
    else:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        fig = px.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            color=clusters,
            labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'},
            title='Cluster Visualization (PCA)'
        )
    return fig

def check_class_imbalance(y):
    """Check if dataset is imbalanced based on class distribution"""
    class_counts = np.bincount(y)
    major_class_count = np.max(class_counts)
    minor_class_count = np.min(class_counts)
    return (major_class_count / minor_class_count) > 3


# def plot_clusters_with_hulls(X, clusters, features, optimal_k, kmeans_object):
#     n_components = 2  # Define n_components here
#     pca = PCA(n_components=n_components)

#     # Scale the data *before* applying PCA
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(X)
#     scaled_df = pd.DataFrame(scaled_data, columns=X.columns)

#     #Replacing scaled data with original data
#     df = df.drop(X.columns, axis=1)
#     df = pd.concat([scaled_df, df], axis=1) 

#     pca_data = pca.fit_transform(scaled_data)  # Fit PCA on the *scaled* data
#     pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
#     pca_df['cluster'] = clusters  # Add the cluster labels

#     plt.figure(figsize=(8, 6))

#     for i in range(optimal_k):
#         cluster_data = pca_df[pca_df['cluster'] == i]
#         plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {i}')

#         if len(cluster_data) > 2:
#             points = cluster_data[['PC1', 'PC2']].values
#             hull = ConvexHull(points)
#             for simplex in hull.simplices:
#                 plt.plot(points[simplex, 0], points[simplex, 1], 'k-', linewidth=1)

#     plt.scatter(kmeans_object.cluster_centers_[:, 0], kmeans_object.cluster_centers_[:, 1], marker='x', s=200, c='black', label='Centroids')
#     plt.title('K-means Clustering (2D PCA Visualization)')
#     plt.xlabel('Principal Component 1 (PC1)')
#     plt.ylabel('Principal Component 2 (PC2)')
#     plt.legend()
#     plt.grid(True)
#     st.pyplot(plt)

def plot_clusters_with_hulls(X_scaled_pca, clusters, optimal_k, kmeans_object):  # Corrected parameter
    plt.figure(figsize=(8, 6))

    for i in range(optimal_k):
        cluster_data = X_scaled_pca[X_scaled_pca['cluster'] == i] # Use X_scaled_pca
        plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {i}')

        if len(cluster_data) > 2:
            points = cluster_data[['PC1', 'PC2']].values
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], 'k-', linewidth=1)

    plt.scatter(kmeans_object.cluster_centers_[:, 0], kmeans_object.cluster_centers_[:, 1], marker='x', s=200, c='black', label='Centroids')
    plt.title('K-means Clustering (2D PCA Visualization)')
    plt.xlabel('Principal Component 1 (PC1)')
    plt.ylabel('Principal Component 2 (PC2)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt) 

# def perform_kmeans_clustering(df, n_numerical_cols=None, optimal_k=5, visualize=True):  # Added n_numerical_cols as argument
#     """Performs K-means clustering and returns necessary data."""

#     if n_numerical_cols is None:  # Infer if not provided
#         n_numerical_cols = len(df.select_dtypes(include=np.number).columns) # Number of numeric cols

#     numerical_cols = df.columns[:n_numerical_cols]
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(df[numerical_cols])
#     scaled_df = pd.DataFrame(scaled_data, columns=numerical_cols)

#     # Handle other columns (if any), but keep them separate for clustering
#     other_cols = df.drop(numerical_cols, axis=1)
    
#     n_components = 2
#     pca = PCA(n_components=n_components)
#     pca_data = pca.fit_transform(scaled_data) #Fit PCA on the whole dataframe
#     pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])


#     kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, init='k-means++')
#     kmeans.fit(scaled_data) #Fit kmeans on scaled data not pca data
#     labels = kmeans.labels_
#     pca_df['cluster'] = labels

#     if visualize:
#     #     # ... (visualization code - same as before)
#         plt.figure(figsize=(8, 6))

#         for i in range(optimal_k):
#             cluster_data = pca_df[pca_df['cluster'] == i]
#             plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {i}')

#             if len(cluster_data) > 2:
#                 points = cluster_data[['PC1', 'PC2']].values
#                 hull = ConvexHull(points)
#                 for simplex in hull.simplices:
#                     plt.plot(points[simplex, 0], points[simplex, 1], 'k-', linewidth=1)

#         plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, c='black', label='Centroids')
#         plt.title('K-means Clustering with PCA and Convex Hulls')
#         plt.xlabel('Principal Component 1')
#         plt.ylabel('Principal Component 2')
#         plt.legend()
#         plt.grid(True)
#         st.pyplot(plt)  # Display in Streamlit

#         plt.figure(figsize=(8, 6))  # Separate plot without hulls

#         for i in range(optimal_k):
#             cluster_data = pca_df[pca_df['cluster'] == i]
#             plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {i}')

#         plt.title('K-means Clustering (2D PCA Visualization)')
#         plt.xlabel('Principal Component 1 (PC1)')
#         plt.ylabel('Principal Component 2 (PC2)')
#         plt.legend()
#         plt.grid(True)
#         st.pyplot(plt)  # Display in Streamlit

#         return pca_df, kmeans, pca_data, other_cols  # Return other_cols




def classify():
    st.write("# Classification & Clustering")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset in the Home section.")
        return
    
    df = st.session_state.df.copy()
    
    st.write("## Data Preparation")
    st.write("### Feature Selection")
    
    # target = st.selectbox("Select Target Column", df.columns)
    features = st.multiselect("Select Feature Columns", df.columns)
    
    if not features:
        st.warning("⚠️ Please select at least one feature column.")
        return
    
    X = df[features]
    # y = df[target]
    
    # Handle categorical features (same for both KMeans and RF)
    categorical_columns = X.select_dtypes(include=['object']).columns
    numeric_columns = X.select_dtypes(exclude=['object']).columns

    transformers = []
    if numeric_columns.any():
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        transformers.append(('num', numeric_transformer, numeric_columns))

    if categorical_columns.any():
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        transformers.append(('cat', categorical_transformer, categorical_columns))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

 

    # Scale the data and create a new DataFrame for the scaled data (Approach 2)
    X_scaled = preprocessor.fit_transform(X) #Fit and transform the data
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)


    # Model Selection
    st.write("## Model Configuration")
    model_choice = st.selectbox("Choose a model", ["K-Means Clustering", "Random Forest"], index=0) # KMeans default

    optimal_k = 3
    if model_choice == "K-Means Clustering":
        st.write("## K-Means Configuration")
        # optimal_k = 3  # Fixed k=5

        with st.spinner("Performing clustering analysis..."):
            kmeans_pipeline = Pipeline([
                ('kmeans', KMeans(n_clusters=optimal_k, random_state=42, n_init=10, init='k-means++'))
            ])

            X_transformed = kmeans_pipeline.fit_transform(X_scaled_df) #Use scaled data for clustering
            kmeans = kmeans_pipeline.named_steps['kmeans']
            clusters = kmeans.labels_

            # X_scaled_pca = pd.DataFrame(X_transformed, columns=['PC1', 'PC2']) #Create pca dataframe
            # X_scaled_pca['cluster'] = clusters

            df["Cluster"] = clusters
            st.write("### Cluster Assignments")
            st.write(df[["Cluster"]].head(10))

            n_components = 2
            pca = PCA(n_components=n_components)
            X_scaled_pca_data = pca.fit_transform(X_scaled_df)  # Fit PCA on scaled data
            X_scaled_pca = pd.DataFrame(X_scaled_pca_data, columns=['PC1', 'PC2'])
            X_scaled_pca['cluster'] = clusters  # Add cluster labels AFTER PCA

            st.write("### Cluster Visualization")
            plot_clusters_with_hulls(X_scaled_pca, clusters, optimal_k, kmeans) #Corrected call

    # if model_choice == "K-Means Clustering":
    #     st.write("## K-Means Configuration")
    #     optimal_k = 3  # Fixed k=5

    #     with st.spinner("Performing clustering analysis..."):

    #         pca_df, kmeans, pca_data, other_cols = perform_kmeans_clustering(df, optimal_k=optimal_k)

    #         df["Cluster"] = pca_df['cluster'] # Use pca_df to add cluster labels
    #         st.write("### Cluster Assignments")
    #         st.write(df[["Cluster"]].head(10))

    #         n_components = 2
    #         pca = PCA(n_components=n_components)
    #         X_scaled_pca_data = pca.fit_transform(df.iloc[:, :len(df.select_dtypes(include=np.number).columns)])  # Fit PCA on scaled data
    #         X_scaled_pca = pd.DataFrame(X_scaled_pca_data, columns=['PC1', 'PC2'])
    #         X_scaled_pca['cluster'] = pca_df['cluster']  # Add cluster labels AFTER PCA

    #         st.write("### Cluster Visualization")
    #         plot_clusters_with_hulls(X_scaled_pca, pca_df['cluster'], optimal_k, kmeans)  # Pass the PCA-transformed data
            

  # Fixed k=5         
    elif model_choice == "Random Forest":  # Classification after KMeans
        st.write("## Random Forest Classification (using KMeans Clusters)")

        # 1. Perform KMeans Clustering 
        # optimal_k = 3  # Or let the user choose
        kmeans_pipeline = Pipeline([
            ('preprocessor', preprocessor),  # Use same preprocessor as before
            ('kmeans', KMeans(n_clusters=optimal_k, random_state=42, n_init=10, init='k-means++'))
        ])
        X_clustered = kmeans_pipeline.fit_transform(X)  # Fit and transform on original features

        kmeans = kmeans_pipeline.named_steps['kmeans']
        clusters = kmeans.labels_

        # 2. Add cluster labels as a new feature
        X['Cluster'] = clusters  # Add to the original X
        
        # 3. Prepare data for classification (EXCLUDING the 'Cluster' feature)
        X_class = X.drop('Cluster', axis=1)  # Remove the 'Cluster' column


        # 3. Prepare data for classification (including the 'Cluster' feature)
        # X_class = X  # Use all features, including the cluster assignments

        # Split data for classification (No y, so split X only)
        # X_train_class, X_test_class = train_test_split(X, test_size=0.2, random_state=42)
        # X_class = X.drop('Cluster', axis=1)  # Features (excluding 'Cluster')
        y_class = X['Cluster']
        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class,  # Features (excluding 'Cluster') 
                                                                                    y_class,  # Target variable ('Cluster' column)
                                                                                    test_size=0.2, random_state=42)

        # 4. Classification Model Training (Random Forest)
        with st.spinner("Training Random Forest Classifier..."):

            rf_pipeline = Pipeline([  # No need for SMOTE or class imbalance check
                ('classifier', RandomForestClassifier(random_state=42))
            ])

            param_grid = {  # Example grid, adjust as needed
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, 20],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2]
            }

            grid_search_class = GridSearchCV(rf_pipeline, param_grid, cv=3, n_jobs=-1)
            grid_search_class.fit(X_train_class, y_train_class) # Train on Cluster labels
            rf_pipeline = grid_search_class.best_estimator_

        # --- New Point Classification ---
        st.write("### Classify New Data Point (using trained RF)")

        new_data = {}
        for feature in X_class.columns:  # Iterate over all features, excluding 'Cluster'
            if X_class[feature].dtype == 'object':  # Categorical feature
                unique_values = X_class[feature].unique()
                new_data[feature] = st.selectbox(f"Enter value for {feature}", unique_values)
            else:  # Numerical feature
                new_data[feature] = st.number_input(f"Enter value for {feature}", min_value=1, max_value=5, step=1)

        new_df = pd.DataFrame([new_data])

        if st.button("Predict Cluster"): # Changed button text
            try:
                new_df_transformed = preprocessor.transform(new_df)  # Use the preprocessor
                prediction = rf_pipeline.predict(new_df_transformed)  # Predict on the transformed data
                predicted_cluster = prediction[0]
                st.write(f"The predicted cluster is: {predicted_cluster}")
                # prediction = rf_pipeline.predict(new_df)
                # predicted_cluster = prediction[0] # Get the predicted cluster
                # st.write(f"The predicted cluster is: {predicted_cluster}")

            except ValueError as e:  # Catch potential errors
                st.error(f"Error during prediction: {e}")
                st.write("Check if all features are entered correctly and in the expected format.")
    