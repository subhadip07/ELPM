import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report

   # with st.spinner("Training model and optimizing parameters..."):
    #     is_imbalanced = check_class_imbalance(y)
    #     # ... (Model Selection and Training)
    #     if is_imbalanced:
    #         steps = [
    #             ('preprocessor', preprocessor),
    #             ('smote', SMOTE(random_state=42)),
    #             ('classifier', RandomForestClassifier(random_state=42))
    #         ]
    #         pipeline = ImbPipeline(steps=steps)
    #     else:
    #         steps = [
    #             ('preprocessor', preprocessor),
    #             ('classifier', RandomForestClassifier(random_state=42))
    #         ]
    #         pipeline = Pipeline(steps=steps)
    
    # # Handle categorical target
    # le_target = LabelEncoder()
    # y = le_target.fit_transform(y)
    
    # Data splitting
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    


        # 5. Classification Model Evaluation (No y_test to evaluate against)
        # y_pred_class = rf_pipeline.predict(X_test_class)  # Cannot evaluate without y_test

        # st.write("### Random Forest Classification Results")
        # ...(Cannot display classification metrics without true labels)



# if len(features) >= 2:
            #     centers = pipeline.named_steps['kmeans'].cluster_centers_
            #     fig_centers = go.Figure()

            #     fig_centers.add_trace(go.Scatter(
            #         x=X_transformed[:, 0], y=X_transformed[:, 1],
            #         mode='markers',
            #         marker=dict(color=clusters),
            #         name='Data Points'
            #     ))

            #     fig_centers.add_trace(go.Scatter(
            #         x=centers[:, 0], y=centers[:, 1],
            #         mode='markers',
            #         marker=dict(
            #             color='red',
            #             size=15,
            #             symbol='x'
            #         ),
            #         name='Cluster Centers'
            #     ))

            #     fig_centers.update_layout(title='Cluster Centers Analysis',
            #                                 xaxis_title=features[0],
            #                                 yaxis_title=features[1])
            #     st.plotly_chart(fig_centers)
    
    
    # elif model_choice == "Random Forest":
    #     st.write("## Random Forest Classification (using KMeans Clusters)")

    #     # 1. Perform KMeans Clustering (if not already done)
    #     optimal_k = 5  # Or let the user choose
    #     kmeans_pipeline = Pipeline([
    #         ('preprocessor', preprocessor), # Use same preprocessor as before
    #         ('kmeans', KMeans(n_clusters=optimal_k, random_state=42, n_init=10, init='k-means++'))
    #     ])
    #     X_clustered = kmeans_pipeline.fit_transform(X) # Fit and transform on original features

    #     kmeans = kmeans_pipeline.named_steps['kmeans']
    #     clusters = kmeans.labels_

    #     # 2. Add cluster labels as a new feature
    #     X['Cluster'] = clusters  # Add to the original X

    #     # 3. Prepare data for classification (including the 'Cluster' feature)
    #     X_class = X  # Use all features, including the cluster assignments

    #     # Split data for classification
    #     X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    #         X_class, y, test_size=0.2, random_state=42, stratify=y
    #     )

    #     # 4. Classification Model Training (Random Forest)
    #     with st.spinner("Training Random Forest Classifier..."):
    #         is_imbalanced_class = check_class_imbalance(y) # Check class imbalance for classification target

    #         if is_imbalanced_class:
    #             rf_pipeline = ImbPipeline([
    #                 ('classifier', RandomForestClassifier(random_state=42))
    #             ])
    #         else:
    #             rf_pipeline = Pipeline([
    #                 ('classifier', RandomForestClassifier(random_state=42))
    #             ])

    #         param_grid = {  # Example grid, adjust as needed
    #             'classifier__n_estimators': [100, 200],
    #             'classifier__max_depth': [10, 20],
    #             'classifier__min_samples_split': [2, 5],
    #             'classifier__min_samples_leaf': [1, 2]
    #         }

            
    #         grid_search_class = GridSearchCV(rf_pipeline, param_grid, cv=3, n_jobs=-1)
    #         grid_search_class.fit(X_train_class, y_train_class)
    #         rf_pipeline = grid_search_class.best_estimator_
        
    #     # Make predictions
    #     y_pred = pipeline.predict(X_test)
        
    #     st.write("## Model Analysis")
        
    #     # Feature importance first
    #     rf_classifier = pipeline.named_steps['classifier']
    #     importance_df = pd.DataFrame({
    #         'feature': features,
    #         'importance': rf_classifier.feature_importances_
    #     }).sort_values('importance', ascending=False)
        
    #     st.write("### Feature Importance")
    #     fig = px.bar(importance_df, x='feature', y='importance',
    #                 title='Feature Importance Analysis')
    #     st.plotly_chart(fig)
        
    #     # Model Performance Metrics
    #     st.write("### Model Performance")
        
    #     col1, col2, col3, col4, col5 = st.columns(5)
        
    #     with col1:
    #         st.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.3f}")
    #         st.button("ℹ️", key="precision_info", help=metric_tooltip('precision'))
            
    #     with col2:
    #         st.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.3f}")
    #         st.button("ℹ️", key="recall_info", help=metric_tooltip('recall'))
            
    #     with col3:
    #         st.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.3f}")
    #         st.button("ℹ️", key="f1_info", help=metric_tooltip('f1'))

    #     with col4:
    #         st.metric("Support", f"{len(y_test)}")
    #         st.button("ℹ️", key="support_info", help=metric_tooltip('support'))

    #     with col5:
    #         st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}") # Accuracy metric
    #         st.button("ℹ️", key="accuracy_info", help=metric_tooltip('accuracy'))  # Add tooltip for accuracy

        
    #     st.write("### Detailed Classification Report")
    #     report_dict = classification_report(y_test, y_pred, output_dict=True)
    #     report_df = pd.DataFrame(report_dict).transpose()
    #     st.dataframe(report_df)
        
    #     # --- New Point CLassification ---
    #     st.write("### Classify New Data Point (using trained RF)")

    #     new_data = {}
    #     for feature in X_class.columns:  # Iterate over all features, including 'Cluster'
    #         if X_class[feature].dtype == 'object':  # Categorical feature
    #             unique_values = X_class[feature].unique()
    #             new_data[feature] = st.selectbox(f"Enter value for {feature}", unique_values)
    #         else:  # Numerical feature
    #             new_data[feature] = st.number_input(f"Enter value for {feature}")

    #     new_df = pd.DataFrame([new_data])

    #     if st.button("Predict Class"):
    #         try:
    #             prediction = rf_pipeline.predict(new_df)
    #             predicted_class = le_target.inverse_transform(prediction)[0]
    #             st.write(f"The predicted class is: {predicted_class}")

    #         except ValueError as e:  # Catch potential errors
    #             st.error(f"Error during prediction: {e}")
    #             st.write("Check if all features are entered correctly and in the expected format.")


# def metric_tooltip(metric):
#     tooltips = {
#         'precision': """
#             Precision measures how many of the predicted positive cases were actually positive.
            
#             Example: If the model predicts 100 emails as spam and 90 of them are actually spam,
#             the precision is 90%.
            
#             High precision means fewer false positives.
#         """,
#         'recall': """
#             Recall measures how many of the actual positive cases were correctly identified.
            
#             Example: If there are 100 spam emails and the model finds 80 of them,
#             the recall is 80%.
            
#             High recall means fewer false negatives.
#         """,
#         'f1': """
#             F1-Score is the harmonic mean of precision and recall.
#             It provides a single score that balances both measures.
            
#             Perfect F1 score = 1.0
#             Worst F1 score = 0.0
            
#             It's particularly useful when you need to find an optimal blend of precision and recall.
#         """,
#         'support': """
#             Support is the number of samples of each class in the test dataset.
            
#             Example: In a binary classification:
#             - Class 0: 150 samples
#             - Class 1: 100 samples
            
#             This helps you understand if your test data is balanced.
#         """
#     }
#     return tooltips.get(metric, "No explanation available")



def classify():
    st.title("Classification & Clustering")
    
    # Check if DataFrame is loaded
    if st.session_state.df is None:
        st.warning("Please upload a dataset in the Home section.")
        return
    
    df = st.session_state.df.copy()
    
    # Select features and target
    st.subheader("Feature Selection")
    target = st.selectbox("Select Target Column", df.columns)
    features = st.multiselect("Select Feature Columns", [col for col in df.columns if col != target])
    
    if not features:
        st.warning("Please select at least one feature column.")
        return
    
    X = df[features]
    y = df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Select Model
    model_choice = st.selectbox("Choose a model", ["Decision Tree", "K-Means Clustering"])
    
    if model_choice == "Decision Tree":
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.subheader("Model Performance")
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))
    
    elif model_choice == "K-Means Clustering":
        n_clusters = st.slider("Select number of clusters", 2, 10, 3)
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = model.fit_predict(X)
        df["Cluster"] = clusters
        st.subheader("Cluster Assignments")
        st.write(df[[target, "Cluster"]].head(10))
    
    # Save trained model to session state
    # st.session_state.model = model
