import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def display_association_rules(df, min_support=0.05, min_confidence=0.3, min_lift=0.2):
    """
    Displays association rules to the user in a readable format.

    Args:
        df (pd.DataFrame): The input DataFrame.
        min_support (float): Minimum support threshold.
        min_confidence (float): Minimum confidence threshold.
        min_lift (float): Minimum lift threshold.
    """
    try:
        for col in df.columns:
          if pd.api.types.is_numeric_dtype(df[col]):  # Check if numeric
            df[col] = df[col].astype('category')  # Convert to categorical

        df_encoded = pd.get_dummies(df)
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        rules = rules[rules['lift'] >= min_lift]

        st.subheader("Generated Association Rules")

        if not rules.empty:
            st.write(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])  # Display table

            st.subheader("Associated Features")
            for index, row in rules.iterrows():
                antecedents = ", ".join([str(x) for x in row['antecedents']])
                consequents = ", ".join([str(x) for x in row['consequents']])
                st.write(f"**If** {antecedents}  **then** {consequents}")
                st.write(f"Support: {row['support']:.2f}, Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f}")
                st.write("---")

        else:
            st.info("No association rules found with the selected parameters.")

    except Exception as e:
        st.error(f"Error processing dataset: {e}")



def association():
    st.title("Association Rule Mining")
    st.write("Association analysis is a data mining technique used to find the probability of the co-occurrence of items in a collection. It is used to identify patterns within data based on the concept of strong association. It is used to find the likelihood of relationships between products or events. The most common application of association analysis is in market basket analysis.")
    st.write("In this section, we will perform association analysis on the dataset to find the most frequent itemsets and association rules.")
    st.write("The dataset used for this analysis is the same dataset used for classification and prediction.")
    
    # Check if DataFrame is loaded
    if st.session_state.df is None:
        st.warning("Please upload a dataset in the Home section.")
        return
    
    df = st.session_state.df.copy()
    
    st.subheader("Dataset Preview")
    st.write(df.head())
    
    # Convert categorical data to binary format (One-Hot Encoding)
    st.subheader("Data Preprocessing")
    min_support = st.slider("Select Minimum Support", 0.01, 1.0, 0.05, 0.01)
    min_confidence = st.slider("Select Minimum Confidence", 0.01, 1.0, 0.3, 0.01)
    min_lift = st.slider("Select Minimum Lift", 1.0, 10.0, 1.0, 0.1)
    
    display_association_rules(df, min_support, min_confidence, min_lift) # Call the function to display association rules

    