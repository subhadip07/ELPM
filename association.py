import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
# import networkx as nx
# import matplotlib.pyplot as plt
# import seaborn as sns  # For heatmap

# def draw_heatmap(rules, min_lift_threshold=1.0):
#   # Add min_lift_threshold parameter
#     """Draws a heatmap of association rules, filtering by minimum lift."""

#     filtered_rules = rules[rules['lift'] >= min_lift_threshold] #Filter the rules

#     if filtered_rules.empty: # Check if there are any rules after filtering
#         st.info(f"No association rules found with a lift greater than or equal to {min_lift_threshold}.")
#         return # If not, return

#     antecedents = filtered_rules['antecedents'].apply(lambda x: ", ".join(list(x)))
#     consequents = filtered_rules['consequents'].apply(lambda x: ", ".join(list(x)))

#     association_matrix = pd.pivot_table(filtered_rules, values='lift', index=antecedents, columns=consequents, aggfunc='max', fill_value=0)

#     plt.figure(figsize=(10, 8))
#     sns.heatmap(association_matrix, annot=True, cmap="YlGnBu", fmt=".2f")
#     plt.title(f"Association Rule Heatmap (Lift >= {min_lift_threshold})")  # Update title
#     st.pyplot(plt.gcf())

def display_association_rules(df, min_support=0.05, min_confidence=0.3, min_lift=0.2): #top_n_rules=20):
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

            # --- Heatmap Visualization ---
            # st.subheader("Association Rule Heatmap")

            # min_lift_heatmap = st.number_input("Minimum Lift for Heatmap", min_value=1.0, value=2.0, step=0.1) #Input for lift value
            # draw_heatmap(rules, min_lift_threshold=min_lift_heatmap) #Pass the value to function
            


            # --- Network Graph Visualization ---
            # st.subheader("Network Graph Visualization")

            # def draw_network(rules, top_n=10):
            #     rules_to_visualize = rules.sort_values('lift', ascending=False).head(top_n)
            #     graph = nx.DiGraph()

            #     for _, row in rules_to_visualize.iterrows():
            #         antecedents = ", ".join([str(x) for x in row['antecedents']])
            #         consequents = ", ".join([str(x) for x in row['consequents']])
            #         graph.add_edge(antecedents, consequents, weight=row['lift'])

            #     pos = nx.spring_layout(graph, k=0.3)  # Adjust k for layout spacing
            #     edge_labels = nx.get_edge_attributes(graph, 'weight')

            #     fig, ax = plt.subplots(figsize=(12, 10))  # Create figure and axes
            #     nx.draw(graph, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10,
            #             arrowsize=20, arrowstyle='-|>', edge_color="gray", edge_cmap=plt.cm.Blues,
            #             connectionstyle="arc3,rad=0.2", ax=ax)  # Pass ax to draw
            #     nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red', ax=ax)  # Pass ax here too
            #     ax.set_title(f"Top {top_n} Association Rules (Network Graph)")  # Set title for the axes
            #     st.pyplot(fig)  # Use st.pyplot to display the plot

            # draw_network(rules, top_n=top_n_rules)  # Draw the network graph

            top_10_rules = rules.nlargest(10, 'lift')  # Get top 10 rules by lift
            st.subheader("Associated Features")
            if not top_10_rules.empty: #Check if there are any top 10 rules
                for index, row in top_10_rules.iterrows(): #Iterate to top 10 rules
                    antecedents = ", ".join([str(x) for x in row['antecedents']])
                    consequents = ", ".join([str(x) for x in row['consequents']])
                    st.write(f"**If** {antecedents}  **then** {consequents}")
                    st.write(f"Support: {row['support']:.2f}, Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f}")
                    st.write("---")
            
            
            # for index, row in rules.iterrows():
            #     antecedents = ", ".join([str(x) for x in row['antecedents']])
            #     consequents = ", ".join([str(x) for x in row['consequents']])
            #     st.write(f"**If** {antecedents}  **then** {consequents}")
            #     st.write(f"Support: {row['support']:.2f}, Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f}")
            #     st.write("---")

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
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):  # Check if numeric
            df[col] = df[col].astype('category')  # Convert to categorical



    # st.subheader("Data Preprocessing")
    # min_support = st.slider("Select Minimum Support", 0.01, 1.0, 0.05, 0.01)
    # min_confidence = st.slider("Select Minimum Confidence", 0.01, 1.0, 0.3, 0.01)
    # min_lift = st.slider("Select Minimum Lift", 1.0, 10.0, 1.0, 0.1)

     # --- Set parameter values directly ---
    min_support = 0.05  
    min_confidence = 0.5  
    min_lift = 1.0       
    
    # top_n_rules = st.slider("Number of Rules to Visualize in Network Graph", 1, 50, 20, 1) #Slider for top n rules


    display_association_rules(df, min_support, min_confidence, min_lift) # Call the function to display association rules

    