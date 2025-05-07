import sqlite3
import pandas as pd
import re
import time
import joblib
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

hypernyms_count = joblib.load("hypernyms_count.joblib")
range_and_domain_classes_for_general_predicates = joblib.load("range_and_domain_classes_for_general_predicates2.joblib")
semantic_and_functional_relations_general_predicates = joblib.load("semantic_and_functional_relations_general_predicates2.joblib")

# Function to load Numberbatch embeddings and filter for English entries
def load_numberbatch(file_path):
    embeddings = joblib.load(file_path)
    english_embeddings = {key: value for key, value in embeddings.items() if key.startswith('/c/en/')}
    return english_embeddings

# Load the Numberbatch embeddings
file_path = 'numberbatch.joblib'  # Update with your file path
embedding_dict = load_numberbatch(file_path)

# Example DataFrame setup (your DataFrame should already have this structure)
# hypernyms_count = pd.DataFrame({
#     'concept': ['concept1', 'concept2', ...],
#     'count': [...],
#     'count_in_Visual_Genome': [...]
# })

# Step 1: Retrieve embeddings for each concept using Numberbatch
# def get_embedding(concept):
#     # Format the concept to match Numberbatch format, e.g., '/c/en/concept'
#     return embedding_dict.get(concept, None)
#
# # Step 2: Add embeddings to your DataFrame
# hypernyms_count['embedding'] = hypernyms_count['Hypernyms'].apply(get_embedding)
# hypernyms_count = hypernyms_count.dropna(subset=['embedding'])
#
# # Filter out rows without embeddings
# hypernyms_with_embeddings = hypernyms_count.dropna(subset=['embedding'])
#
#
# scaler = MinMaxScaler()
# hypernyms_with_embeddings[['count', 'count_in_Visual_Genome']] = scaler.fit_transform(
#     hypernyms_with_embeddings[['count', 'count_in_Visual_Genome']]
# )
# # Step 3: Convert embeddings into a matrix for clustering
# embeddings_matrix = np.stack(hypernyms_with_embeddings['embedding'].values)
#
# # Step 4: Apply clustering to group similar hypernyms
# n_clusters = 20  # Set based on the desired granularity
# clustering_model = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
# hypernyms_with_embeddings['cluster'] = clustering_model.fit_predict(embeddings_matrix)
#
# # Step 5: Frequency filtering and relevance scoring within clusters
# # Group by clusters and select the most frequent hypernyms
# def select_representative_hypernym(group):
#     # Score based on count and count_in_Visual_Genome
#     group['relevance_score'] = group['count_in_Visual_Genome']
#     return group.sort_values(by='relevance_score', ascending=False).iloc[0]
#
# # Apply the selection function per cluster
# representative_hypernyms = hypernyms_with_embeddings.groupby('cluster').apply(select_representative_hypernym)
#
# # Step 6: Finalize and output
# # Drop intermediate columns if needed and retain only relevant columns
# representative_hypernyms = representative_hypernyms[['Hypernyms', 'relevance_score']].reset_index(drop=True)
#
# # Display or save the final set of representative hypernyms
# print(representative_hypernyms)

# # Step 1: Initialize a list to store the results
# top_hypernyms_results = []
# # Step 2: Iterate through the DataFrame
# for index, row in range_and_domain_classes_for_general_predicates.iterrows():
#     # Get the list of hypernyms from the current row
#     hypernym_list = row['positive_domain_with_hypernyms']
#     # Step 3: Filter hypernyms_count for the relevant hypernyms
#     relevant_hypernyms = hypernyms_count[hypernyms_count['Hypernyms'].isin(hypernym_list)]
#
#     # Skip if no relevant hypernyms found
#     if relevant_hypernyms.empty:
#         continue
#
#     # Step 4: Normalize count_in_Visual_Genome for clustering
#     scaler = MinMaxScaler()
#     relevant_hypernyms[['count_in_Visual_Genome']] = scaler.fit_transform(
#         relevant_hypernyms[['count_in_Visual_Genome']]
#     )
#
#     # Step 5: Prepare embeddings for clustering
#     embeddings_matrix = np.stack(relevant_hypernyms['embedding'].values)
#
#     # Step 6: Apply clustering to group similar hypernyms
#     n_clusters = min(15, len(relevant_hypernyms))  # Limit clusters to available hypernyms
#     clustering_model = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
#     relevant_hypernyms['cluster'] = clustering_model.fit_predict(embeddings_matrix)
#
#
#     # Step 7: Select top hypernym from each cluster based on count_in_Visual_Genome
#     def select_top_hypernym(group):
#         return group.sort_values(by='count_in_Visual_Genome', ascending=False).iloc[0]


    # top_hypernyms = relevant_hypernyms.groupby('cluster').apply(select_top_hypernym)

    # # Step 8: Store the results
    # top_hypernyms_results.append({
    #     'concept': row['concept'],
    #     'positive_domain': row['positive_domain_with_hypernyms'],
    #     'top_hypernyms': top_hypernyms['Hypernyms']
    # })

# # Convert the results to a DataFrame
# top_hypernyms_df = pd.DataFrame(top_hypernyms_results)
#
# # Display or save the final results
# print(top_hypernyms_df)




# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.metrics.pairwise import cosine_similarity

# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.metrics.pairwise import cosine_similarity

# import pandas as pd
# from collections import Counter

# def cluster_hypernyms_by_row(df, embedding_dict):
#     representative_hypernyms = []

#     for index, row in df.iterrows():
#         hypernyms_array = row['positive_domain_with_hypernyms'].split(",")  # Assuming hypernyms are comma-separated
#         hypernyms_array = [hypernym.strip() for hypernym in hypernyms_array]  # Clean whitespace

#         if len(hypernyms_array) > 5:
#             # Count occurrences of each hypernym
#             hypernym_counts = Counter(hypernyms_array)

#             # Select the most common hypernyms, up to 5
#             most_common_hypernyms = hypernym_counts.most_common(5)
#             # Create a list of hypernyms, accounting for their frequency
#             top_hypernyms = []
#             for hypernym, count in most_common_hypernyms:
#                 top_hypernyms.extend([hypernym] * count)  # Add hypernym based on its count

#             # Ensure that only up to 5 unique hypernyms are returned
#             representative_hypernyms.append(list(dict.fromkeys(top_hypernyms))[:5])  # Remove duplicates while preserving order
#         else:
#             representative_hypernyms.append(hypernyms_array)  # Append available hypernyms

#     # Ensure the list is of appropriate length to be added to the DataFrame
#     if len(representative_hypernyms) == len(df):
#         df['representative_hypernyms'] = representative_hypernyms
#     else:
#         raise ValueError(f"Length mismatch: {len(representative_hypernyms)} (hypernyms) vs {len(df)} (DataFrame)")

#     return df






# import pandas as pd
# from collections import Counter
# from sklearn.cluster import KMeans
# import numpy as np

# def calculate_hypernym_frequencies(df):
#     # Flatten the list of hypernyms and count their frequencies
#     all_hypernyms = [hypernym for sublist in df['positive_domain_with_hypernyms'] for hypernym in sublist]
#     return Counter(all_hypernyms)

# def cluster_hypernyms_by_row(df, embedding_dict):
#     # Calculate the frequency of hypernyms across the DataFrame
#     hypernym_counts = calculate_hypernym_frequencies(df)

#     representative_hypernyms = []

#     for index, row in df.iterrows():
#         # Get the hypernyms for the current row
#         hypernyms_array = row['positive_domain_with_hypernyms']
        
#         # Get frequencies of the hypernyms in the current row
#         row_frequencies = [hypernym_counts[hypernym] for hypernym in hypernyms_array]
#         min_freq = min(row_frequencies) if row_frequencies else 0
#         max_freq = max(row_frequencies) if row_frequencies else 0
        
#         # Define thresholds based on the row's hypernym frequencies
#         frequency_thresholds = (min_freq, max_freq)

#         # Filter hypernyms based on the frequency thresholds
#         filtered_hypernyms = [hypernym for hypernym in hypernyms_array 
#                               if frequency_thresholds[0] <= hypernym_counts[hypernym] <= frequency_thresholds[1]]

#         # Create a list of embeddings for the filtered hypernyms
#         embeddings = []
#         for hypernym in filtered_hypernyms:
#             if hypernym in embedding_dict:
#                 embeddings.append(embedding_dict[hypernym])

#         if len(embeddings) > 0:
#             # Convert to a NumPy array for clustering
#             embeddings_array = np.array(embeddings)

#             # Apply K-Means clustering
#             n_clusters = min(5, len(embeddings))  # Limit the number of clusters to the number of unique hypernyms
#             kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#             labels = kmeans.fit_predict(embeddings_array)

#             # Group hypernyms by their cluster labels
#             cluster_hypernyms = {}
#             for label, hypernym in zip(labels, filtered_hypernyms):
#                 if label not in cluster_hypernyms:
#                     cluster_hypernyms[label] = []
#                 cluster_hypernyms[label].append(hypernym)

#             # Select one representative hypernym from each cluster
#             top_hypernyms = [hypernym_list[0] for hypernym_list in cluster_hypernyms.values()][:5]

#             # Append the top hypernyms to the representative list
#             representative_hypernyms.append(top_hypernyms)
#         else:
#             representative_hypernyms.append([])  # No embeddings available

#     # Ensure the list is of appropriate length to be added to the DataFrame
#     if len(representative_hypernyms) == len(df):
#         df['representative_hypernyms'] = representative_hypernyms
#     else:
#         raise ValueError(f"Length mismatch: {len(representative_hypernyms)} (hypernyms) vs {len(df)} (DataFrame)")

#     return df

# df = cluster_hypernyms_by_row(range_and_domain_classes_for_general_predicates, embedding_dict)





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


# Step 1: Sort and transform log counts
sorted_counts = hypernyms_count['count'].sort_values(ascending=False)
log_counts = np.log(sorted_counts)

# Step 2: Find the elbow point using Kneedle
kneedle = KneeLocator(range(len(log_counts)), log_counts, curve='convex', direction='decreasing')
elbow_index = kneedle.knee

# Step 3: Calculate thresholds
# Setting the lower threshold based on elbow point
lower_threshold_count = sorted_counts.iloc[elbow_index]

# Setting the upper threshold (e.g., 75th percentile)
upper_threshold_count = np.percentile(hypernyms_count['count'], 90)

# Step 4: Filter the hypernyms
filtered_hypernyms = hypernyms_count[
    (hypernyms_count['count'] >= lower_threshold_count) & 
    (hypernyms_count['count'] <= upper_threshold_count)
]

# Step 5: Output the result
print("Lower Threshold (Elbow Point):", lower_threshold_count)
print("Upper Threshold (75th Percentile):", upper_threshold_count)
print("Filtered Hypernyms:")
print(filtered_hypernyms)

# Optional: Plot the results for visualization
plt.figure(figsize=(10, 6))
plt.plot(log_counts, label='Log Counts')
plt.axvline(elbow_index, color='red', linestyle='--', label='Elbow Point')
plt.axhline(np.log(lower_threshold_count), color='green', linestyle='--', label='Lower Threshold')
plt.axhline(np.log(upper_threshold_count), color='blue', linestyle='--', label='Upper Threshold')
plt.title('Log-Transformed Hypernym Frequencies')
plt.xlabel('Index of Hypernym')
plt.ylabel('Log Frequency')
plt.legend()
plt.show()


# Step 1: Sort and transform log counts
sorted_counts = hypernyms_count['count_in_Visual_Genome'].sort_values(ascending=False)
log_counts = np.log(sorted_counts)

# Step 2: Find the elbow point using Kneedle
kneedle = KneeLocator(range(len(log_counts)), log_counts, curve='convex', direction='decreasing')
elbow_index = kneedle.knee

# Step 3: Calculate thresholds
# Setting the lower threshold based on elbow point
lower_threshold_count_in_Visual_Genome = sorted_counts.iloc[elbow_index]

# Setting the upper threshold (e.g., 75th percentile)
upper_threshold_count_in_Visual_Genome = np.percentile(hypernyms_count['count_in_Visual_Genome'], 90)

# Step 4: Filter the hypernyms
filtered_hypernyms = hypernyms_count[
    (hypernyms_count['count_in_Visual_Genome'] >= lower_threshold_count_in_Visual_Genome) & 
    (hypernyms_count['count_in_Visual_Genome'] <= upper_threshold_count_in_Visual_Genome)
]

# Step 5: Output the result
print("Lower Threshold (Elbow Point):", lower_threshold_count_in_Visual_Genome)
print("Upper Threshold (75th Percentile):", upper_threshold_count_in_Visual_Genome)
print("Filtered Hypernyms:")
print(filtered_hypernyms)

# Optional: Plot the results for visualization
plt.figure(figsize=(10, 6))
plt.plot(log_counts, label='Log Counts')
plt.axvline(elbow_index, color='red', linestyle='--', label='Elbow Point')
plt.axhline(np.log(lower_threshold_count_in_Visual_Genome), color='green', linestyle='--', label='Lower Threshold')
plt.axhline(np.log(upper_threshold_count_in_Visual_Genome), color='blue', linestyle='--', label='Upper Threshold')
plt.title('Log-Transformed Hypernym Frequencies')
plt.xlabel('Index of Hypernym')
plt.ylabel('Log Frequency')
plt.legend()
plt.show()


hypernyms_count=hypernyms_count[hypernyms_count['count']>upper_threshold_count]
hypernyms_count=hypernyms_count[hypernyms_count['count']<lower_threshold_count]
hypernyms_count=hypernyms_count[hypernyms_count['count_in_Visual_Genome']>upper_threshold_count_in_Visual_Genome]
hypernyms_count=hypernyms_count[hypernyms_count['count_in_Visual_Genome']<lower_threshold_count_in_Visual_Genome]





# Step 2: Filter embeddings for the hypernyms of interest
# hypernyms_embeddings = {k: embedding_dict[k] for k in hypernyms_count['Hypernyms'] if k in embedding_dict}

# # Step 3: Prepare data for clustering
# hypernyms = list(hypernyms_embeddings.keys())
# embeddings = np.array(list(hypernyms_embeddings.values()))


def get_embedding(concept):
    # Format the concept to match Numberbatch format, e.g., '/c/en/concept'
    return embedding_dict.get(concept, None)

# Step 2: Add embeddings to your DataFrame
hypernyms_count['embedding'] = hypernyms_count['Hypernyms'].apply(get_embedding)
hypernyms_count = hypernyms_count.dropna(subset=['embedding'])
hypernyms_count= hypernyms_count.reset_index(drop=True)
joblib.dump(hypernyms_count, "hypernyms_count2.joblib" )


# # Step 4: Normalize embeddings
# normalized_embeddings = normalize(embeddings)

# # Step 5: Clustering
# k = 100  # Set the number of clusters (adjust based on your analysis)
# kmeans = KMeans(n_clusters=k, random_state=42)
# clusters = kmeans.fit_predict(normalized_embeddings)

# # Step 6: Create a DataFrame to hold hypernyms and their cluster assignments
# hypernyms_embeddings = pd.DataFrame({
#     'Hypernyms': hypernyms,
#     'Embeddings': list(hypernyms_embeddings.values()),  # List of embeddings
#     'Cluster': clusters
# })

# # Step 7: Select representative hypernyms for each cluster
# representative_hypernyms = []
# for cluster in range(k):
#     cluster_indices = hypernyms_embeddings[hypernyms_embeddings['Cluster'] == cluster].index
#     if len(cluster_indices) == 0:
#         continue

# cluster_embeddings = normalized_embeddings[cluster_indices]
# centroid = np.mean(cluster_embeddings, axis=0)

# distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
# closest_index = cluster_indices[np.argmin(distances)]

# # Append the representative hypernym
# representative_hypernym = hypernyms_embeddings.loc[closest_index, 'Hypernyms']
# representative_hypernyms.append(representative_hypernym)

# # Step 8: Construct FOL statements
# for hypernym in representative_hypernyms:
#     statement = f"hypernym_of(x, {hypernym})"
#     print(statement)

print("Finished")