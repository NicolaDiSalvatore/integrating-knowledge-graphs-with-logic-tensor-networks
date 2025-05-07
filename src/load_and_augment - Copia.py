import json
import os
import sqlite3
import multiprocessing as mp
from functools import lru_cache
import joblib
import ast
import numpy as np
from nltk.corpus import wordnet as wn
import requests
import pandas as pd
import re
import nltk
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
import diskcache as dc
import hashlib
import pandas as pd
from nltk.corpus import wordnet as wn
import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import sqlite3


# Directory containing your JSON files
data_dir = 'C:/Users/nicol/PycharmProjects/VisualGenomeProject'
nltk.download('wordnet')


# List of JSON files to load
joblib_files = [
    'attributes.joblib',
    # 'attribute_synsets.joblib',
    # 'objects.joblib',
    # 'objects_attributes.joblib',
    # 'object_synsets.joblib',
    # 'qa_region_mapping_full.joblib',
    # 'qa_to_region_mapping.joblib',
    # 'question_answers.joblib',
    # 'region_descriptions.joblib',
    # 'region_descriptions_full.joblib',
    # 'region_graphs.joblib',
    # 'region_graphs_full.joblib',
    # 'relationships.joblib',
    # 'relationships_full.joblib',
    # 'relationship_synsets.joblib',
    # 'scene_graphs.joblib',
    # 'scene_graphs_full.joblib',
    # 'synsets.joblib'
]


# Dictionary to store loaded data
loaded_data_dict = {}

# Load each .joblib file
for joblib_file in joblib_files:
    joblib_filepath = os.path.join(data_dir, joblib_file)

    # Load the DataFrame from the .joblib file
    data_df = joblib.load(joblib_filepath)

    # Store the loaded DataFrame in the dictionary
    var_name = joblib_file.replace('.joblib', '_data')
    loaded_data_dict[var_name] = data_df
    locals()[var_name] = data_df
    print(f"Successfully loaded {joblib_file} into {var_name}")


# # Example usage
# attributes_data = loaded_data_dict['attributes_data']
# print(attributes_data.head())
#
# attribute_synsets_data = loaded_data_dict['attribute_synsets_data']
# print(attribute_synsets_data.head())
#
# objects_data = loaded_data_dict['objects_data']
# print(objects_data.head())
#
# object_synsets_data = loaded_data_dict['object_synsets_data']
# print(object_synsets_data.head())
#
# qa_to_region_mapping_data = loaded_data_dict['qa_to_region_mapping_data']
# print(qa_to_region_mapping_data.head())
#
# question_answers_data = loaded_data_dict['question_answers_data']
# print(question_answers_data.head())
#
# region_descriptions_data = loaded_data_dict['region_descriptions_data']
# print(region_descriptions_data.head())
#
# region_graphs_data = loaded_data_dict['region_graphs_data']
# print(region_graphs_data.head())
#
# relationships_data = loaded_data_dict['relationships_data']
# print(relationships_data.head())
#
# relationship_synsets_data = loaded_data_dict['relationship_synsets_data']
# print(relationship_synsets_data.head())
#
# scene_graphs_data = loaded_data_dict['scene_graphs_data']
# print(scene_graphs_data.head())
#
# synsets_data = loaded_data_dict['synsets_data']
# print(synsets_data.head())





def vectorized_extract_keywords(synsets):
    """
    Extract keywords from the list of synsets.
    """
    return [synset.split('.')[0].lower() for synset in synsets]


def vectorized_get_keywords_from_names(names):
    """
    Extract keywords from the names.
    """
    return [name.lower() for name in names]



def update_data_with_keywords(data, sumo_mapping=None):
    """
    Update the data with a new 'keywords' column.
    """
    # Safely parse the synsets column
    def safe_parse_synsets(x):
        if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
            try:
                return ast.literal_eval(x)
            except Exception as e:
                print(f"Error parsing synsets: {e}")
                return []
        return x

    # Ensure 'synsets' is a list of strings
    data['parsed_synsets'] = data['synsets'].apply(safe_parse_synsets)


    # Convert wordnet synsets list in sumo concepts list
    def process_synsets(synsets):
        sumo_concepts = []
        if synsets:
            for synset in synsets:
                offset = convert_synset_to_offset(synset)
                if offset:
                    sumo_concepts.extend(sumo_mapping.get(offset, []))
        return sumo_concepts


    # Vectorized extraction of keywords from synsets
    if sumo_mapping is None:
        data['keywords'] = data['parsed_synsets'].apply(vectorized_extract_keywords)
    else:
        data['keywords'] = data['parsed_synsets'].apply(process_synsets)

    # Identify rows where keywords are missing
    missing_keywords_mask = data['keywords'].apply(lambda x: len(x) == 0)

    # For rows with missing keywords, use names to get the keywords
    data.loc[missing_keywords_mask, 'keywords'] = data.loc[missing_keywords_mask, 'names'].apply(
        vectorized_get_keywords_from_names)

    # Flatten keywords to strings (if needed)
    data['keywords'] = data['keywords'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

    # Drop the auxiliary column
    data.drop(columns=['parsed_synsets'], inplace=True)

    # Get unique keywords
    all_keywords = pd.Series([kw for sublist in data['keywords'].apply(lambda x: x.split(', ')) for kw in sublist]).unique().tolist()

    return data, all_keywords
# Example data structure for testing

# Update data with keywords
# objects_data, all_keywords = update_data_with_keywords(objects_data.head(100))
# print(objects_data.head())
# print(all_keywords)


# import faiss
# import numpy as np
#
# import numpy as np
# import gzip
#
#
# def load_numberbatch_embeddings(file_path):
#     """
#     Loads Numberbatch embeddings from a file into a dictionary.
#     The file can be a plain text file or a gzipped file.
#
#     Args:
#     - file_path (str): Path to the Numberbatch embeddings file (can be .txt or .gz).
#
#     Returns:
#     - embeddings (dict): A dictionary mapping concepts (str) to their embedding vectors (np.array).
#     """
#     embeddings = {}
#
#     # Open the file (supports both .txt and .txt.gz)
#     open_func = gzip.open if file_path.endswith('.gz') else open
#
#     with open_func(file_path, 'rt', encoding='utf-8') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) > 1:  # Skip empty or malformed lines
#                 concept = parts[0]  # First part is the concept name
#                 vector = np.array(parts[1:], dtype=float)  # Remaining parts are the vector
#                 embeddings[concept] = vector  # Store the concept and its vector in the dictionary
#
#     return embeddings
#
#
# import sqlite3
#
#
# def fetch_concepts_in_batches(db_path, batch_size):
#     """
#     Fetch concepts from the ConceptNet database in batches.
#
#     Args:
#     - db_path (str): Path to the ConceptNet SQLite database.
#     - batch_size (int): Number of concepts to fetch in each batch.
#
#     Yields:
#     - List[str]: A batch of concept names (strings).
#     """
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()
#
#     # Start with the first batch, and fetch subsequent batches
#     offset = 0
#     while True:
#         query = f"SELECT DISTINCT start FROM conceptnet LIMIT {batch_size} OFFSET {offset}"
#         cursor.execute(query)
#         concepts_batch = [row[0].split('/')[-1] for row in cursor.fetchall()]  # Extract the last part of the URI
#
#         if not concepts_batch:
#             break  # Stop when no more rows are returned
#
#         yield concepts_batch
#
#         offset += batch_size  # Move to the next batch
#
#     conn.close()
#
#
# # Build and return a FAISS index for efficient distance queries
# def build_faiss_index(embedding_dim, use_gpu=False):
#     index = faiss.IndexFlatIP(embedding_dim)  # IP = Inner Product for cosine similarity
#     if use_gpu:
#         res = faiss.StandardGpuResources()
#         index = faiss.index_cpu_to_gpu(res, 0, index)
#     return index
#
# # Normalize embeddings for cosine similarity (by converting to unit vectors)
# def normalize_embeddings(embeddings):
#     norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
#     return embeddings / (norm + 1e-10)  # To avoid division by zero
#
# # Embed and add to FAISS index in batches
# def embed_and_add_to_faiss(db_path, numberbatch_embeddings, index, batch_size=1000):
#     for concepts_batch in fetch_concepts_in_batches(db_path, batch_size):
#         embeddings_batch = []
#         for concept in concepts_batch:
#             if concept in numberbatch_embeddings:
#                 embeddings_batch.append(numberbatch_embeddings[concept])
#
#         if embeddings_batch:
#             embeddings_batch = np.vstack(embeddings_batch)  # Convert list to matrix
#
#             # Check if embedding dimensions match the FAISS index
#             if embeddings_batch.shape[1] != index.d:
#                 raise ValueError(f"Embedding dimension mismatch: "
#                                  f"FAISS index dimension is {index.d}, "
#                                  f"but embedding dimension is {embeddings_batch.shape[1]}.")
#
#             normalized_embeddings = normalize_embeddings(embeddings_batch)  # Normalize for cosine similarity
#
#             # Add to FAISS index (ensure it's 2D)
#             try:
#                 index.add(normalized_embeddings)
#             except AssertionError as e:
#                 print(f"Error while adding to FAISS index: {e}")
#                 print(f"Embedding shape: {normalized_embeddings.shape}")
#                 raise
#
# # Main execution
# numberbatch_file = 'numberbatch-en.txt'  # Update this path
# db_path = 'conceptnet.db'  # Update this path
#
# # Load the Numberbatch embeddings (assuming a load function is already defined)
# numberbatch_embeddings = load_numberbatch_embeddings(numberbatch_file)
#
# # Get the embedding dimension (e.g., Numberbatch typically has 300 dimensions)
# embedding_dim = len(next(iter(numberbatch_embeddings.values())))
#
# # Build a FAISS index (use cosine similarity via inner product)
# index = build_faiss_index(embedding_dim)
#
# # Embed and add all ConceptNet concepts to FAISS index
# embed_and_add_to_faiss(db_path, numberbatch_embeddings, index, batch_size=1000)
#
# # Saving the FAISS index for later use
# faiss.write_index(index, 'conceptnet_faiss.index')
#
# print("ConceptNet embeddings indexed successfully and saved to 'conceptnet_faiss.index'")


import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity

# Download WordNet data
nltk.download('wordnet')



import pickle
import numpy as np

# def convert_to_pickle(numberbatch_txt_path, pickle_save_path):
#     numberbatch_dict = {}
#     with open(numberbatch_txt_path, 'r', encoding='utf-8') as f:
#         next(f)  # Skip the first line (metadata)
#         for line in f:
#             parts = line.strip().split()
#             concept = parts[0]
#             embedding = np.array([float(x) for x in parts[1:]], dtype=np.float32)
#             numberbatch_dict[concept] = embedding
#
#     # Save the dictionary as a pickle file
#     with open(pickle_save_path, 'wb') as pf:
#         pickle.dump(numberbatch_dict, pf)
#
# # Convert and save
# convert_to_pickle('numberbatch.txt', 'numberbatch_embeddings.pkl')

# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from joblib import Parallel, delayed
# import pickle
# import nltk
# from nltk.corpus import wordnet as wn
#
#
# # Load the Numberbatch embeddings (assuming the embeddings are saved as a pickle file)
# def load_numberbatch_pickle(file_path):
#     with open(file_path, 'rb') as f:
#         return pickle.load(f)
#
#
# # Function to get the synset embedding
# def get_synset_embedding(synset, embeddings_dict):
#     lemmas = synset.lemma_names()
#     vectors = []
#
#     for lemma in lemmas:
#         lemma_key = f'/c/en/{lemma.lower()}'  # Assumes lemmas are in English and in ConceptNet format
#         if lemma_key in embeddings_dict:
#             vectors.append(embeddings_dict[lemma_key])
#
#     if len(vectors) > 0:
#         return np.mean(vectors, axis=0)  # Average vector for the synset
#     else:
#         return None  # No embedding found for synset
#
#
# # Function to compute cosine similarity for a chunk of vectors
# def compute_similarity_chunk(synset_embedding, chunk):
#     return cosine_similarity(synset_embedding.reshape(1, -1), chunk)[0]
#
#
# # Parallelized function to find the most similar concept using cosine similarity
# def find_most_similar_parallel(synset_embedding, concepts, vectors, n_jobs=-1):
#     if synset_embedding is None:
#         return None, 0.0
#
#     # Split vectors into chunks for parallel processing
#     num_chunks = 10  # You can adjust the number of chunks depending on your data size
#     chunk_size = len(vectors) // num_chunks
#
#     # Use Parallel to process chunks in parallel
#     results = Parallel(n_jobs=n_jobs)(
#         delayed(compute_similarity_chunk)(synset_embedding, vectors[i * chunk_size: (i + 1) * chunk_size])
#         for i in range(num_chunks)
#     )
#
#     # Concatenate the results back into a single array
#     similarities = np.concatenate(results)
#
#     # Find the index of the maximum similarity
#     max_idx = np.argmax(similarities)
#
#     return concepts[max_idx], similarities[max_idx]
#
#
# # Main function to find the most similar concept for a WordNet synset
# def find_similar_concept_for_synset_parallel(synset_name, concepts, vectors, embeddings_dict, n_jobs=-1):
#     synset = wn.synset(synset_name)
#     synset_embedding = get_synset_embedding(synset, embeddings_dict)
#
#     # Find the most similar concept in ConceptNet using parallel processing
#     return find_most_similar_parallel(synset_embedding, concepts, vectors, n_jobs)
#
#
# # Example usage:
#
# # Load the ConceptNet Numberbatch embeddings from a pickle file
# file_path = 'numberbatch.pkl'  # Replace with your actual pickle file path
# embeddings_dict = load_numberbatch_pickle(file_path)
#
# # Prepare the vectors and concepts from the embeddings dictionary
# concepts = list(embeddings_dict.keys())
# vectors = np.array(list(embeddings_dict.values()))  # Convert to numpy array for efficient computation
#
# # Find the most similar concept for a WordNet synset
# synset_name = 'dog.n.01'  # Example WordNet synset
# most_similar_concept, similarity = find_similar_concept_for_synset_parallel(synset_name, concepts, vectors,
#                                                                             embeddings_dict, n_jobs=4)
#
# print(
#     f"The most similar concept in ConceptNet to '{synset_name}' is '{most_similar_concept}' with similarity {similarity:.4f}")



# import pickle
# import numpy as np
# from scipy.spatial.distance import cdist
# from nltk.corpus import wordnet as wn
# import pandas as pd
#
#
# # Load the Numberbatch embeddings (assuming the embeddings are saved as a pickle file)
# def load_numberbatch_pickle(file_path):
#     with open(file_path, 'rb') as f:
#         embeddings = pickle.load(f)
#
#     # Filter for English entries
#     english_embeddings = {key: value for key, value in embeddings.items() if key.startswith('/c/en/')}
#     return english_embeddings
#
#
# # Function to get the synset embedding
# def get_synset_embedding(synsets, embeddings_dict):
#     vectors = []
#
#     for synset_name in synsets:
#         synset = wn.synset(synset_name)
#         lemmas = synset.lemma_names()
#
#         for lemma in lemmas:
#             lemma_key = f'/c/en/{lemma.lower()}'  # Assumes lemmas are in English and in ConceptNet format
#             if lemma_key in embeddings_dict:
#                 vectors.append(embeddings_dict[lemma_key])
#
#     if vectors:
#         return np.mean(vectors, axis=0)  # Average vector for the synset
#     else:
#         return None  # No embedding found for synset
#
#
# Function to find the most similar concept using batch processing
# def find_most_similar_batch(synset_embedding, vectors, batch_size=10000):
#     if synset_embedding is None:
#         return None, 0.0
#
#     # Normalize the synset_embedding
#     synset_embedding = synset_embedding / np.linalg.norm(synset_embedding)
#
#     max_similarity = -1
#     best_concept_idx = None
#
#     # Process vectors in batches to save memory
#     num_batches = len(vectors) // batch_size + 1
#
#     for i in range(num_batches):
#         start_idx = i * batch_size
#         end_idx = min((i + 1) * batch_size, len(vectors))
#
#         # Get the current batch of vectors
#         batch = vectors[start_idx:end_idx]
#
#         # Normalize the batch
#         batch_norm = batch / np.linalg.norm(batch, axis=1, keepdims=True)
#
#         # Compute cosine similarity between synset_embedding and the batch
#         similarities = np.dot(batch_norm, synset_embedding)
#
#         # Find the best match in this batch
#         batch_max_idx = np.argmax(similarities)
#         batch_max_similarity = similarities[batch_max_idx]
#
#         if batch_max_similarity > max_similarity:
#             max_similarity = batch_max_similarity
#             best_concept_idx = start_idx + batch_max_idx
#
#     return best_concept_idx, max_similarity
#
#
# # Main function to find the most similar concept for a DataFrame of WordNet synsets
# def find_similar_concepts_in_dataframe(df, embeddings_dict, vectors):
#     df['most_similar_concept'] = None
#     df['similarity'] = 0.0
#
#     for idx, row in df.iterrows():
#         synsets = row['synsets']  # List of synsets
#         synset_embedding = get_synset_embedding(synsets, embeddings_dict)
#
#         # Find the most similar concept using batch processing
#         best_concept_idx, similarity = find_most_similar_batch(synset_embedding, vectors)
#
#         if best_concept_idx is not None:
#             df.at[idx, 'most_similar_concept'] = concepts[best_concept_idx]
#             df.at[idx, 'similarity'] = similarity
#
#     return df



import pickle
import numpy as np
from scipy.spatial.distance import cosine
from nltk.corpus import wordnet as wn
import pandas as pd
import time
import joblib


# Load the Numberbatch embeddings (assuming the embeddings are saved as a joblib file)
def load_numberbatch(file_path):
    embeddings = joblib.load(file_path)

    # Filter for English entries
    english_embeddings = {key: value for key, value in embeddings.items() if key.startswith('/c/en/')}
    return english_embeddings


# Function to get the synset embedding
def get_synset_embedding(synsets, embeddings_dict):
    vectors = []
    for synset_name in synsets:
        try:
            synset = wn.synset(synset_name)
            lemmas = synset.lemma_names()
            for lemma in lemmas:
                lemma_key = f'/c/en/{lemma.lower()}'
                if lemma_key in embeddings_dict:
                    vectors.append(embeddings_dict[lemma_key])
        except:
            pass

    if vectors:
        return np.mean(vectors, axis=0)  # Return the average vector
    else:
        return None


# Function to find the most similar concept using batch processing
def find_most_similar_batch(synset_embedding, vectors, batch_size=10000):
    if synset_embedding is None:
        return None, 0.0

    # Normalize the synset_embedding
    synset_embedding = synset_embedding / np.linalg.norm(synset_embedding)

    max_similarity = -1
    best_concept_idx = None

    # Process vectors in batches to save memory
    num_batches = len(vectors) // batch_size + 1

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(vectors))

        # Get the current batch of vectors
        batch = vectors[start_idx:end_idx]

        # Normalize the batch
        batch_norm = batch / np.linalg.norm(batch, axis=1, keepdims=True)

        # Compute cosine similarity between synset_embedding and the batch
        similarities = np.dot(batch_norm, synset_embedding)

        # Find the best match in this batch
        batch_max_idx = np.argmax(similarities)
        batch_max_similarity = similarities[batch_max_idx]

        if batch_max_similarity > max_similarity:
            max_similarity = batch_max_similarity
            best_concept_idx = start_idx + batch_max_idx

    return best_concept_idx, max_similarity


# Hybrid function to find the most similar concept
def find_most_similar_hybrid(synset_embedding, vectors, embeddings_dict, concepts, synsets, batch_size=10000):
    if synset_embedding is None:
        return None, 0.0

    best_concept = None
    max_similarity = -1

    # Check similarity with lemmas and synset name
    for synset_name in synsets:
        # Check for the synset name itself
        if f'/c/en/{synset_name.lower()}' in embeddings_dict:
            similarity = 1 - cosine(synset_embedding, embeddings_dict[f'/c/en/{synset_name.lower()}'])
            if similarity > max_similarity:
                best_concept = synset_name
                max_similarity = similarity

    # Now check lemmas
    for lemma_key, lemma_vector in embeddings_dict.items():
        similarity = 1 - cosine(synset_embedding, lemma_vector)
        if similarity > 0.9:  # Threshold for high similarity
            if similarity > max_similarity:
                best_concept = lemma_key.split('/')[-1]  # Get the concept name from the key
                max_similarity = similarity

    # If no high similarity match, fallback to searching the full batch
    if max_similarity <= 0.9:
        best_concept_idx, max_similarity = find_most_similar_batch(synset_embedding, vectors, batch_size)
        if best_concept_idx is not None:
            best_concept = concepts[best_concept_idx]

    return best_concept, max_similarity


# Main function to find the most similar concept for a DataFrame of WordNet synsets
def find_similar_concepts_in_dataframe(df, embeddings_dict, vectors, concepts):
    df['most_similar_concept'] = None
    df['similarity'] = 0.0

    # Create a cache to store results for previously processed synsets
    synset_cache = {}

    for idx, row in df.iterrows():
        synsets = row['synsets']  # List of synsets
        synset_key = tuple(sorted(synsets))  # Use sorted tuple of synsets as a key

        # Check if this synset has already been processed
        if synset_key in synset_cache:
            # If it's already processed, reuse the cached result
            best_concept, similarity = synset_cache[synset_key]
        else:
            # Otherwise, compute the most similar concept
            synset_embedding = get_synset_embedding(synsets, embeddings_dict)

            # Use hybrid approach for concept search
            best_concept, similarity = find_most_similar_hybrid(synset_embedding, vectors, embeddings_dict, concepts,
                                                                synsets)

            # Cache the result for future use
            synset_cache[synset_key] = (best_concept, similarity)

        if best_concept is not None:
            df.at[idx, 'most_similar_concept'] = best_concept
            df.at[idx, 'similarity'] = similarity

    return df


# Example usage
file_path = 'numberbatch.joblib'  # Adjust path accordingly
embeddings_dict = load_numberbatch(file_path)
concepts = list(embeddings_dict.keys())
vectors = np.array(list(embeddings_dict.values()), dtype=np.float16)  # Use float16 to reduce memory

# # Assuming 'objects_data' is your DataFrame with synsets and names columns
# start_time = time.time()
# aligned_objects = find_similar_concepts_in_dataframe(objects_data.head(3), embeddings_dict, vectors, concepts)
# end_time = time.time()

# # Calculate and print elapsed time
# time_taken = end_time - start_time
# minutes, seconds = divmod(time_taken, 60)
# print(f"Time taken: {minutes} minutes and {seconds:.2f} seconds")





# aligned_relationships = find_similar_concepts_in_dataframe(relationships_data, embeddings_dict, vectors, concepts)
# aligned_attributes = find_similar_concepts_in_dataframe(attributes_data, embeddings_dict, vectors, concepts)

# Save the DataFrame to a .joblib file
# joblib.dump(aligned_objects, 'aligned_objects.joblib')
# joblib.dump(aligned_relationships, 'aligned_relationships.joblib')
# joblib.dump(aligned_attributes, 'aligned_attributes.joblib')



# Example usage:

# # Load the ConceptNet Numberbatch embeddings from a pickle file
# file_path = 'numberbatch.pkl'  # Replace with your actual pickle file path
# embeddings_dict = load_numberbatch_pickle(file_path)
#
# # Prepare the vectors and concepts from the embeddings dictionary, use float16 for reduced memory
# concepts = list(embeddings_dict.keys())
# vectors = np.array(list(embeddings_dict.values()), dtype=np.float16)  # Use float16 to reduce memory
#
# # start_time = time.time()
# # Find the most similar concept for a WordNet synset
# synset_name = 'dog.n.01'  # Example WordNet synset
# most_similar_concept, similarity = find_similar_concept_for_synset_batch(synset_name, concepts, vectors, embeddings_dict)
# # end_time = time.time()
# # time_taken = end_time - start_time
# minutes, seconds = divmod(time_taken, 60)
# print(f"The most similar concept in ConceptNet to '{synset_name}' is '{most_similar_concept}' with similarity {similarity:.4f}")
# print(f"Time taken: {int(minutes)} minutes and {seconds:.2f} seconds")





# # Initialize BERT tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
#
#
# def get_bert_embeddings(text):
#     """
#     Generate BERT embeddings for the provided text.
#     """
#     if isinstance(text, list):  # If text is a list, join into a single string
#         text = ' '.join(text)
#     tokens = tokenizer(text.lower(), return_tensors='pt')
#     with torch.no_grad():
#         outputs = model(**tokens)
#     return outputs.last_hidden_state.mean(dim=1).numpy().flatten()
#
# # Example usage for get_bert_embeddings function
# # text = "dog"
# # embeddings = get_bert_embeddings(text)
# # print(embeddings)  # This will print the BERT embeddings for the given text
#
#
# def extract_keywords_from_synset(synset, top_n=5):
#     """
#     Extract keywords from a given WordNet synset using its lemmas.
#     """
#     # Extract lemma names from the WordNet synset
#     synset = wn.synset(synset)
#     lemmas = synset.lemma_names()  # This gets lemmas (synonyms) from the synset
#
#     # Get BERT embeddings for each lemma and compare similarities
#     lemma_text = ' '.join(lemmas)  # Join all lemmas into a string
#     synset_embedding = get_bert_embeddings(lemma_text)
#
#     # Compute similarity between each lemma and the synset embedding
#     word_embeddings = {word: get_bert_embeddings(word) for word in lemmas}
#     similarities = {word: 1 - cosine(synset_embedding, emb) for word, emb in word_embeddings.items()}
#
#     # Sort lemmas by similarity and return the top_n most similar lemmas as keywords
#     sorted_keywords = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
#
#     return [kw[0] for kw in sorted_keywords]
#
# # Example usage for extract_keywords_from_synset
# # synset_name = 'dog.n.01'  # Synset ID for 'dog'
# # top_keywords = extract_keywords_from_synset(synset_name, top_n=3)
# # print(top_keywords)  # Outputs top 3 keywords for the synset 'dog'
#
# def align_data_with_conceptnet_using_BERT(data, top_n=5):
#     """
#     Extract knowledge by iterating over synsets in the synsets column.
#     Returns a DataFrame with keywords extracted from the synsets.
#     """
#     knowledge_data = []
#
#     for index, row in data.iterrows():
#         synsets_list = row['synsets']  # Get the list of synsets for this object
#         object_name = row['names']  # Assuming there's a 'names' column for the object
#
#         # Iterate over each synset in the list and extract keywords
#         for synset in synsets_list:
#             try:
#                 keywords = extract_keywords_from_synset(synset, top_n=top_n)
#                 knowledge_data.append({
#                     'object_name': object_name,
#                     'synset': synset,
#                     'keywords': keywords
#                 })
#             except Exception as e:
#                 print(f"Error processing synset '{synset}': {e}")
#
#     # Convert the knowledge data into a DataFrame
#     knowledge_df = pd.DataFrame(knowledge_data)
#
#     return knowledge_df
#
# # Example
# # aligned_df = align_data_with_conceptnet_using_BERT(objects_data.head(10), top_n=3)
# # print(knowledge_df)

def find_conceptnet_matches_sqlite(conn, keyword, max_results=100):
    """
    Finds exact matches for a keyword in the ConceptNet SQLite database using a persistent connection and executes a query.
    Only matches the keyword exactly in the start or end concept.
    """
    cursor = conn.cursor()

    # Ensure indexes exist on relevant columns to speed up queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_start ON conceptnet(start)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_end ON conceptnet(end)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_relation ON conceptnet(relation)')
    conn.commit()

    # The ConceptNet entries are in URI format like '/c/en/word', so we format the keyword accordingly
    keyword_uri = f'/c/en/{keyword.lower()}'  # Ensure lowercase for consistency

    # Query ConceptNet for exact matches
    query = """
        SELECT * FROM conceptnet
        WHERE (start = ? OR end = ?)
          AND relation IN ('/r/HasA', '/r/AtLocation', '/r/IsA', '/r/PartOf', '/r/HasProperty', '/r/CapableOf', '/r/UsedFor', '/r/RelatedTo')
        LIMIT ?
    """

    # Execute the query with the keyword URI
    df = pd.read_sql_query(query, conn, params=(keyword_uri, keyword_uri, max_results))

    return df

# # Example
# keyword = 'dog'
# db_path = 'conceptnet.db'
# conn = sqlite3.connect(db_path)
# results = find_conceptnet_matches_sqlite(conn, keyword, max_results=100)
# print(results)


def build_triples_with_conceptnet(data, db_path, max_results_per_keyword=100):
    """
    Align extracted keywords with ConceptNet by querying the ConceptNet SQLite database.
    Returns a DataFrame containing enriched knowledge.
    """
    # Open the SQLite database
    conn = sqlite3.connect(db_path)

    knowledge_data = []
    aligned_df = align_data_with_conceptnet_using_BERT(data)

    # Iterate through each row in the keywords DataFrame
    for index, row in aligned_df.iterrows():
        object_name = row['object_name']
        synset = row['synset']
        keywords = row['keywords']

        for keyword in keywords:
            # Query ConceptNet for the current keyword with an exact match
            conceptnet_matches = find_conceptnet_matches_sqlite(conn, keyword, max_results=max_results_per_keyword)

            # For each match found in ConceptNet, add it to the knowledge data
            for _, match in conceptnet_matches.iterrows():
                knowledge_data.append({
                    'object_name': object_name,
                    'synset': synset,
                    'keyword': keyword,
                    'start_concept': match['start'],
                    'end_concept': match['end'],
                    'relation': match['relation'],
                    'uri': match['uri']
                })

    # Convert the collected knowledge into a DataFrame
    knowledge_df = pd.DataFrame(knowledge_data)

    # Close the database connection
    conn.close()

    return knowledge_df

# Example usage:

# # Path to the ConceptNet SQLite database
# db_path = 'conceptnet.db'
#
# # Build triples from ConceptNet using BERT
# conceptnet_triples_df = build_triples_with_conceptnet(objects_data.head(5), db_path)
#
# # Display the enriched knowledge
# print(conceptnet_triples_df.head())


def generate_fol_statements_using_conceptnet(df):
    """
    Generates a DataFrame containing subjects, relationships, and objects
    from the flattened ConceptNet DataFrame.
    """
    # Create an empty list to store individual rows
    rows = []

    # Iterate over the rows of the input DataFrame
    for _, row in df.iterrows():
        start_concept = row['start_concept']
        end_concept = row['end_concept']
        relation = row['relation']

        # Append a dictionary to the list
        rows.append({
            'subjects': start_concept,
            'relationships': relation,
            'objects': end_concept
        })

    # Convert the list of dictionaries to a DataFrame
    fol_df = pd.DataFrame(rows, columns=['subjects', 'relationships', 'objects'])

    return fol_df

# Example usage
# fol_statements=[]
# fol_statements = generate_fol_statements_using_conceptnet(aligned_objects)
# print(fol_statements.head())


def process_and_save_aligned_data_with_conceptnet(data, db_path, output_file):
    """
    Processes the Visual Genome dataset, aligns it with ConceptNet, and saves it in an efficient format.
    """

    # Step 2: Align data with ConceptNet
    data = align_data_with_conceptnet_sqlite_and_flatten(data, db_path)

    # Step 3: Save the processed data in an efficient format
    data.to_parquet(output_file, compression='gzip')  # Save as Parquet with gzip compression



# # Example usage:
# db_path = 'conceptnet.db'
# output_file = 'data_aligned_with_conceptnet.parquet'
#
# process_and_save_aligned_data_with_conceptnet(objects_data.head(10), db_path, output_file)


def extract_concept_name(concept):
    """
    Extracts the concept name from the given string by removing the prefix '/c/en/'.

    Parameters:
    - concept (str): The concept string from which to extract the name.

    Returns:
    - str: The extracted concept name.
    """
    return concept.replace('/c/en/', '')


def query_conceptnet_db(concept):
    # Connect to the local ConceptNet database
    conn = sqlite3.connect('conceptnet.db')
    cursor = conn.cursor()

    # Query to count relationships for the given concept
    query = f"""
    SELECT relationship, COUNT(*) 
    FROM conceptnet 
    WHERE start = ? 
    GROUP BY relationship
    """

    cursor.execute(query, (concept,))
    results = cursor.fetchall()

    conn.close()

    # Convert results to a DataFrame
    return pd.DataFrame(results, columns=['relationship', 'count'])


import sqlite3
import pandas as pd
from collections import Counter


def count_relationships(df):
    # Initialize a dictionary to hold total counts for each relationship
    total_counts = {}

    # Iterate over each most similar concept in the DataFrame
    for concept in df['most_similar_concept']:
        counts_df = query_conceptnet_db(concept)

        # Aggregate counts into the total_counts dictionary
        for index, row in counts_df.iterrows():
            relationship = row['relationship']
            count = row['count']

            if relationship in total_counts:
                total_counts[relationship] += count
            else:
                total_counts[relationship] = count

    # Convert the total_counts dictionary to a DataFrame
    return pd.DataFrame(list(total_counts.items()), columns=['relationship', 'total_count'])

aligned_objects = joblib.load('aligned_objects.joblib')
concept_counts = count_relationships(aligned_objects)
print("Concept Counts:", concept_counts)

# Query ConceptNet database with the counted concepts
db_path = 'conceptnet.db'  # Replace with your actual database path
results = query_conceptnet_db(db_path, concept_counts.keys())


def escape_regex_special_chars(keyword):
    """
    Escape special characters in the keyword for safe use in regex patterns.
    """
    return re.escape(keyword)

def extract_knowledge_from_dbpedia(keyword, max_results=100):
    """
    Extract knowledge from DBpedia using SPARQL queries.
    This version retrieves subject-predicate-object triples for FOL construction,
    with an exact match for escaped keyword variations.
    """
    # Escape special characters in the keyword
    keyword_lower = escape_regex_special_chars(keyword.lower())
    keyword_plural = keyword_lower + "s"
    keyword_upper_first = escape_regex_special_chars(keyword.capitalize())
    keyword_upper_first_plural = keyword_upper_first + "s"

    query = f"""
        SELECT DISTINCT ?subject ?predicate ?object WHERE {{
            ?subject rdfs:label|foaf:name ?label .
            FILTER (
                REGEX(LCASE(STR(?label)), "^{keyword_lower}$", "i") ||
                REGEX(LCASE(STR(?label)), "^{keyword_plural}$", "i") ||
                REGEX(LCASE(STR(?label)), "^{keyword_upper_first}$", "i") ||
                REGEX(LCASE(STR(?label)), "^{keyword_upper_first_plural}$", "i")
            ) .
            ?subject ?predicate ?object .
            FILTER(?predicate IN (
                rdf:type,
                rdfs:label,
                dbo:abstract,
                dbo:wikiPageWikiLink,
                dbo:parent,
                dbo:child,
                dbo:location,
                dbo:creator,
                dbo:author,
                dbo:director,
                dbo:producer,
                dbo:writer
            )) .
            # Optionally, include literals or specific types of objects
            FILTER(isIRI(?object) || isLiteral(?object))
        }}
        LIMIT {max_results}
        """

    url = "https://dbpedia.org/sparql"
    params = {
        "query": query,
        "format": "json"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json()["results"]["bindings"]
        # Flatten the results to a list of dictionaries
        return [{
            "subject": result["subject"]["value"],
            "predicate": result["predicate"]["value"],
            "object": result["object"]["value"]
        } for result in results]
    else:
        return []


def align_data_with_dbpedia_and_flatten(data, max_results_per_keyword=100):
    """
    Processes the Visual Genome dataset, aligns it with DBpedia, flattens the results for FOL, and returns the DataFrame.
    """
    # Step 1: Add keywords to the data
    data, _ = update_data_with_keywords(data)

    def flatten_dbpedia_matches(row):
        flattened_rows = []
        keywords = row['keywords']
        if not isinstance(keywords, list):
            keywords = [keywords]  # Ensure it's a list if it's not already

        for kw in keywords:
            if isinstance(kw, str) and kw.strip():
                matches = extract_knowledge_from_dbpedia(kw, max_results=max_results_per_keyword)
                for match in matches:
                    flattened_rows.append({
                        'original_index': row.name,
                        'original_name': row['names'],
                        'original_synset': row['synsets'],
                        'keyword': kw,
                        "subject": match["subject"],
                        "predicate": match["predicate"],
                        "object": match["object"]
                    })
        return flattened_rows

    # Apply the flattening function to each row
    flattened_rows = []
    for _, row in data.iterrows():
        flattened_rows.extend(flatten_dbpedia_matches(row))

    # Convert to DataFrame
    flattened_df = pd.DataFrame(flattened_rows)

    return flattened_df



# Example using objects_data
# alignment = align_data_with_dbpedia_and_flatten(objects_data.head(5))




import wptools
import requests


def extract_knowledge_from_wikidata(keyword, max_results=100):
    """
    Extract FOL-relevant knowledge from Wikidata using the wptools library.
    This function retrieves subject-predicate-object triples for FOL construction,
    filtering for predicates that are useful for FOL statements, and returns them as a list of dictionaries.
    """
    # Format the keyword
    keyword = keyword.title()
    try:
        # Fetch the page with the keyword
        page = wptools.page(keyword)
        # Attempt to retrieve Wikidata information
        page.get_wikidata()

        # If the page does not exist or there are no claims, return an empty list
        if not page.data or 'claims' not in page.data:
            print(f"No Wikidata page found for keyword: {keyword}")
            return []

        # Extract Wikidata information
        wikidata_info = page.data.get('claims', {})
        labels = page.data.get('labels', {})

        # Define FOL-relevant predicates
        fol_predicates = {
            "P31",  # instance of (rdf:type)
            "P279",  # subclass of (dbo:subClassOf)
            "P131",  # located in the administrative territorial entity (dbo:location)
            "P17",  # country (dbo:location)
            "P50",  # author (dbo:author)
            "P57",  # director (dbo:director)
            "P162",  # producer (dbo:producer)
            "P22",  # father (dbo:parent)
            "P25",  # mother (dbo:parent)
            "P40"   # child (dbo:child)
        }

        data = []
        count = 0

        for prop, values in wikidata_info.items():
            if prop in fol_predicates:
                for value in values:
                    if count >= max_results:
                        break

                    # Get labels for the subject, predicate, and object
                    subject_label = labels.get(page.data['wikibase'], page.data['title'])
                    predicate_label = labels.get(prop, prop)

                    if isinstance(value, dict) and 'value' in value:
                        object_id = value['value']
                    else:
                        object_id = value

                    object_label = labels.get(object_id, object_id)

                    # Append to the data list
                    data.append({
                        "subject": subject_label,
                        "predicate": predicate_label,
                        "object": object_label
                    })
                    count += 1

        return data

    except (requests.exceptions.RequestException, KeyError) as e:
        # Handle specific exceptions related to requests and missing keys
        print(f"An error occurred: {e}")
        return []
    except Exception as e:
        # Handle any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        return []

def handle_wikidata_errors(error_message):
    """
    Handle errors that occur during Wikidata queries.
    This function logs the error and raises a LookupError with the provided error message.
    """
    print(f"An error occurred: {error_message}")
    raise LookupError(error_message)

# Example usage:
# keyword = "tree"
# fol_statements = extract_knowledge_from_wikidata(keyword)
#
# # Print the list of dictionaries
# for statement in fol_statements:
#     print(statement)


def align_data_with_wikidata_and_flatten(data, max_results_per_keyword=100):
    """
    Processes the Visual Genome dataset, aligns it with Wikidata, flattens the results for FOL, and returns the DataFrame.
    """
    # Step 1: Add keywords to the data
    data, _ = update_data_with_keywords(data)

    def flatten_wikidata_matches(row):
        flattened_rows = []
        keywords = row['keywords']
        if not isinstance(keywords, list):
            keywords = [keywords]  # Ensure it's a list if it's not already

        for kw in keywords:
            if isinstance(kw, str) and kw.strip():
                matches = extract_knowledge_from_wikidata(kw, max_results=max_results_per_keyword)
                for match in matches:
                    flattened_rows.append({
                        'original_index': row.name,
                        'original_name': row['names'],
                        'original_synset': row['synsets'],
                        'keyword': kw,
                        "subject": match["subject"],
                        "predicate": match["predicate"],
                        "object": match["object"]
                    })
        return flattened_rows

    # Apply the flattening function to each row
    flattened_rows = []
    for _, row in data.iterrows():
        flattened_rows.extend(flatten_wikidata_matches(row))

    # Convert to DataFrame
    flattened_df = pd.DataFrame(flattened_rows)

    return flattened_df


# Example using objects_data
# aligned_wikidata = align_data_with_wikidata_and_flatten(objects_data.head(5))
# print(aligned_wikidata)






# Function to extract knowledge from YAGO using the SPARQL endpoint
# def extract_knowledge_from_yago(keyword):
#     """
#     Queries the YAGO SPARQL endpoint for entities related to the provided keyword.
#
#     Args:
#     keyword (str): The keyword to search for in YAGO.
#
#     Returns:
#     list of dict: A list of dictionaries containing the YAGO resource and label.
#     """
#     # Define the SPARQL endpoint
#     sparql = SPARQLWrapper("https://yago-knowledge.org/sparql/query")
#
#     # Write the SPARQL query, adjusting for different possible label formats
#     keyword_lower = escape_regex_special_chars(keyword.lower())
#     keyword_plural = keyword_lower + "s"
#     keyword_upper_first = escape_regex_special_chars(keyword.capitalize())
#     keyword_upper_first_plural = keyword_upper_first + "s"
#
#     query = f"""
#             SELECT DISTINCT ?subject ?predicate ?object WHERE {{
#                 ?subject rdfs:label|foaf:name ?label .
#                 FILTER (
#                     REGEX(LCASE(STR(?label)), "^{keyword_lower}$", "i") ||
#                     REGEX(LCASE(STR(?label)), "^{keyword_plural}$", "i") ||
#                     REGEX(LCASE(STR(?label)), "^{keyword_upper_first}$", "i") ||
#                     REGEX(LCASE(STR(?label)), "^{keyword_upper_first_plural}$", "i")
#                 ) .
#                 ?subject ?predicate ?object .
#                 FILTER(?predicate IN (
#                     rdf:type,
#                     rdfs:label,
#                     dbo:abstract,
#                     dbo:wikiPageWikiLink,
#                     dbo:parent,
#                     dbo:child,
#                     dbo:location,
#                     dbo:creator,
#                     dbo:author,
#                     dbo:director,
#                     dbo:producer,
#                     dbo:writer
#                 )) .
#                 # Optionally, include literals or specific types of objects
#                 FILTER(isIRI(?object) || isLiteral(?object))
#             }}
#             LIMIT {max_results}
#             """
#
#     # Set the query and the return format
#     sparql.setQuery(query)
#     sparql.setReturnFormat(JSON)
#
#     # Execute the query and get the results
#     results = sparql.query().convert()
#
#     # Extract the results and format them as a list of dictionaries
#     extracted_data = []
#     for result in results["results"]["bindings"]:
#         entity = result["entity"]["value"]
#         label = result["label"]["value"]
#         extracted_data.append({
#             "resource": entity,
#             "label": label
#         })
#
#     return extracted_data



def extract_knowledge_from_yago(keyword, max_results=100):
    """
    Extract FOL-relevant knowledge from YAGO.
    This function retrieves subject-predicate-object triples from YAGO for FOL construction,
    filtering for predicates that are useful for FOL statements, and returns them as a list of dictionaries.
    """
    keyword = keyword.title()  # Capitalize the keyword
    endpoint = "https://yago-knowledge.org/sparql/query"  # Replace with the correct YAGO SPARQL endpoint

    query = f"""
    SELECT DISTINCT ?subject ?predicate ?object WHERE {{
      ?subject ?predicate ?object.
      ?subject rdfs:label "{keyword}"@en.
    }} LIMIT {max_results}
    """

    try:
        # Send the query to the YAGO SPARQL endpoint
        response = requests.get(endpoint, params={'query': query, 'format': 'json'})
        response.raise_for_status()  # Check for request errors
        results = response.json()

        # If there are no results, return an empty list
        if not results.get('results', {}).get('bindings'):
            print(f"No YAGO results found for keyword: {keyword}")
            return []

        # Extract data
        data = []
        for result in results['results']['bindings']:
            data.append({
                "subject": result['subject']['value'],
                "predicate": result['predicate']['value'],
                "object": result['object']['value']
            })

        return data

    except requests.exceptions.RequestException as e:
        print(f"An error occurred with the request: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

# Example Usage:
# results = extract_knowledge_from_yago("Albert Einstein", max_results=10)
# print(results)


# def extract_knowledge_from_yago(keyword, max_results=100):
#     """
#     Extract knowledge from YAGO using SPARQL queries optimized for FOL construction.
#     This version retrieves subject-predicate-object triples with a broader match for escaped keyword variations.
#     """
#     # Escape special characters in the keyword
#     keyword_lower = escape_regex_special_chars(keyword.lower())
#     keyword_upper_first = escape_regex_special_chars(keyword.capitalize())
#
#     query = f"""
#             SELECT DISTINCT ?subject ?predicate ?object WHERE {{
#                 ?subject rdfs:label|foaf:name ?label .
#                 FILTER (
#                     LCASE(STR(?label)) = "{keyword_lower}" ||
#                     STR(?label) = "{keyword_upper_first}"
#                 ) .
#                 ?subject ?predicate ?object .
#                 FILTER(?predicate IN (
#                     rdf:type,
#                     rdfs:label,
#                     yago:isLocatedIn,
#                     yago:hasFamilyName,
#                     yago:hasGivenName,
#                     yago:wasBornIn,
#                     yago:diedIn,
#                     yago:created,
#                     yago:directed,
#                     yago:produced,
#                     yago:wrote,
#                     yago:hasGender,
#                     yago:hasAcademicDegree,
#                     yago:hasProfession,
#                     yago:hasCitizenship,
#                     yago:isAffiliatedTo,
#                     yago:isMarriedTo,
#                     yago:isConnectedTo
#                 )) .
#                 FILTER(isIRI(?object) || isLiteral(?object))
#             }}
#             LIMIT {max_results}
#         """
#
#     url = "https://yago-knowledge.org/sparql/query"
#     params = {
#         "query": query,
#         "format": "json"
#     }
#     response = requests.get(url, params=params)
#     if response.status_code == 200:
#         results = response.json().get("results", {}).get("bindings", [])
#         return [{
#             "subject": result["subject"]["value"],
#             "predicate": result["predicate"]["value"],
#             "object": result["object"]["value"]
#         } for result in results]
#     else:
#         print(f"Error: HTTP {response.status_code}")
#         return []
#
# from SPARQLWrapper import SPARQLWrapper, JSON
#
# knowledge= extract_knowledge_from_yago("Albert Einstein")

# def extract_knowledge_with_domain_range(keyword, max_results=100, endpoint_url="http://localhost:3030/yago/sparql"):
#     """
#     Extract knowledge from a local YAGO repository using SPARQL queries, including domain and range of predicates.
#     This version retrieves subject-predicate-object triples and adds domain and range information for each predicate.
#
#     Parameters:
#     - keyword (str): The keyword to search for.
#     - max_results (int): The maximum number of results to return.
#     - endpoint_url (str): The URL of the local SPARQL endpoint.
#
#     Returns:
#     - List[Dict[str, str]]: A list of dictionaries containing the subject, predicate, object, domain, and range.
#     """
#     # Escape special characters in the keyword
#     keyword_lower = escape_regex_special_chars(keyword.lower())
#     keyword_upper_first = escape_regex_special_chars(keyword.capitalize())
#
#     query = f"""
#         SELECT DISTINCT ?subject ?predicate ?object ?domain ?range WHERE {{
#             ?subject rdfs:label|foaf:name ?label .
#             FILTER (
#                 LCASE(STR(?label)) = "{keyword_lower}" ||
#                 STR(?label) = "{keyword_upper_first}"
#             ) .
#             ?subject ?predicate ?object .
#             OPTIONAL {{ ?predicate rdfs:domain ?domain . }}
#             OPTIONAL {{ ?predicate rdfs:range ?range . }}
#             FILTER(?predicate IN (
#                 rdf:type,
#                 rdfs:label,
#                 yago:isLocatedIn,
#                 yago:hasFamilyName,
#                 yago:hasGivenName,
#                 yago:wasBornIn,
#                 yago:diedIn,
#                 yago:created,
#                 yago:directed,
#                 yago:produced,
#                 yago:wrote,
#                 yago:hasGender,
#                 yago:hasAcademicDegree,
#                 yago:hasProfession,
#                 yago:hasCitizenship,
#                 yago:isAffiliatedTo,
#                 yago:isMarriedTo,
#                 yago:isConnectedTo
#             )) .
#             FILTER(isIRI(?object) || isLiteral(?object))
#         }}
#         LIMIT {max_results}
#     """
#
#     sparql = SPARQLWrapper(endpoint_url)
#     sparql.setQuery(query)
#     sparql.setReturnFormat(JSON)
#
#     try:
#         response = sparql.query().convert()
#         results = response["results"]["bindings"]
#         return [{
#             "subject": result["subject"]["value"],
#             "predicate": result["predicate"]["value"],
#             "object": result["object"]["value"],
#             "domain": result.get("domain", {}).get("value", None),
#             "range": result.get("range", {}).get("value", None)
#         } for result in results]
#     except Exception as e:
#         print(f"Error querying the SPARQL endpoint: {e}")
#         return []
import requests
from SPARQLWrapper import SPARQLWrapper, JSON

def escape_regex_special_chars(keyword):
    """
    Escape special characters in the keyword for safe SPARQL query usage.
    """
    return re.escape(keyword)

# def extract_knowledge_from_yago(keyword, max_results=100, blazegraph_url="http://localhost:9999/blazegraph/namespace/YAGO/sparql"):
#     """
#     Extract knowledge from Blazegraph using SPARQL queries, optimized for FOL construction.
#     This version retrieves subject-predicate-object triples and includes domain and range
#     information for constructing logical formulas.
#
#     Parameters:
#     - keyword (str): The keyword to search for.
#     - max_results (int): The maximum number of results to return.
#     - blazegraph_url (str): The SPARQL endpoint URL for Blazegraph.
#
#     Returns:
#     - List[Dict[str, str]]: A list of dictionaries containing the subject, predicate, object,
#                             domain, and range.
#     """
#     # Escape special characters in the keyword
#     keyword_lower = escape_regex_special_chars(keyword.lower())
#     keyword_upper_first = escape_regex_special_chars(keyword.capitalize())
#
#     query = f"""
#         SELECT DISTINCT ?subject ?predicate ?object ?domain ?range WHERE {{
#             ?subject rdfs:label|foaf:name ?label .
#             FILTER (
#                 LCASE(STR(?label)) = "{keyword_lower}" ||
#                 STR(?label) = "{keyword_upper_first}"
#             ) .
#             ?subject ?predicate ?object .
#             OPTIONAL {{ ?predicate rdfs:domain ?domain . }}
#             OPTIONAL {{ ?predicate rdfs:range ?range . }}
#             FILTER(?predicate IN (
#                 rdf:type,
#                 rdfs:label,
#                 <http://yago-knowledge.org/resource/isLocatedIn>,
#                 <http://yago-knowledge.org/resource/hasFamilyName>,
#                 <http://yago-knowledge.org/resource/hasGivenName>,
#                 <http://yago-knowledge.org/resource/wasBornIn>,
#                 <http://yago-knowledge.org/resource/diedIn>,
#                 <http://yago-knowledge.org/resource/created>,
#                 <http://yago-knowledge.org/resource/directed>,
#                 <http://yago-knowledge.org/resource/produced>,
#                 <http://yago-knowledge.org/resource/wrote>,
#                 <http://yago-knowledge.org/resource/hasGender>,
#                 <http://yago-knowledge.org/resource/hasAcademicDegree>,
#                 <http://yago-knowledge.org/resource/hasProfession>,
#                 <http://yago-knowledge.org/resource/hasCitizenship>,
#                 <http://yago-knowledge.org/resource/isAffiliatedTo>,
#                 <http://yago-knowledge.org/resource/isMarriedTo>,
#                 <http://yago-knowledge.org/resource/isConnectedTo>
#             )) .
#             FILTER(isIRI(?object) || isLiteral(?object))
#         }}
#         LIMIT {max_results}
#     """
#
#     # Setup SPARQL connection
#     sparql = SPARQLWrapper(blazegraph_url)
#     sparql.setQuery(query)
#     sparql.setReturnFormat(JSON)
#
#     try:
#         results = sparql.query().convert()
#         result_list = [{
#             "subject": result["subject"]["value"],
#             "predicate": result["predicate"]["value"],
#             "object": result["object"]["value"],
#             "domain": result["domain"]["value"] if "domain" in result else None,
#             "range": result["range"]["value"] if "range" in result else None
#         } for result in results["results"]["bindings"]]
#     except Exception as e:
#         print(f"Error querying Blazegraph: {e}")
#         return []
#
#     return result_list

# Example usage
# blazegraph_data = extract_knowledge_from_yago("Berlin", max_results=50, blazegraph_url="http://localhost:9999/blazegraph/sparql")
# print(blazegraph_data)



# Function to align object names in objects_data with YAGO via SPARQL
# def align_data_with_yago(data):
#     """
#     Aligns object names in objects_data with entities in YAGO using SPARQL.
#
#     Args:
#     objects_data (DataFrame): A DataFrame containing object names.
#
#     Returns:
#     dict: A dictionary where keys are object names and values are lists of YAGO entities.
#     """
#     # Ensure the 'name' column is present in objects_data
#     if 'names' not in data.columns:
#         raise ValueError("The input DataFrame must contain a 'name' column.")
#
#     aligned_data = {}
#
#     # Iterate over each unique object name
#     for keyword in get_keywords_from_data(data):
#         # Extract knowledge from YAGO via SPARQL for the object name
#         knowledge = extract_knowledge_from_yago(keyword)
#
#         # Store the extracted data in the aligned_data dictionary
#         if knowledge:
#             aligned_data[keyword] = knowledge
#
#     return aligned_data


# #Example using objects_data
#
# # Align the object names in objects_data with YAGO entities using SPARQL
# aligned_yago_sparql = align_data_with_yago(objects_data)
#
# # Output the aligned data
# for object_name, entities in aligned_yago_sparql.items():
#     print(f"Object: {object_name}")
#     for entity in entities:
#         print(f" - Resource: {entity['resource']}, Label: {entity['label']}")

def align_data_with_yago_and_flatten(data, max_results_per_keyword=100):
    """
    Processes the data, aligns it with YAGO, flattens the results, and returns the DataFrame.
    """
    # Step 1: Add keywords to the data
    data, _ = update_data_with_keywords(data)

    def flatten_yago_matches(row):
        flattened_rows = []
        keywords = row['keywords']
        if not isinstance(keywords, list):
            keywords = [keywords]  # Ensure it's a list if it's not already

        for kw in keywords:
            if isinstance(kw, str) and kw.strip():
                matches = extract_knowledge_from_yago(kw, max_results=max_results_per_keyword)
                for match in matches:
                    flattened_rows.append({
                        'original_index': row.name,
                        'original_name': row['names'],
                        'original_synset': row['synsets'],
                        'keyword': kw,
                        "subject": match["subject"],
                        "predicate": match["predicate"],
                        "object": match["object"]
                    })
        return flattened_rows

    # Apply the flattening function to each row
    flattened_rows = []
    for _, row in data.iterrows():
        flattened_rows.extend(flatten_yago_matches(row))

    # Convert to DataFrame
    flattened_df = pd.DataFrame(flattened_rows)

    return flattened_df

# Example using objects_data
# aligned_yago_data = align_data_with_yago_and_flatten(objects_data.head(5))
# print(aligned_yago_data)
def convert_synset_to_offset(synset):
    """
    Convert a WordNet synset like 'dog.n.01' to its corresponding WordNet offset (e.g., '02084071').

    Parameters:
    synset (str): WordNet synset in the form 'lemma.pos.sense'.

    Returns:
    str: The WordNet offset (as a string).
    """
    try:
        wn_synset = wn.synset(synset)
        offset = str(wn_synset.offset()).zfill(8)
        print(f"Converted {synset} to offset {offset}")
        return offset
    except Exception as e:
        print(f"Error converting {synset}: {e}")
        return None



def parse_sumo_mapping(filepath):
    """
    Parses the SUMO mapping file and returns a dictionary mapping offsets to SUMO concepts.
    Each offset can map to multiple SUMO concepts.
    """
    sumo_mapping = {}

    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            synset_offset = parts[0].zfill(8)  # Ensure the offset is zero-padded to 8 digits
            sumo_part = parts[-1]  # The SUMO concept is the last part
            sumo_concept = sumo_part.split('&%')[-1]  # Extract SUMO concept

            # Remove the suffix ('=', '+', '@', etc.) from the SUMO concept
            sumo_concept = sumo_concept.rstrip('=+@:[].')

            if synset_offset in sumo_mapping:
                sumo_mapping[synset_offset].append(sumo_concept)
            else:
                sumo_mapping[synset_offset] = [sumo_concept]

    return sumo_mapping



from rdflib import Graph, URIRef, RDF, RDFS, OWL


# def parse_owl_file(owl_file_path):
#     """
#     Parses the OWL file using rdflib and returns a graph object.
#
#     Parameters:
#     - owl_file_path: Path to the OWL file.
#
#     Returns:
#     - A rdflib Graph containing the parsed ontology data.
#     """
#     g = Graph()
#     g.parse(owl_file_path, format="xml")
#     return g
#
#
# def find_fol_predicates(graph, keyword):
#     """
#     Finds FOL predicates in the OWL graph that match the given keyword.
#
#     Parameters:
#     - graph: rdflib Graph containing the OWL data.
#     - keyword: Keyword to search for in the OWL graph.
#
#     Returns:
#     - A list of triples matching the FOL predicates.
#     """
#     fol_triples = []
#     keyword_lower = keyword.lower()
#
#     for s, p, o in graph.triples((None, None, None)):
#         if keyword_lower in str(s).lower() or keyword_lower in str(o).lower():
#             fol_triples.append((s, p, o))
#
#     return fol_triples


# def align_data_with_fol_predicates(data, owl_file_path, sumo_mapping):
#     """
#     Aligns data with FOL predicates extracted from the SUMO OWL file.
#
#     Parameters:
#     - data: DataFrame containing the data to be aligned.
#     - owl_file_path: Path to the OWL file.
#
#     Returns:
#     - A DataFrame with the aligned FOL predicates.
#     """
#     # Step 1: Parse the OWL file
#     owl_graph = parse_owl_file(owl_file_path)
#     data, all_keywords = update_data_with_keywords(data, sumo_mapping)
#
#     def flatten_fol_matches(row):
#         flattened_rows = []
#         keywords = row['keywords']
#         if not isinstance(keywords, list):
#             keywords = [keywords]  # Ensure it's a list if it's not already
#
#         for kw in keywords:
#             if isinstance(kw, str) and kw.strip():
#                 fol_triples = find_fol_predicates(owl_graph, kw)
#                 for s, p, o in fol_triples:
#                     flattened_rows.append({
#                         'original_index': row.name,
#                         'original_name': row['names'],
#                         'original_synset': row['synsets'],
#                         'keyword': kw,
#                         'subject': s,
#                         'predicate': p,
#                         'object': o
#                     })
#         return flattened_rows
#
#     # Step 2: Apply the flattening function to each row in the DataFrame
#     flattened_rows = []
#     for _, row in data.iterrows():
#         flattened_rows.extend(flatten_fol_matches(row))
#
#     # Step 3: Convert the list of dictionaries to a DataFrame
#     flattened_df = pd.DataFrame(flattened_rows)
#
#     return flattened_df

import re


def parse_kif_file(kif_file_path, encoding='utf-8'):
    """
    Parses a SUMO KIF file and extracts the relevant data.

    Parameters:
    - kif_file_path: Path to the KIF file.
    - encoding: The encoding to use when reading the file (default is 'utf-8').

    Returns:
    - A dictionary containing the parsed KIF elements.
    """
    kif_data = {
        'instances': [],
        'domains': [],
        'subrelations': [],
        'documentations': []
    }

    with open(kif_file_path, 'r', encoding=encoding) as file:
        lines = file.readlines()

    current_documentation = None
    for line in lines:
        line = line.strip()

        # Ignore comments and empty lines
        if line.startswith(';;') or not line:
            continue

        # Match (instance <entity> <class>)
        instance_match = re.match(r'\(instance\s+(\S+)\s+(\S+)\)', line)
        if instance_match:
            entity, class_ = instance_match.groups()
            kif_data['instances'].append({'entity': entity, 'class': class_})
            continue

        # Match (domain <relation> <position> <class>)
        domain_match = re.match(r'\(domain\s+(\S+)\s+(\d+)\s+(\S+)\)', line)
        if domain_match:
            relation, position, class_ = domain_match.groups()
            kif_data['domains'].append({'relation': relation, 'position': position, 'class': class_})
            continue

        # Match (subrelation <rel1> <rel2>)
        subrelation_match = re.match(r'\(subrelation\s+(\S+)\s+(\S+)\)', line)
        if subrelation_match:
            rel1, rel2 = subrelation_match.groups()
            kif_data['subrelations'].append({'rel1': rel1, 'rel2': rel2})
            continue

        # Match (documentation <entity> <language> <text>)
        documentation_match = re.match(r'\(documentation\s+(\S+)\s+(\S+)\s+"(.+)"\)', line)
        if documentation_match:
            entity, language, text = documentation_match.groups()
            kif_data['documentations'].append({'entity': entity, 'language': language, 'text': text})
            current_documentation = None
            continue

        # Handle multiline documentation
        if current_documentation:
            if line.endswith('"'):
                current_documentation['text'] += f" {line[:-1]}"
                kif_data['documentations'].append(current_documentation)
                current_documentation = None
            else:
                current_documentation['text'] += f" {line}"
            continue

        # Detect beginning of multiline documentation
        multiline_doc_match = re.match(r'\(documentation\s+(\S+)\s+(\S+)\s+"(.+)', line)
        if multiline_doc_match:
            entity, language, text = multiline_doc_match.groups()
            current_documentation = {'entity': entity, 'language': language, 'text': text}
            continue

    return kif_data


def find_fol_predicates(kif_data, keyword):
    """
    Finds FOL predicates in the KIF data that match the given keyword.

    Parameters:
    - kif_data: Dictionary containing the parsed KIF data.
    - keyword: Keyword to search for in the KIF data.

    Returns:
    - A list of tuples matching the FOL predicates.
    """
    fol_triples = []
    keyword_lower = keyword.lower()

    # Search in instances
    for instance in kif_data['instances']:
        entity = instance['entity']
        class_ = instance['class']
        if keyword_lower in entity.lower() or keyword_lower in class_.lower():
            fol_triples.append((entity, 'instance', class_))

    # Search in domains
    for domain in kif_data['domains']:
        relation = domain['relation']
        position = domain['position']
        class_ = domain['class']
        if keyword_lower in relation.lower() or keyword_lower in class_.lower():
            fol_triples.append((relation, f'domain_{position}', class_))

    # Search in subrelations
    for subrelation in kif_data['subrelations']:
        rel1 = subrelation['rel1']
        rel2 = subrelation['rel2']
        if keyword_lower in rel1.lower() or keyword_lower in rel2.lower():
            fol_triples.append((rel1, 'subrelation', rel2))

    # Search in documentation
    for doc in kif_data['documentations']:
        entity = doc['entity']
        text = doc['text']
        if keyword_lower in entity.lower() or keyword_lower in text.lower():
            fol_triples.append((entity, 'documentation', text))

    return fol_triples
def align_data_with_fol_predicates(data, kif_file_path, sumo_mapping):
    """
    Aligns data with FOL predicates extracted from the SUMO KIF file.

    Parameters:
    - data: DataFrame containing the data to be aligned.
    - kif_file_path: Path to the KIF file.
    - sumo_mapping: Mapping dictionary for SUMO terms.

    Returns:
    - A DataFrame with the aligned FOL predicates.
    """
    # Step 1: Parse the KIF file
    kif_data = parse_kif_file(kif_file_path)  # KIF parser function
    data, all_keywords = update_data_with_keywords(data, sumo_mapping)

    def flatten_fol_matches(row):
        flattened_rows = []
        keywords = row['keywords']
        if not isinstance(keywords, list):
            keywords = [keywords]  # Ensure it's a list if it's not already

        for kw in keywords:
            if isinstance(kw, str) and kw.strip():
                fol_triples = find_fol_predicates(kif_data, kw)  # Updated to work with KIF
                for s, p, o in fol_triples:
                    flattened_rows.append({
                        'original_index': row.name,
                        'original_name': row['names'],
                        'original_synset': row['synsets'],
                        'keyword': kw,
                        'subject': s,
                        'predicate': p,
                        'object': o
                    })
        return flattened_rows

    # Step 2: Apply the flattening function to each row in the DataFrame
    flattened_rows = []
    for _, row in data.iterrows():
        flattened_rows.extend(flatten_fol_matches(row))

    # Step 3: Convert the list of dictionaries to a DataFrame
    flattened_df = pd.DataFrame(flattened_rows)

    return flattened_df


# Example usage:
# sumo_mapping = parse_sumo_mapping('WordNetMappings30-noun.txt')
# owl_file_path = 'SUMO.kif'
# result_df = align_data_with_fol_predicates(objects_data.head(10), owl_file_path, sumo_mapping)
# print(result_df)