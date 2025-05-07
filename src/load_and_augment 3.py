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
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
import diskcache as dc
import nltk
import hashlib
import pandas as pd
from nltk.corpus import wordnet as wn
import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import sqlite3
import pickle
import numpy as np
from scipy.spatial.distance import cosine
from nltk.corpus import wordnet as wn
import pandas as pd
import time
import joblib



# Directory containing your JSON files
data_dir = 'C:/Users/nicol/PycharmProjects/VisualGenomeProject'
nltk.download('wordnet')


# List of JSON files to load
joblib_files = [
    # 'attributes.joblib',
    # 'attribute_synsets.joblib',
    'objects.joblib',
    # 'objects_attributes.joblib',
    # 'object_synsets.joblib',
    # 'qa_region_mapping_full.joblib',
    # 'qa_to_region_mapping.joblib',
    # 'question_answers.joblib',
    # 'region_descriptions.joblib',
    # 'region_descriptions_full.joblib',
    # 'region_graphs.joblib',
    # 'region_graphs_full.joblib',
    'relationships.joblib',
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




# The following 3 function are used in case of alignment using only the synset name, without considering its lemma
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

    # For rows with missing keywords, use names to get the keywords (disabled)
    # data.loc[missing_keywords_mask, 'keywords'] = data.loc[missing_keywords_mask, 'names'].apply(
    #     vectorized_get_keywords_from_names)

    # Flatten keywords to strings (if needed)
    data['keywords'] = data['keywords'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

    # Drop the auxiliary column
    data.drop(columns=['parsed_synsets'], inplace=True)

    # Get unique keywords
    all_keywords = pd.Series([kw for sublist in data['keywords'].apply(lambda x: x.split(', ')) for kw in sublist]).unique().tolist()

    return data, all_keywords

# Example

# Update data with keywords
# objects_data, all_keywords = update_data_with_keywords(objects_data.head(100))
# print(objects_data.head())
# print(all_keywords)





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



# Example usage
# file_path = 'numberbatch.joblib'  # Adjust path accordingly
# embeddings_dict = load_numberbatch(file_path)
# concepts = list(embeddings_dict.keys())
# vectors = np.array(list(embeddings_dict.values()), dtype=np.float16)  # Use float16 to reduce memory

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
# print(1)






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


# The following part can be avoided
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

    # query = f"""
    #     SELECT DISTINCT ?subject ?predicate ?object WHERE {{
    #         ?subject rdfs:label|foaf:name ?label .
    #         FILTER (
    #             REGEX(LCASE(STR(?label)), "^{keyword_lower}$", "i") ||
    #             REGEX(LCASE(STR(?label)), "^{keyword_plural}$", "i") ||
    #             REGEX(LCASE(STR(?label)), "^{keyword_upper_first}$", "i") ||
    #             REGEX(LCASE(STR(?label)), "^{keyword_upper_first_plural}$", "i")
    #         ) .
    #         ?subject ?predicate ?object .
    #         FILTER(?predicate IN (
    #             rdf:type,
    #             rdfs:label,
    #             dbo:abstract,
    #             dbo:wikiPageWikiLink,
    #             dbo:parent,
    #             dbo:child,
    #             dbo:location,
    #             dbo:creator,
    #             dbo:author,
    #             dbo:director,
    #             dbo:producer,
    #             dbo:writer
    #         )) .
    #         # Optionally, include literals or specific types of objects
    #         FILTER(isIRI(?object) || isLiteral(?object))
    #     }}
    #     LIMIT {max_results}
    #     """


    query = f"""
    SELECT DISTINCT ?subject ?predicate ?object WHERE {{
        ?subject rdfs:label|foaf:name ?label .
        FILTER (
            LANG(?label) = "en" &&  # Ensure the label is in English
            (
                REGEX(LCASE(STR(?label)), "^{keyword_lower}$", "i") ||
                REGEX(LCASE(STR(?label)), "^{keyword_plural}$", "i") ||
                REGEX(LCASE(STR(?label)), "^{keyword_upper_first}$", "i") ||
                REGEX(LCASE(STR(?label)), "^{keyword_upper_first_plural}$", "i")
            )
        ) .
        ?subject ?predicate ?object .
        FILTER(
            (isIRI(?object) || (isLiteral(?object) && LANG(?object) = "en")) # Filter for English literals or IRIs
        )
        FILTER(?predicate IN (
            rdf:type,                     # Object types (e.g., car, tree)
            rdfs:label,                   # Labels for objects or attributes
            dbo:attribute,                # Attributes of an object (color, size, etc.)
            dbo:related,                  # General relatedness (matches predicates in Visual Genome)
            dbo:partOf,                   # Part-whole relationships
            dbo:hasPart,                  # Part-whole relationships (reverse direction)
            dbo:location,                 # Spatial predicates
            dbo:hasAttribute              # Attribute relationships for mapping VG attributes
        )) .
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
                        'original_name': row['predicate'],
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
alignment = align_data_with_dbpedia_and_flatten(relationships_data.head(100))




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


# def find_fol_predicates(kif_data, keyword):
#     """
#     Finds FOL predicates in the KIF data that match the given keyword.

#     Parameters:
#     - kif_data: Dictionary containing the parsed KIF data.
#     - keyword: Keyword to search for in the KIF data.

#     Returns:
#     - A list of tuples matching the FOL predicates.
#     """
#     fol_triples = []
#     keyword_lower = keyword.lower()

#     # Search in instances
#     for instance in kif_data['instances']:
#         entity = instance['entity']
#         class_ = instance['class']
#         if keyword_lower in entity.lower() or keyword_lower in class_.lower():
#             fol_triples.append((entity, 'instance', class_))

#     # Search in domains
#     for domain in kif_data['domains']:
#         relation = domain['relation']
#         position = domain['position']
#         class_ = domain['class']
#         if keyword_lower in relation.lower() or keyword_lower in class_.lower():
#             fol_triples.append((relation, f'domain_{position}', class_))

#     # Search in subrelations
#     for subrelation in kif_data['subrelations']:
#         rel1 = subrelation['rel1']
#         rel2 = subrelation['rel2']
#         if keyword_lower in rel1.lower() or keyword_lower in rel2.lower():
#             fol_triples.append((rel1, 'subrelation', rel2))

#     # Search in documentation
#     for doc in kif_data['documentations']:
#         entity = doc['entity']
#         text = doc['text']
#         if keyword_lower in entity.lower() or keyword_lower in text.lower():
#             fol_triples.append((entity, 'documentation', text))

#     return fol_triples


# def normalize_string(s):
#     """
#     Normalizes a string by removing alphanumeric characters and converting to lowercase.
    
#     Parameters:
#     - s: The input string to normalize.
    
#     Returns:
#     - A normalized string with only lowercase alphabetic characters.
#     """
#     # Remove non-alphabetic characters and convert to lowercase
#     return re.sub(r'[^a-z]', '', s.lower())

# def find_fol_predicates(kif_data, keyword):
#     """
#     Finds FOL predicates in the KIF data that match the given keyword exactly, after normalization.
    
#     Parameters:
#     - kif_data: Dictionary containing the parsed KIF data.
#     - keyword: Keyword to search for in the KIF data.
    
#     Returns:
#     - A list of tuples matching the FOL predicates.
#     """
#     fol_triples = []
    
#     # Normalize the keyword
#     normalized_keyword = normalize_string(keyword)

#     # Search in instances
#     for instance in kif_data['instances']:
#         entity = instance['entity']
#         class_ = instance['class']
#         # Normalize both entity and class
#         normalized_entity = normalize_string(entity)
#         normalized_class = normalize_string(class_)
#         if normalized_entity == normalized_keyword or normalized_class == normalized_keyword:
#             fol_triples.append((entity, 'instance', class_))

#     # Search in domains
#     for domain in kif_data['domains']:
#         relation = domain['relation']
#         position = domain['position']
#         class_ = domain['class']
#         # Normalize both relation and class
#         normalized_relation = normalize_string(relation)
#         normalized_class = normalize_string(class_)
#         if normalized_relation == normalized_keyword or normalized_class == normalized_keyword:
#             fol_triples.append((relation, f'domain_{position}', class_))

#     # Search in subrelations
#     for subrelation in kif_data['subrelations']:
#         rel1 = subrelation['rel1']
#         rel2 = subrelation['rel2']
#         # Normalize both rel1 and rel2
#         normalized_rel1 = normalize_string(rel1)
#         normalized_rel2 = normalize_string(rel2)
#         if normalized_rel1 == normalized_keyword or normalized_rel2 == normalized_keyword:
#             fol_triples.append((rel1, 'subrelation', rel2))

#     # Search in documentation
#     for doc in kif_data['documentations']:
#         entity = doc['entity']
#         text = doc['text']
#         # Normalize both entity and text
#         normalized_entity = normalize_string(entity)
#         normalized_text = normalize_string(text)
#         if normalized_entity == normalized_keyword or normalized_text == normalized_keyword:
#             fol_triples.append((entity, 'documentation', text))

#     return fol_triples

def normalize_string(s):
    """
    Normalizes a string by removing alphanumeric characters and converting to lowercase.
    
    Parameters:
    - s: The input string to normalize.
    
    Returns:
    - A normalized string with only lowercase alphabetic characters.
    """
    return re.sub(r'[^a-z]', '', s.lower())

def find_fol_predicates(kif_data, keyword):
    """
    Finds relevant FOL predicates in the KIF data that match the given keyword exactly, 
    after normalization, focusing on relationships useful for LTNs.
    
    Parameters:
    - kif_data: Dictionary containing the parsed KIF data.
    - keyword: Keyword to search for in the KIF data.
    
    Returns:
    - A list of tuples matching the FOL predicates for relationships useful in LTNs.
    """
    fol_triples = []
    
    # Normalize the keyword
    normalized_keyword = normalize_string(keyword)

    # Search for instances (e.g., instance(x, Class))
    for instance in kif_data.get('instances', []):
        entity = instance['entity']
        class_ = instance['class']
        normalized_entity = normalize_string(entity)
        normalized_class = normalize_string(class_)
        if normalized_entity == normalized_keyword or normalized_class == normalized_keyword:
            fol_triples.append((entity, 'instance', class_))

    # Search for subclasses (e.g., subclass(Class1, Class2))
    for subclass in kif_data.get('subclasses', []):
        class1 = subclass['class1']
        class2 = subclass['class2']
        normalized_class1 = normalize_string(class1)
        normalized_class2 = normalize_string(class2)
        if normalized_class1 == normalized_keyword or normalized_class2 == normalized_keyword:
            fol_triples.append((class1, 'subclass', class2))

    # Search for part-whole relationships (e.g., part(x, y))
    for part in kif_data.get('parts', []):
        whole = part['whole']
        part_ = part['part']
        normalized_whole = normalize_string(whole)
        normalized_part = normalize_string(part_)
        if normalized_whole == normalized_keyword or normalized_part == normalized_keyword:
            fol_triples.append((part_, 'part', whole))

    # Search for causal relationships (e.g., causes(x, y))
    for cause in kif_data.get('causes', []):
        cause_ = cause['cause']
        effect = cause['effect']
        normalized_cause = normalize_string(cause_)
        normalized_effect = normalize_string(effect)
        if normalized_cause == normalized_keyword or normalized_effect == normalized_keyword:
            fol_triples.append((cause_, 'causes', effect))

    # Search for temporal relationships (e.g., before(x, y), after(x, y))
    for temporal in kif_data.get('temporals', []):
        event1 = temporal['event1']
        event2 = temporal['event2']
        relation = temporal['relation']  # e.g., before, after
        normalized_event1 = normalize_string(event1)
        normalized_event2 = normalize_string(event2)
        if normalized_event1 == normalized_keyword or normalized_event2 == normalized_keyword:
            fol_triples.append((event1, relation, event2))

    # Search for ownership/possession (e.g., owns(x, y))
    for ownership in kif_data.get('ownerships', []):
        owner = ownership['owner']
        owned = ownership['owned']
        normalized_owner = normalize_string(owner)
        normalized_owned = normalize_string(owned)
        if normalized_owner == normalized_keyword or normalized_owned == normalized_keyword:
            fol_triples.append((owner, 'owns', owned))

    # Search for generic related-to/connected relationships (e.g., relatedTo(x, y))
    for related in kif_data.get('relations', []):
        entity1 = related['entity1']
        entity2 = related['entity2']
        relation_type = related.get('relation_type', 'relatedTo')  # Default to 'relatedTo'
        normalized_entity1 = normalize_string(entity1)
        normalized_entity2 = normalize_string(entity2)
        if normalized_entity1 == normalized_keyword or normalized_entity2 == normalized_keyword:
            fol_triples.append((entity1, relation_type, entity2))

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


# # Example usage:
# sumo_mapping = parse_sumo_mapping('WordNetMappings30-noun.txt')
# owl_file_path = 'SUMO.kif'
# start_time = time.time()
# result_df = align_data_with_fol_predicates(objects_data.head(100), owl_file_path, sumo_mapping)
# result_df = result_df.drop('original_index', axis=1)
# end_time = time.time()

# # Calculate and print elapsed time
# time_taken = end_time - start_time
# minutes, seconds = divmod(time_taken, 60)
# print(f"Time taken: {minutes} minutes and {seconds:.2f} seconds")
# print(result_df)