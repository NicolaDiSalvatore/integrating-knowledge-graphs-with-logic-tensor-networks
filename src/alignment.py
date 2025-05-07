import joblib
import numpy as np
from scipy.spatial.distance import cosine
from nltk.corpus import wordnet as wn
import spacy
import re
from nltk.corpus import words
nlp = spacy.load("en_core_web_sm")


def load_numberbatch(file_path):
    """
    Load the Numberbatch (english) embeddings (assuming the embeddings are saved as a joblib file)
    :param file_path: file_path pointing the embeddings file
    :return: dictionary with all the (english) embeddings
    """
    embeddings = joblib.load(file_path)

    # Filter for English entries
    english_embeddings = {key: value for key, value in embeddings.items() if key.startswith('/c/en/')}
    return english_embeddings



def format_concept(value):
    """
    Format a concept in the ConceptNet format (/c/en/{value})
    :param value: concept to be formatted
    :return: the formatted concept
    """
    if value is None:
        return None
    elif not value.startswith('/c/en/'):
        return f'/c/en/{value}'
    else:
        return value



def get_synset_embedding(synsets, embeddings_dict):
    """
    Function to get the synsets embedding
    :param synsets: synsets among which I should calcualte the embedding
    :param embeddings_dict: dictionary containing the embeddings
    :return: the embedding calculated considering all the synsets
    """
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



def find_most_similar_batch(synset_embedding, vectors, batch_size=10000):
    """
    Function to find the most similar concept using batch processing
    :param synset_embedding: the embedding of the synset
    :param vectors: list containing all the embedding vectors
    :param batch_size: size of the batches used to process the 'vectors'
    :return: the index of the most similar vector and its similarity with respect to the 'synset_embedding'
    """
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


def find_most_similar_hybrid(synset_embedding, vectors, embeddings_dict, concepts, synsets, batch_size=10000):
    """
    Function to find the most similar concept in ConceptNet with respect to the 'synset_embedding'
    :param synset_embedding: the embedding of the synset
    :param vectors: list containing the embedding vectors
    :param embeddings_dict: the dictionary containing embeddings for ConceptNet concepts
    :param concepts: list of ConceptNet concept names
    :param synsets: list of synset names (without part-of-speech tags)
    :param batch_size: size of the batches
    :return: the most similar concept and its similarity with respect to the synset
    """
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


def find_similar_concepts_in_dataframe(df, embeddings_dict, vectors, concepts):
    """
        Function to find the most similar ConceptNet concept for a DataFrame of WordNet synsets.

        :param df: DataFrame containing a column 'synsets' with lists of WordNet synsets.
        :param embeddings_dict: Dictionary containing Numberbatch embeddings ({concept: embedding}).
        :param vectors: List of all concept embeddings used for batch similarity search.
        :param concepts: List of ConceptNet concept names corresponding to 'vectors'.
        :return: The updated DataFrame with new columns 'most_similar_concept' and 'similarity'.
    """
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









def get_synset_embedding_with_pos(synsets, embeddings_dict):
    """
    Computes the average embedding vector for a given list of synsets
    and determines the part-of-speech (POS) of the first valid lemma.

    :param synsets: List of WordNet synset names (strings).
    :param embeddings_dict: Dictionary mapping lemma keys to their embedding vectors.
    :return: Tuple containing the average embedding vector and the POS tag of the first valid lemma.
    """
    vectors = []
    synsets_pos = None

    for synset_name in synsets:
        print(f"Processing synset: {synset_name}")
        try:
            synset = wn.synset(synset_name)
            lemmas = synset.lemma_names()
            for lemma in lemmas:
                # Eliminate POS and index
                lemma_key = f'/c/en/{lemma.lower() }'

                if lemma_key in embeddings_dict:
                    vectors.append(embeddings_dict[lemma_key])
                    if synsets_pos is None:
                        synsets_pos = nlp(lemma)
                else:
                    print(f"Missing embedding for {lemma_key}")

        except Exception as e:
            print(f"Error processing synset {synset_name}: {e}")

    if vectors:
        return np.mean(vectors, axis=0), synsets_pos  # Return the average vector
    else:
        print("No valid vectors found.")
        return None, None







#
def is_valid_concept(concept):

    """
    Function to check if a concept is valid or not.
    :param concept: The concept to be validated, expected to be a string.
    :return: A boolean indicating whether the concept is valid.
    """
    # Check if the concept contains only alphabetic characters
    if not re.match(r'^[a-zA-Z]+$', concept):
        return False

    # Check if the concept is in the list of English words
    if concept.lower() not in words.words():
        return False

    # Use spaCy to check if the concept is a named entity
    doc = nlp(concept)
    for token in doc:
        if token.ent_type_ != "":  # Exclude named entities
            return False

    return True


# Example usage
# concepts = ['/c/en/dog', '/c/en/ablude', '/c/en/ahwai', '/c/en/50', '/c/en/run']
# valid_concepts = []
#
# for concept in concepts:
#     lemma = concept.split('/')[2]  # Extract the lemma from the concept
#     if is_valid_concept(lemma):
#         valid_concepts.append(concept)
#
# print("Valid concepts:", valid_concepts)

def find_top_5_similar_batch(synset_embedding, vectors, concepts, synset_pos, batch_size=10000):
    """
    Function to find the top 5 concepts similar to a given synset embedding from a batch of vectors.

    :param synset_embedding: The embedding of the synset to compare against, expected to be a numpy array.
    :param vectors: A list or array of concept embeddings to compare with the synset embedding.
    :param concepts: A list of concept identifiers corresponding to the vectors.
    :param synset_pos: The part of speech (POS) tag of the synset to filter concepts by.
    :param batch_size: The size of batches to process at a time, default is 10,000.
    :return: A tuple containing a list of the top 5 concept identifiers and their corresponding similarity scores.
    """
    if synset_embedding is None:
        return [], []

    # Normalize the synset_embedding
    synset_embedding = synset_embedding / np.linalg.norm(synset_embedding)

    top_20_concepts = []

    num_batches = len(vectors) // batch_size + (1 if len(vectors) % batch_size != 0 else 0)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(vectors))

        # Get the current batch of vectors
        batch = vectors[start_idx:end_idx]

        if len(batch) == 0:
            continue

        # Normalize the batch
        batch_norm = batch / np.linalg.norm(batch, axis=1, keepdims=True)

        # Compute cosine similarity between synset_embedding and batch
        similarities = np.dot(batch_norm, synset_embedding)

        # Sort indices by descending similarity
        top_batch_idx = np.argsort(similarities)[::-1]  # Indices in descending order of similarity
        top_batch_sim = similarities[top_batch_idx]

        # Convert np.float32 to regular float
        top_batch_sim = [float(sim) for sim in top_batch_sim]

        # Collect the top 20 most similar concepts regardless of POS
        for idx, sim in zip(top_batch_idx, top_batch_sim):
            concept_lemma = concepts[start_idx + idx].split('/')[-1]

            # Apply filtering to exclude strange concepts
            if is_valid_concept(concept_lemma):
                top_20_concepts.append((start_idx + idx, sim))

            # Stop once we've collected 20 concepts
            if len(top_20_concepts) >= 20:
                break

        # Stop processing further batches if 20 concepts are found
        if len(top_20_concepts) >= 20:
            break

    # Now that we have the top 20, filter them by POS
    matching_pos_concepts = [(idx, sim) for idx, sim in top_20_concepts if nlp(concepts[idx].split('/')[-1]) == synset_pos]

    # If there are fewer than 5 matching POS concepts, fill with other top concepts
    if len(matching_pos_concepts) < 5:
        remaining_slots = 5 - len(matching_pos_concepts)
        # Fill the rest with the top concepts (regardless of POS), avoiding duplicates
        for idx, sim in top_20_concepts:
            if len(matching_pos_concepts) >= 5:
                break
            if (idx, sim) not in matching_pos_concepts:
                matching_pos_concepts.append((idx, sim))

    # Sort the final concepts by similarity and ensure exactly 5 concepts
    top_5_concepts = sorted(matching_pos_concepts, key=lambda x: x[1], reverse=True)[:5]

    return [concepts[idx] for idx, sim in top_5_concepts], [sim for idx, sim in top_5_concepts]


def find_top_5_similar_hybrid(synset_embedding, vectors, embeddings_dict, concepts, synsets, synsets_pos, batch_size=10000):
    """
    Finds the top 5 concepts most similar to a given synset embedding by combining direct lookup in an embeddings dictionary
    and a batch-based search through a list of vectors.

    :param synset_embedding: The embedding of the synset to compare against.
    :param vectors: A list of vectors representing concept embeddings.
    :param embeddings_dict: A dictionary mapping concept names to their embeddings.
    :param concepts: A list of concept names corresponding to the vectors.
    :param synsets: A list of synset names to check for direct similarity.
    :param synsets_pos: The part of speech for the synsets, used in batch search.
    :param batch_size: The size of batches to process at a time in the batch search.
    :return: A tuple containing a list of the top 5 concept names and their corresponding similarity scores.
    """
    if synset_embedding is None:
        return [], []

    top_concepts = []


    # Check similarity with lemmas and synset name
    for synset_name in synsets:
        lemma_key = f'/c/en/{synset_name.split(".")[0].lower()}'
        if lemma_key in embeddings_dict:
            similarity = 1 - cosine(synset_embedding, embeddings_dict[lemma_key])
            top_concepts.append((lemma_key, similarity))
        else:
            print(f"Missing embedding for {lemma_key}")

    # If not enough concepts, fallback to batch search
    if len(top_concepts) < 5:
        top_batch_concepts, top_batch_similarities = find_top_5_similar_batch(
            synset_embedding, vectors, concepts, synsets_pos, batch_size)
        top_concepts.extend(zip(top_batch_concepts, top_batch_similarities))

    # Sort by similarity and return top 5
    top_concepts = sorted(top_concepts, key=lambda x: x[1], reverse=True)[:5]
    return [c[0] for c in top_concepts], [c[1] for c in top_concepts]

def find_top_5_similar_concepts_in_dataframe(df, embeddings_dict, vectors, concepts):
    """
    Finds the top 5 similar concepts for each row in a DataFrame based on synset embeddings and updates the DataFrame
    with these concepts and their similarity scores.

    :param df: A pandas DataFrame containing a column 'synsets' with synset information for each row.
    :param embeddings_dict: A dictionary mapping concept names to their embeddings.
    :param vectors: A list of vectors representing concept embeddings.
    :param concepts: A list of concept names corresponding to the vectors.
    :return: The updated DataFrame with two new columns: 'top_5_concepts' and 'top_5_similarities'.
    """
    df['top_5_concepts'] = None
    df['top_5_similarities'] = None
    synset_cache = {}

    for idx, row in df.iterrows():
        synsets = row['synsets']
        synset_key = tuple(sorted(synsets))

        # Check cache
        if synset_key in synset_cache:
            top_5_concepts, top_5_similarities = synset_cache[synset_key]
        else:
            synset_embedding, synsets_pos = get_synset_embedding_with_pos(synsets, embeddings_dict)

            # Use hybrid approach
            top_5_concepts, top_5_similarities = find_top_5_similar_hybrid(
                synset_embedding, vectors, embeddings_dict, concepts, synsets, synsets_pos)

            synset_cache[synset_key] = (top_5_concepts, top_5_similarities)

        # Update DataFrame
        if top_5_concepts:
            df.at[idx, 'top_5_concepts'] = top_5_concepts
            df.at[idx, 'top_5_similarities'] = top_5_similarities
        else:
            print(f"No top concepts found for index {idx}")

    return df



