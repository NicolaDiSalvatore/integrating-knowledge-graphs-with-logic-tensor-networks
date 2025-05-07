import sqlite3
import pandas as pd
import joblib
import re
import time


semantic_relationships = ['/r/Antonym', '/r/DefinedAs', '/r/DistinctFrom',
    '/r/EtymologicallyDerivedFrom', '/r/EtymologicallyRelatedTo', '/r/FormOf',
    '/r/HasProperty', '/r/MannerOf', '/r/SimilarTo', '/r/SymbolOf', '/r/Synonym'
]
spatial_relationships = ['/r/AtLocation', '/r/LocatedNear']
ontological_relationships = ['/r/DerivedFrom', '/r/InstanceOf', '/r/IsA', '/r/HasA', '/r/PartOf']


def find_extended_synonyms_in_df(df, admissible_concepts=None, db_path="assertions.db",  max_depth=2):
    """
    Finds extended synonyms for each concept in the DataFrame by querying ConceptNet.

    Parameters:
    df (pd.DataFrame): DataFrame containing a 'concept' column with concepts to find synonyms for.
    admissible_concepts (set or list, optional): A set of valid concepts. If provided, only synonyms in this set are kept.
    db_path (str, optional): Path to the SQLite database containing ConceptNet data.
    max_depth (int, optional): Maximum recursion depth for finding synonyms.

    Returns:
    pd.DataFrame: The input DataFrame with an additional column 'extended_synonyms' containing lists of synonyms.
    """
    # Ensure admissible_concepts is a set for fast lookup

    def normalize_concept(concept):
        """
        Normalize a concept to match ConceptNet's expected format.

        Parameters:
        concept (str): The concept to be normalized.

        Returns:
        str: The normalized concept in the format '/c/en/lemma'.
        """
        concept = concept.lower().strip()
        if not concept.startswith("/c/en/"):
            concept = f"/c/en/{concept}"
        return concept


    if admissible_concepts is not None:
        admissible_concepts = set(normalize_concept(concept) for concept in admissible_concepts)

    # Establish a database connection
    conn = sqlite3.connect(db_path)
    cache = {}

    def retrieve_synonyms(concept, depth=1):
        """
        Recursively find synonyms of a given concept up to a specified depth.

        :param concept: The concept for which synonyms need to be retrieved (expected in ConceptNet format, e.g., '/c/en/dog').
        :param depth: The current recursion depth (default is 1).
        :return: A set of synonyms for the given concept.
        """
        if depth > max_depth:
            return set()
        if concept in cache:
            return cache[concept]

        # Query to find synonyms and similar relationships
        query = """
            SELECT end FROM conceptnet WHERE start = ?
            AND (relation = '/r/Synonym'  )
            AND end LIKE '/c/en/%'

            UNION

            SELECT start FROM conceptnet WHERE end = ?
            AND (relation = '/r/Synonym' )
            AND start LIKE '/c/en/%'
        """

        results = conn.execute(query, (concept, concept)).fetchall()
        direct_synonyms = {normalize_concept(row[0]) for row in results}

        all_synonyms = set(direct_synonyms)
        for synonym in direct_synonyms:
            all_synonyms.update(retrieve_synonyms(synonym, depth + 1))

        cache[concept] = all_synonyms
        # all_synonyms = {postprocessing_concept(concept) for concept in all_synonyms}
        return all_synonyms

    # Apply the synonym search to each concept in the DataFrame
    def get_synonyms_for_row(concept):
        """
        Retrieve and return sorted synonyms for a given concept.

        :param concept: The concept for which synonyms should be retrieved.
                            Expected in ConceptNet format (e.g., '/c/en/dog').
        :return: A sorted list of synonyms.
        """
        concept = normalize_concept(concept)
        return sorted(retrieve_synonyms(concept))

    def postprocessing_concepts_list(concepts_list):
        """
        Processes a list of concepts by removing language tags and filtering out the root concept.

        :param concepts_list: A list of concept strings in the format "/c/en/concept".
        :return: A list of unique concepts without language tags.
        """
        filtered_concepts = {re.sub(r"/[a-z]+$", "", concept) for concept in concepts_list if concept != "/c/en"}
        return list(filtered_concepts)

    # Add a new column with synonyms
    df['extended_synonyms'] = df['concept'].apply(get_synonyms_for_row)
    df['extended_synonyms'] = df['extended_synonyms'].apply(postprocessing_concepts_list)
    # Close the database connection
    conn.close()

    return df


def find_extended_antonyms_in_df(df, admissible_concepts=None, db_path="assertions.db",  max_depth=0):
    """
    Finds and adds extended antonyms for each concept in the given DataFrame.

    :param df: A pandas DataFrame with a 'concept' column containing concepts in ConceptNet format.
    :param admissible_concepts: A set of admissible concepts to filter results (default: None).
    :param db_path: Path to the SQLite database containing ConceptNet relationships (default: "assertions.db").
    :param max_depth: Maximum depth for recursive antonym retrieval (default: 0).
    :return: The input DataFrame with an added 'extended_antonyms' column.
    """
    # Ensure admissible_concepts is a set for fast lookup

    def normalize_concept(concept):
        """Normalize the concept to match ConceptNet's expected format."""
        concept = concept.lower().strip()
        if not concept.startswith("/c/en/"):
            concept = f"/c/en/{concept}"
        return concept


    if admissible_concepts is not None:
        admissible_concepts = set(normalize_concept(concept) for concept in admissible_concepts)

    # Establish a database connection
    conn = sqlite3.connect(db_path)
    cache = {}

    def retrieve_antonyms(concept, depth=0):
        """Recursively find synonyms up to a certain depth, filtering by admissible_concepts."""
        if depth > max_depth:
            return set()
        if concept in cache:
            return cache[concept]

        # Query to find synonyms and similar relationships
        query = """
            SELECT end FROM conceptnet WHERE start = ?
            AND relation = '/r/Antonym' 
            AND end LIKE '/c/en/%'

            UNION

            SELECT start FROM conceptnet WHERE end = ?
            AND relation =  '/r/Antonym'
            AND start LIKE '/c/en/%'
        """

        results = conn.execute(query, (concept, concept)).fetchall()
        direct_synonyms = {normalize_concept(row[0]) for row in results}

        all_synonyms = set(direct_synonyms)
        for synonym in direct_synonyms:
            all_synonyms.update(retrieve_antonyms(synonym, depth + 1))

        cache[concept] = all_synonyms
        # all_synonyms = {postprocessing_concept(concept) for concept in all_synonyms}
        return all_synonyms

    # Apply the synonym search to each concept in the DataFrame
    def get_antonyms_for_row(concept):
        concept = normalize_concept(concept)
        return sorted(retrieve_antonyms(concept))

    def postprocessing_concepts_list(concepts_list):
        for concept in concepts_list:
            concept = re.sub(r"/[a-z]+$", "", concept)
            if concept == "/c/en":
                concept = ""
            if (admissible_concepts is not None) and (concept not in admissible_concepts):
                concept = ""
            concept = re.sub(r"/[a-z]+$", "", concept)
        return concepts_list


    # Add a new column with synonyms
    df['extended_antonyms'] = df['concept'].apply(get_antonyms_for_row)
    df['extended_antonyms'] = df['extended_antonyms'].apply(postprocessing_concepts_list)
    # Close the database connection
    conn.close()

    return df

# Example usage



