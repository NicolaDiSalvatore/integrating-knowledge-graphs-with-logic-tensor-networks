import sqlite3
import pandas as pd


semantic_relationships = ['/r/Antonym', '/r/DefinedAs', '/r/DistinctFrom',
    '/r/EtymologicallyDerivedFrom', '/r/EtymologicallyRelatedTo', '/r/FormOf',
    '/r/HasProperty', '/r/MannerOf', '/r/NotHasProperty',
    '/r/RelatedTo', '/r/SimilarTo', '/r/SymbolOf', '/r/Synonym'
]
spatial_relationships = ['/r/AtLocation', '/r/LocatedNear']
ontological_relationships = ['/r/DerivedFrom', '/r/InstanceOf', '/r/IsA', '/r/HasA', '/r/PartOf']


def find_hierarchical_concepts_batch(df, concept_column = 'concept', hypernyms_column = 'Hypernyms', hyponyms_column = 'Hyponyms', db_path="assertions.db"):
    """
     Finds hypernyms and hyponyms for each concept in a given DataFrame using ConceptNet stored in an SQLite database.

    :param df (pd.DataFrame): DataFrame containing concepts.
    :param concept_column (str): Column name containing concepts.
    :param hypernyms_column (str): Column name to store hypernyms.
    :param hyponyms_column (str): Column name to store hyponyms.
    :param db_path (str): Path to the SQLite database file.
    :return pd.DataFrame: DataFrame with additional columns for hypernyms and hyponyms.
    """
    conn = sqlite3.connect(db_path)

    def get_hierarchies_iterative(concept, relation_type, cache):
        if concept in cache:
            return cache[concept]

        hierarchies = set()
        stack = [concept]

        while stack:
            current_concept = stack.pop()

            query = f"""
            SELECT end FROM conceptnet WHERE start = ? AND relation = '{relation_type}' AND end LIKE '/c/en/%'
            """
            results = conn.execute(query, (current_concept,)).fetchall()

            for row in results:
                end_concept = row[0]
                if end_concept not in hierarchies:
                    hierarchies.add(end_concept)
                    stack.append(end_concept)

        cache[concept] = list(hierarchies)
        return cache[concept]

    def get_filtered_hierarchies(concept):
        is_a_hierarchies = get_hierarchies_iterative(concept, '/r/IsA', cache={})
        hypernyms = is_a_hierarchies
        hyponyms = []

        query = """
        SELECT start FROM conceptnet WHERE end = ? AND relation = '/r/IsA' AND end LIKE '/c/en/%'
        """
        results = conn.execute(query, (concept,)).fetchall()
        hyponyms = [row[0] for row in results]  # Hyponyms are concepts that point to this concept as a hypernym

        return hypernyms, hyponyms

    cache = {}

    df[[hypernyms_column, hyponyms_column]] = df[concept_column].apply(
        lambda concept: pd.Series(get_filtered_hierarchies(concept))
    )

    conn.close()

    return df






print("Finisched")