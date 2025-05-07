import sqlite3
import pandas as pd

REFLEXIVE_RELATIONSHIPS = {
    '/r/Synonym', '/r/Antonym', '/r/RelatedTo', '/r/SimilarTo', '/r/DistinctFrom', '/r/EtymologicallyRelatedTo', '/r/LocatedNear',
}


# Function to fetch in batches
def fetch_in_batches(cursor, query, parameters, batch_size=1000):
    """
       Fetches query results in batches to optimize performance when dealing with large datasets.

       :param cursor: Database cursor object used to execute queries.
       :param query: SQL query string with placeholders for parameters.
       :param parameters: List of parameters to be used in the query.
       :param batch_size: Number of parameters to process in each batch (default is 1000).
       :yield: Results of the executed query in batches.
       """
    for i in range(0, len(parameters), batch_size):
        batch_parameters = parameters[i:i + batch_size]
        print(f"Executing batch with {len(batch_parameters)} parameters.")
        cursor.execute(query, batch_parameters)
        yield cursor.fetchall()

def execute_query_in_batches(cursor, base_query, concepts, batch_size=500):
    results = []
    
    # Split the concepts list into batches to avoid too many variables error
    for i in range(0, len(concepts), batch_size):
        batch = concepts[i:i + batch_size]
        placeholders = ','.join('?' for _ in batch)
        query = base_query.format(placeholders=placeholders)
        
        print(f"Executing batch query with {len(batch)} parameters.")
        cursor.execute(query, batch + batch)  # Using the batch twice for start and end
        results.extend(cursor.fetchall())
    
    return results

# Function to find synonyms from the database
def find_synonyms(concepts, cursor, batch_size=500):
    base_query = """
        SELECT start, end
        FROM conceptnet
        WHERE relation = '/r/Synonym'
        AND (start IN ({placeholders}) OR end IN ({placeholders}))
    """
    
    print(f"Executing find_synonyms with batch size: {batch_size}")
    return execute_query_in_batches(cursor, base_query, concepts, batch_size)

def find_relationships_with_synonyms(concept_df, db_path="assertions.db", batch_size=500):
    """
       Queries the ConceptNet database to find relationships between a large set of concepts,
       considering their synonyms and reflexive relationships. Optimized for large datasets.
       This version focuses only on relationships between original concepts.

       :param concept_df: DataFrame containing a column 'concept' with concept URIs.
       :param db_path: Path to the ConceptNet database.
       :param batch_size: Size of each query batch to avoid SQLite's variable limits.
       :return: DataFrame containing the relationships between the concepts.
                Columns: ['relation', 'start', 'end', 'metadata']
       """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Extract the list of concepts from the DataFrame
    concepts = concept_df['concept'].tolist()

    # Step 1: Find synonyms for all concepts
    synonyms = find_synonyms(concepts, cursor, batch_size)

    # Step 2: Expand the list of concepts to include their synonyms
    expanded_concepts = set(concepts)  # Start with original concepts
    for start, end in synonyms:
        expanded_concepts.add(start)
        expanded_concepts.add(end)

    # Convert the expanded set back to a list
    expanded_concepts = list(expanded_concepts)

    # Step 3: Query for relationships between all concepts (including synonyms)
    base_query = """
        SELECT relation, start, end, metadata
        FROM conceptnet
        WHERE start IN ({placeholders}) OR end IN ({placeholders})
    """
    
    print(f"Executing find_relationships_with_synonyms with batch size: {batch_size}")
    relationships = execute_query_in_batches(cursor, base_query, expanded_concepts, batch_size)

    # Step 4: Filter relationships to keep only those involving the original concepts
    filtered_rows = []
    seen_pairs = set()  # To track pairs we've already processed

    for relation, start, end, metadata in relationships:
        if start in concepts and end in concepts:
            if (start, end) not in seen_pairs and (end, start) not in seen_pairs:
                if relation in REFLEXIVE_RELATIONSHIPS:
                    # If the relationship is reflexive, only count it once
                    seen_pairs.add((start, end))
                    seen_pairs.add((end, start))
                else:
                    # For non-reflexive relationships, count them as is
                    seen_pairs.add((start, end))

                # Add the relationship to the filtered list
                filtered_rows.append((relation, start, end, metadata))

    # Step 5: Create a DataFrame from the results
    filtered_df = pd.DataFrame(filtered_rows, columns=['relation', 'start', 'end', 'metadata'])

    conn.close()
    
    return filtered_df

def format_predicate(value):
    if value is None:
        return None
    elif not value.startswith('/c/en/'):
        return f'/c/en/{value}'
    else:
        return value


def find_relationships_with_synonyms_complete(concept_df, db_path="assertions.db", batch_size=500):
    """
    Find relationships between concepts and their synonyms in a ConceptNet database.
    Returns two DataFrames:
    1. One where the start node is in the original concepts.
    2. One where the end node is in the original concepts.

    :param concept_df: DataFrame with a column 'concept' containing the list of concepts.
    :param db_path: Path to the ConceptNet SQLite database.
    :param batch_size: Batch size for querying the database.

    :return:
        - start_node_df: DataFrame containing triples where the start node is in the original concepts.
        - end_node_df: DataFrame containing triples where the end node is in the original concepts.
    """


    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Extract the list of concepts from the DataFrame
    concepts = concept_df['concept'].tolist()

    # Step 1: Find synonyms for all concepts (using a pre-defined function)
    synonyms = find_synonyms(concepts, cursor, batch_size)

    # Step 2: Expand the list of concepts to include their synonyms
    expanded_concepts = set(concepts)  # Start with original concepts
    for start, end in synonyms:
        expanded_concepts.add(start)
        expanded_concepts.add(end)

    # Convert the expanded set back to a list
    expanded_concepts = list(expanded_concepts)

    # Step 3: Query for relationships between all concepts (including synonyms)
    base_query = """
        SELECT relation, start, end, metadata
        FROM conceptnet
        WHERE start IN ({placeholders}) OR end IN ({placeholders}) AND relation IN
    """

    print(f"Executing find_relationships_with_synonyms with batch size: {batch_size}")
    relationships = execute_query_in_batches(cursor, base_query, expanded_concepts, batch_size)

    # Step 4: Create two lists for filtered relationships
    start_node_rows = []  # To store triples where start node is in the original concepts
    end_node_rows = []  # To store triples where end node is in the original concepts
    seen_pairs = set()  # To track pairs we've already processed

    for relation, start, end, metadata in relationships:
        # Case 1: If the start node is in the original concepts
        if start in concepts:
            if (start, end) not in seen_pairs:
                if relation in REFLEXIVE_RELATIONSHIPS:
                    # Track both directions in reflexive relationships
                    seen_pairs.add((start, end))
                    seen_pairs.add((end, start))
                else:
                    seen_pairs.add((start, end))
                # Add to start_node_rows
                start_node_rows.append((relation, start, end, metadata))

        # Case 2: If the end node is in the original concepts
        if end in concepts:
            if (end, start) not in seen_pairs:
                if relation in REFLEXIVE_RELATIONSHIPS:
                    # Track both directions in reflexive relationships
                    seen_pairs.add((end, start))
                    seen_pairs.add((start, end))
                else:
                    seen_pairs.add((end, start))
                # Add to end_node_rows
                end_node_rows.append((relation, start, end, metadata))

    # Step 5: Create two DataFrames from the results
    start_node_df = pd.DataFrame(start_node_rows, columns=['relation', 'start', 'end', 'metadata'])
    end_node_df = pd.DataFrame(end_node_rows, columns=['relation', 'start', 'end', 'metadata'])

    conn.close()

    # Return the two DataFrames
    return start_node_df, end_node_df






def add_related_and_capable_columns(concept_df, db_path="assertions.db", batch_size=500, delimiter=', '):
    """
    Adds three columns to the input DataFrame:
    1. 'relatedto': concepts that are related to the original concepts in extended_antonyms (in either start or end node).
    2. 'capableof': concepts that the original concepts are capable of (where the concept appears as the end node).
    3. 'notcapableof': concepts that the original concepts are not capable of (where the concept appears as the end node).

    Parameters:
    - concept_df: DataFrame with an 'extended_antonyms' column containing lists of strings.
    - db_path: Path to the ConceptNet SQLite database.
    - batch_size: Number of concepts to query at a time for efficiency.
    - delimiter: Delimiter to use when concatenating strings.

    Returns:
    - concept_df: Original DataFrame with added 'relatedto', 'capableof', and 'notcapableof' columns.
    """

    def fetch_related_and_capable_for_batch(batch_concepts, cursor):
        # Fetch related and capable concepts from the database
        placeholders = ', '.join(['?'] * len(batch_concepts))

        # Query for RelatedTo (reflexive)
        related_query = f"""
            SELECT start, end
            FROM conceptnet
            WHERE (start IN ({placeholders}) OR end IN ({placeholders}))
            AND relation = '/r/RelatedTo'
            AND start LIKE '/c/en/%' AND end LIKE '/c/en/%'
        """
        cursor.execute(related_query, batch_concepts * 2)  # Duplicate list for both start and end placeholders
        related_results = cursor.fetchall()
        print(f"Related results for {batch_concepts}: {related_results}")  # Debug print

        # Query for CapableOf (non-reflexive, concept as end node)
        capable_query = f"""
            SELECT start, end
            FROM conceptnet
            WHERE end IN ({placeholders})
            AND relation = '/r/CapableOf'
            AND start LIKE '/c/en/%' AND end LIKE '/c/en/%'
        """
        cursor.execute(capable_query, batch_concepts)
        capable_results = cursor.fetchall()
        print(f"Capable results for {batch_concepts}: {capable_results}")  # Debug print

        # Query for NotCapableOf (non-reflexive, concept as end node)
        not_capable_query = f"""
            SELECT start, end
            FROM conceptnet
            WHERE end IN ({placeholders})
            AND relation = '/r/NotCapableOf'
            AND start LIKE '/c/en/%' AND end LIKE '/c/en/%'
        """
        cursor.execute(not_capable_query, batch_concepts)
        not_capable_results = cursor.fetchall()
        print(f"Not capable results for {batch_concepts}: {not_capable_results}")  # Debug print

        # Organize results into strings for efficient lookup
        relatedto_dict = {concept: [] for concept in batch_concepts}
        for start, end in related_results:
            if start in relatedto_dict:
                relatedto_dict[start].append(end)
            if end in relatedto_dict:
                relatedto_dict[end].append(start)

        capableof_dict = {concept: [] for concept in batch_concepts}
        for start, end in capable_results:
            if end in capableof_dict:
                capableof_dict[end].append(start)

        notcapableof_dict = {concept: [] for concept in batch_concepts}
        for start, end in not_capable_results:
            if end in notcapableof_dict:
                notcapableof_dict[end].append(start)

        # Concatenate lists into strings with a delimiter
        for concept in relatedto_dict:
            relatedto_dict[concept] = delimiter.join(relatedto_dict[concept])
        for concept in capableof_dict:
            capableof_dict[concept] = delimiter.join(capableof_dict[concept])
        for concept in notcapableof_dict:
            notcapableof_dict[concept] = delimiter.join(notcapableof_dict[concept])

        return relatedto_dict, capableof_dict, notcapableof_dict

    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Initialize new columns
    concept_df['related_to'] = ''
    concept_df['capable_of'] = ''
    concept_df['not_capable_of'] = ''

    # Iterate through each row in the DataFrame
    for index, row in concept_df.iterrows():
        # Flatten the list of strings from the extended_synonyms column
        batch_concepts = [row['concept']]
                          # + row['extended_synonyms']

        if batch_concepts:  # Check if the list is not empty
            relatedto_dict, capableof_dict, notcapableof_dict = fetch_related_and_capable_for_batch(batch_concepts,
                                                                                                    cursor)

            # Assign results back to the DataFrame
            first_concept = batch_concepts[0]  # Use the first concept to get values
            concept_df.at[index, 'related_to'] = relatedto_dict.get(first_concept, '')
            concept_df.at[index, 'capable_of'] = capableof_dict.get(first_concept, '')
            concept_df.at[index, 'not_capable_of'] = notcapableof_dict.get(first_concept, '')

    conn.close()
    return concept_df


def add_related_and_capable_to_concept_columns(concept_df, db_path="assertions.db", batch_size=500, delimiter=', '):
    """
    Adds three columns to the input DataFrame:
    1. 'relatedto': concepts that are related to the original concept (in either start or end node).
    2. 'capableof': concepts that the original concept is capable of (where the concept appears as the end node).
    3. 'notcapableof': concepts that the original concept is not capable of (where the concept appears as the end node).

    :param concept_df: DataFrame with a 'concept' column containing strings.
    :param db_path: Path to the ConceptNet SQLite database.
    :param batch_size: Number of concepts to query at a time for efficiency.
    :param delimiter: Delimiter to use when concatenating strings.

    :return: concept_df: Original DataFrame with added 'relatedto', 'capableof', and 'notcapableof' columns.
    """

    def fetch_related_and_capable_for_batch(batch_concepts, cursor):
        # Fetch related and capable concepts from the database
        placeholders = ', '.join(['?'] * len(batch_concepts))

        # Query for RelatedTo (reflexive)
        related_query = f"""
            SELECT start, end
            FROM conceptnet
            WHERE (start IN ({placeholders}) OR end IN ({placeholders}))
            AND relation = '/r/RelatedTo'
            AND start LIKE '/c/en/%' AND end LIKE '/c/en/%'
        """
        cursor.execute(related_query, batch_concepts * 2)  # Duplicate list for both start and end placeholders
        related_results = cursor.fetchall()

        # Query for CapableOf (non-reflexive, concept as end node)
        capable_query = f"""
            SELECT start, end
            FROM conceptnet
            WHERE end IN ({placeholders})
            AND relation = '/r/CapableOf'
            AND start LIKE '/c/en/%' AND end LIKE '/c/en/%'
        """
        cursor.execute(capable_query, batch_concepts)
        capable_results = cursor.fetchall()

        # Query for NotCapableOf (non-reflexive, concept as end node)
        not_capable_query = f"""
            SELECT start, end
            FROM conceptnet
            WHERE end IN ({placeholders})
            AND relation = '/r/NotCapableOf'
            AND start LIKE '/c/en/%' AND end LIKE '/c/en/%'
        """
        cursor.execute(not_capable_query, batch_concepts)
        not_capable_results = cursor.fetchall()

        # Organize results into dictionaries
        relatedto_dict = {concept: [] for concept in batch_concepts}
        for start, end in related_results:
            if start in relatedto_dict:
                relatedto_dict[start].append(end)
            if end in relatedto_dict:
                relatedto_dict[end].append(start)

        capableof_dict = {concept: [] for concept in batch_concepts}
        for start, end in capable_results:
            if end in capableof_dict:
                capableof_dict[end].append(start)

        notcapableof_dict = {concept: [] for concept in batch_concepts}
        for start, end in not_capable_results:
            if end in notcapableof_dict:
                notcapableof_dict[end].append(start)

        # Concatenate lists into strings with a delimiter
        for concept in relatedto_dict:
            relatedto_dict[concept] = delimiter.join(relatedto_dict[concept])
        for concept in capableof_dict:
            capableof_dict[concept] = delimiter.join(capableof_dict[concept])
        for concept in notcapableof_dict:
            notcapableof_dict[concept] = delimiter.join(notcapableof_dict[concept])

        return relatedto_dict, capableof_dict, notcapableof_dict

    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Initialize new columns
    concept_df['related_to'] = ''
    concept_df['capable_of'] = ''
    concept_df['not_capable_of'] = ''

    # Process the DataFrame in batches
    concept_list = concept_df['concept'].tolist()
    for i in range(0, len(concept_list), batch_size):
        batch_concepts = concept_list[i:i + batch_size]

        # Fetch related, capable, and not capable concepts for the current batch
        relatedto_dict, capableof_dict, notcapableof_dict = fetch_related_and_capable_for_batch(batch_concepts, cursor)

        # Assign results back to the DataFrame
        for concept in batch_concepts:
            concept_df.loc[concept_df['concept'] == concept, 'related_to'] = relatedto_dict.get(concept, '')
            concept_df.loc[concept_df['concept'] == concept, 'capable_of'] = capableof_dict.get(concept, '')
            concept_df.loc[concept_df['concept'] == concept, 'not_capable_of'] = notcapableof_dict.get(concept, '')

    # Close the database connection
    conn.close()
    return concept_df

def add_related_and_capable_to_synonyms_columns(concept_df, db_path="assertions.db", batch_size=500, delimiter=', '):
    """
    Adds related and capable concepts to each row in the given DataFrame by querying the ConceptNet database.

    :param concept_df: A pandas DataFrame containing a 'concept' column and an 'extended_synonyms' column.
    :param db_path: Path to the SQLite database containing ConceptNet relationships (default: "assertions.db").
    :param batch_size: Number of concepts to process in each batch (default: 500).
    :param delimiter: Delimiter used to concatenate related concepts in the output (default: ', ').
    :return: The input DataFrame with three new columns:
             - 'related_to_with_synonyms': Concepts related to the given concept and its synonyms.
             - 'capable_of_with_synonyms': Concepts that the given concept and its synonyms are capable of.
             - 'not_capable_of_with_synonyms': Concepts that the given concept and its synonyms are *not* capable of.
    """

    def fetch_related_and_capable_for_batch(batch_concepts, cursor):
        """
        Fetches related and capable concepts from the database for a batch of concepts.

        :param batch_concepts: A list of concept strings to query.
        :param cursor: SQLite database cursor to execute queries.
        :return: Three dictionaries mapping each concept in the batch to its related, capable, and not capable concepts.
        """
        # Fetch related and capable concepts from the database
        placeholders = ', '.join(['?'] * len(batch_concepts))

        # Query for RelatedTo (reflexive)
        related_query = f"""
            SELECT start, end
            FROM conceptnet
            WHERE (start IN ({placeholders}) OR end IN ({placeholders}))
            AND relation = '/r/RelatedTo'
            AND start LIKE '/c/en/%' AND end LIKE '/c/en/%'
        """
        cursor.execute(related_query, batch_concepts * 2)  # Duplicate list for both start and end placeholders
        related_results = cursor.fetchall()
        print(f"Related results for {batch_concepts}: {related_results}")  # Debug print

        # Query for CapableOf (non-reflexive, concept as end node)
        capable_query = f"""
            SELECT start, end
            FROM conceptnet
            WHERE end IN ({placeholders})
            AND relation = '/r/CapableOf'
            AND start LIKE '/c/en/%' AND end LIKE '/c/en/%'
        """
        cursor.execute(capable_query, batch_concepts)
        capable_results = cursor.fetchall()
        print(f"Capable results for {batch_concepts}: {capable_results}")  # Debug print

        # Query for NotCapableOf (non-reflexive, concept as end node)
        not_capable_query = f"""
            SELECT start, end
            FROM conceptnet
            WHERE end IN ({placeholders})
            AND relation = '/r/NotCapableOf'
            AND start LIKE '/c/en/%' AND end LIKE '/c/en/%'
        """
        cursor.execute(not_capable_query, batch_concepts)
        not_capable_results = cursor.fetchall()
        print(f"Not capable results for {batch_concepts}: {not_capable_results}")  # Debug print

        # Organize results into strings for efficient lookup
        relatedto_dict = {concept: [] for concept in batch_concepts}
        for start, end in related_results:
            if start in relatedto_dict:
                relatedto_dict[start].append(end)
            if end in relatedto_dict:
                relatedto_dict[end].append(start)

        capableof_dict = {concept: [] for concept in batch_concepts}
        for start, end in capable_results:
            if end in capableof_dict:
                capableof_dict[end].append(start)

        notcapableof_dict = {concept: [] for concept in batch_concepts}
        for start, end in not_capable_results:
            if end in notcapableof_dict:
                notcapableof_dict[end].append(start)

        # Concatenate lists into strings with a delimiter
        for concept in relatedto_dict:
            relatedto_dict[concept] = delimiter.join(relatedto_dict[concept])
        for concept in capableof_dict:
            capableof_dict[concept] = delimiter.join(capableof_dict[concept])
        for concept in notcapableof_dict:
            notcapableof_dict[concept] = delimiter.join(notcapableof_dict[concept])

        return relatedto_dict, capableof_dict, notcapableof_dict

    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Initialize new columns
    concept_df['related_to_with_synonyms'] = ''
    concept_df['capable_of_with_synonyms'] = ''
    concept_df['not_capable_of_with_synonyms'] = ''

    # Iterate through each row in the DataFrame
    for index, row in concept_df.iterrows():
        # Flatten the list of strings from the extended_synonyms column
        batch_concepts = [row['concept']] + row['extended_synonyms']

        if batch_concepts:  # Check if the list is not empty
            relatedto_dict, capableof_dict, notcapableof_dict = fetch_related_and_capable_for_batch(batch_concepts,
                                                                                                    cursor)

            # Assign results back to the DataFrame
            first_concept = batch_concepts[0]  # Use the first concept to get values
            concept_df.at[index, 'related_to_with_synonyms'] = relatedto_dict.get(first_concept, '')
            concept_df.at[index, 'capable_of_with_synonyms'] = capableof_dict.get(first_concept, '')
            concept_df.at[index, 'not_capable_of_with_synonyms'] = notcapableof_dict.get(first_concept, '')

    conn.close()
    return concept_df



def add_related_and_capable_to_antonyms_columns(concept_df, db_path="assertions.db", batch_size=500, delimiter=', '):
    """
       Adds related and capable concepts to antonyms columns in the given DataFrame.

       :param concept_df: DataFrame containing concepts with an 'extended_antonyms' column.
       :param db_path: Path to the SQLite database containing ConceptNet relationships (default: "assertions.db").
       :param batch_size: Number of concepts to process in a single batch (default: 500).
       :param delimiter: String used to join related concepts (default: ', ').
       :return: Updated DataFrame with new columns: 'related_to_antonyms', 'capable_of_antonyms', and 'not_capable_of_antonyms'.
       """
    def fetch_related_and_capable_for_batch(batch_concepts, cursor):
        """
        Fetches related and capable concepts from the database for a batch of concepts.

        :param batch_concepts: A list of concept strings to query.
        :param cursor: SQLite database cursor to execute queries.
        :return: Three dictionaries mapping each concept in the batch to its related, capable, and not capable concepts.
        """
        # Fetch related and capable concepts from the database
        placeholders = ', '.join(['?'] * len(batch_concepts))
        
        # Query for RelatedTo (reflexive)
        related_query = f"""
            SELECT start, end
            FROM conceptnet
            WHERE (start IN ({placeholders}) OR end IN ({placeholders}))
            AND relation = '/r/RelatedTo'
            AND start LIKE '/c/en/%' AND end LIKE '/c/en/%'
        """
        cursor.execute(related_query, batch_concepts * 2)  # Duplicate list for both start and end placeholders
        related_results = cursor.fetchall()
        print(f"Related results for {batch_concepts}: {related_results}")  # Debug print

        # Query for CapableOf (non-reflexive, concept as end node)
        capable_query = f"""
            SELECT start, end
            FROM conceptnet
            WHERE end IN ({placeholders})
            AND relation = '/r/CapableOf'
            AND start LIKE '/c/en/%' AND end LIKE '/c/en/%'
        """
        cursor.execute(capable_query, batch_concepts)
        capable_results = cursor.fetchall()
        print(f"Capable results for {batch_concepts}: {capable_results}")  # Debug print

        # Query for NotCapableOf (non-reflexive, concept as end node)
        not_capable_query = f"""
            SELECT start, end
            FROM conceptnet
            WHERE end IN ({placeholders})
            AND relation = '/r/NotCapableOf'
            AND start LIKE '/c/en/%' AND end LIKE '/c/en/%'
        """
        cursor.execute(not_capable_query, batch_concepts)
        not_capable_results = cursor.fetchall()
        print(f"Not capable results for {batch_concepts}: {not_capable_results}")  # Debug print

        # Organize results into strings for efficient lookup
        relatedto_dict = {concept: [] for concept in batch_concepts}
        for start, end in related_results:
            if start in relatedto_dict:
                relatedto_dict[start].append(end)
            if end in relatedto_dict:
                relatedto_dict[end].append(start)
        
        capableof_dict = {concept: [] for concept in batch_concepts}
        for start, end in capable_results:
            if end in capableof_dict:
                capableof_dict[end].append(start)

        notcapableof_dict = {concept: [] for concept in batch_concepts}
        for start, end in not_capable_results:
            if end in notcapableof_dict:
                notcapableof_dict[end].append(start)

        # Concatenate lists into strings with a delimiter
        for concept in relatedto_dict:
            relatedto_dict[concept] = delimiter.join(relatedto_dict[concept])
        for concept in capableof_dict:
            capableof_dict[concept] = delimiter.join(capableof_dict[concept])
        for concept in notcapableof_dict:
            notcapableof_dict[concept] = delimiter.join(notcapableof_dict[concept])

        return relatedto_dict, capableof_dict, notcapableof_dict

    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Initialize new columns
    concept_df['related_to_antonyms'] = ''
    concept_df['capable_of_antonyms'] = ''
    concept_df['not_capable_of_antonyms'] = ''

    # Iterate through each row in the DataFrame
    for index, row in concept_df.iterrows():
        # Flatten the list of strings from the extended_antonyms column
        batch_concepts = row['extended_antonyms']
        
        if batch_concepts:  # Check if the list is not empty
            relatedto_dict, capableof_dict, notcapableof_dict = fetch_related_and_capable_for_batch(batch_concepts, cursor)

            # Assign results back to the DataFrame
            first_concept = batch_concepts[0]  # Use the first concept to get values
            concept_df.at[index, 'related_to_antonyms'] = relatedto_dict.get(first_concept, '')
            concept_df.at[index, 'capable_of_antonyms'] = capableof_dict.get(first_concept, '')
            concept_df.at[index, 'not_capable_of_antonyms'] = notcapableof_dict.get(first_concept, '')

    conn.close()
    return concept_df
