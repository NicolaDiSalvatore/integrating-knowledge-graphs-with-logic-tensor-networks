import pandas as pd
from itertools import combinations
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.spatial.distance import cosine



def load_numberbatch(file_path):
    """
    Load the Numberbatch embeddings
    :param file_path:
    :return: dictionary containing the english Numberbatch embeddings
    """
    embeddings = joblib.load(file_path)

    english_embeddings = {key: value for key, value in embeddings.items() if key.startswith('/c/en/')}
    return english_embeddings




def get_concept_name(concept):
    """Extracts concept name from ConceptNet URI or label."""
    return concept.split('/')[-1]


def generate_equivalence_fol_axioms_for_predicates(df, embeddings_dict, threshold=0.7):
    """
    Generate first-order logic (FOL) equivalence axioms for predicates based on cosine similarity
    between their embeddings.

    :param df (pd.DataFrame): DataFrame containing predicate concepts and their extended synonyms.
    :param embeddings_dict (dict): Dictionary mapping each concept to its embedding.
    :param threshold (float): Minimum cosine similarity to consider two predicates equivalent.
    :return: List[str]: A list of FOL equivalence statements between predicates.
    """
    equivalent_property_fol_statements = []
    generated_statements = set()

    for _, row in df.iterrows():
        subject = get_concept_name(row['concept'])

        if row['concept'] not in embeddings_dict:
            continue
        subject_embedding = embeddings_dict[row['concept']].reshape(1, -1)

        print(f"Processing subject: {subject}")

        valid_synonyms = []
        for synonym in row['extended_synonyms']:
            if synonym in df['concept'].values and synonym != row['concept'] and synonym in embeddings_dict:

                synonym_embedding = embeddings_dict[synonym].reshape(1, -1)
                similarity = cosine_similarity(subject_embedding, synonym_embedding)[0, 0]
                if similarity >= threshold:
                    valid_synonyms.append(synonym)

        for equivalent_property in valid_synonyms:
            equivalent_property = get_concept_name(equivalent_property)
            
            # Create pairs in both directions to avoid duplicate statements
            statement_pairs = [
                (subject, equivalent_property),
                (equivalent_property, subject)
            ]
            
            # Only add if neither direction has been generated
            if statement_pairs[0] not in generated_statements and statement_pairs[1] not in generated_statements:
                equivalent_property_fol_statements.append(
                    f"∀x ∀y ({subject}(x, y) ↔ {equivalent_property}(x, y))"
                )
                
                # Add both directions to generated statements to prevent duplication
                generated_statements.add(statement_pairs[0])
                generated_statements.add(statement_pairs[1])

    return equivalent_property_fol_statements




def generate_equivalence_fol_axioms_for_objects_and_attributes(df, embeddings_dict, threshold=0.9):
    """
    Generate first-order logic (FOL) equivalence axioms for objects and attributes based on cosine similarity
    between their embeddings.

    :param df (pd.DataFrame): DataFrame containing predicate concepts and their extended synonyms.
    :param embeddings_dict (dict): Dictionary mapping each concept to its embedding.
    :param threshold (float): Minimum cosine similarity to consider two predicates equivalent.
    :return: List[str]: A list of FOL equivalence statements between predicates.
    """
    equivalent_property_fol_statements = []
    generated_statements = set()

    for _, row in df.iterrows():
        subject = get_concept_name(row['concept'])
        synonyms = row['extended_synonyms']
        hypernyms = row['Hypernyms']

        if row['concept'] not in embeddings_dict:
            continue
        subject_embedding = embeddings_dict[row['concept']].reshape(1, -1)

        print(f"Processing subject: {subject}")
        if row['concept'] not in embeddings_dict:
            print(f"Embedding for subject '{subject}' not found.")
        
        # Filter synonyms that are present in the 'concept' column, not in hypernyms, and have high similarity
        valid_synonyms = []
        for synonym in synonyms:

            if (synonym in df['concept'].values) and (synonym != row['concept']) and (synonym not in hypernyms):
                if synonym in embeddings_dict:
                    synonym_embedding = embeddings_dict[synonym].reshape(1, -1)
                    similarity = cosine_similarity(subject_embedding, synonym_embedding)[0, 0]
                    if similarity >= threshold:
                        valid_synonyms.append(synonym)
        
        for equivalent_property in valid_synonyms:
            equivalent_property = get_concept_name(equivalent_property)
            
            # Check if the statement has already been generated (in both possible directions)
            statement_pairs = [
                (subject, equivalent_property),
                (equivalent_property, subject)
            ]
            
            if statement_pairs[0] not in generated_statements and statement_pairs[1] not in generated_statements:
                equivalent_property_fol_statements.append(
                    f"∀x ({subject}(x) ↔ {equivalent_property}(x))"
                )
                
                # Add to generated statements in both directions to prevent duplicates
                generated_statements.add(statement_pairs[0])
                generated_statements.add(statement_pairs[1])

    return equivalent_property_fol_statements




def generate_negative_axioms(df, embeddings_dict, similarity_threshold=0):
    """
    Generate mutually exclusive (negative) first-order logic axioms between pairs of concepts.

    Two concepts are considered mutually exclusive if:
    - Neither is a hypernym of the other.
    - Their embedding similarity is below the specified threshold.

    :param df (pd.DataFrame): A DataFrame with columns 'concept' and 'Hypernyms'.
    :param embeddings_dict (dict): Dictionary mapping each concept to its embedding.
    :param similarity_threshold (float): Maximum cosine similarity allowed for negative axioms.
    :return: pd.DataFrame: A DataFrame containing unique mutually exclusive axioms as FOL statements.
    """

    unique_axioms_set = set()

    # Iterate over each unique pair of concepts
    for i in range(len(df)):
        class_name1 = df.loc[i, 'concept']
        hypernyms1 = set(df.loc[i, 'Hypernyms'])

        # Ensure class_name1 has an embedding
        if class_name1 not in embeddings_dict:
            continue

        for j in range(i + 1, len(df)):
            class_name2 = df.loc[j, 'concept']
            hypernyms2 = set(df.loc[j, 'Hypernyms'])

            # Ensure class_name2 has an embedding
            if class_name2 not in embeddings_dict:
                continue

            # Calculate cosine similarity between embeddings
            embedding1 = embeddings_dict[class_name1]
            embedding2 = embeddings_dict[class_name2]
            similarity = 1 - cosine(embedding1, embedding2)

            # Check if class_name1 is not in hypernyms of class_name2 and vice versa
            # and if the similarity is below the threshold add the FOL statement to the set to ensure uniqueness
            if class_name1 not in hypernyms2 and class_name2 not in hypernyms1 and similarity < similarity_threshold:
                fol_statement = f"∀x (¬{class_name1}(x) ∨ ¬{class_name2}(x))"


                unique_axioms_set.add(fol_statement)

    # Convert the unique axioms set to a DataFrame with a single column
    result_df = pd.DataFrame({'unique_axioms': list(unique_axioms_set)})

    return result_df



def generate_negative_axioms_with_objects_and_attributes(df, embeddings_dict, similarity_threshold=0):
    """
    Generate mutually exclusive (negative) first-order logic axioms between objects and attributes.

    Two concepts are considered mutually exclusive if:
    - Neither is a hypernym of the other.
    - Their embedding similarity is below the specified threshold.

    :param df (pd.DataFrame): A DataFrame with columns 'concept' and 'Hypernyms'.
    :param embeddings_dict (dict): Dictionary mapping each concept to its embedding.
    :param similarity_threshold (float): Maximum cosine similarity allowed for negative axioms.
    :return: pd.DataFrame: A DataFrame containing unique mutually exclusive axioms as FOL statements.
    """

    unique_axioms_set = set()

    for i in range(len(df)):
        class_name1 = df.loc[i, 'concept']
        hypernyms1 = set(df.loc[i, 'Hypernyms'])

        if class_name1 not in embeddings_dict:
            continue

        for j in range(i + 1, len(df)):
            class_name2 = df.loc[j, 'concept']
            hypernyms2 = set(df.loc[j, 'Hypernyms'])

            if class_name2 not in embeddings_dict:
                continue

            embedding1 = embeddings_dict[class_name1]
            embedding2 = embeddings_dict[class_name2]
            similarity = 1 - cosine(embedding1, embedding2)

            # Check if class_name1 is not in hypernyms of class_name2 and vice versa
            # and if the similarity is below the threshold add the FOL statement to the set to ensure uniqueness
            if class_name1 not in hypernyms2 and class_name2 not in hypernyms1 and similarity < similarity_threshold:
                fol_statement = f"∀x (¬{class_name1}(x) ∨ ¬{class_name2}(x))"

                unique_axioms_set.add(fol_statement)

    result_df = pd.DataFrame({'unique_axioms': list(unique_axioms_set)})

    return result_df



def generate_negative_axioms_with_predicates_antonyms(df, embeddings_dict, similarity_threshold=1):
    """
    Generate mutually_exclusive (negative) FOL axioms based on antonym relationships between concepts.

    Two concepts are considered mutually exclusive if:
    - They are linked by the antonym relationship in ConceptNet.
    - Their embedding similarity is below the specified threshold.

    :param df (pd.DataFrame): DataFrame containing at least the columns 'concept' and 'extended_antonyms'.
    :param embeddings_dict (dict): Dictionary mapping each concept to its embedding.
    :param similarity_threshold (float): Maximum cosine similarity allowed for negative axioms.
    :return: DataFrame with one column 'negative_fol_axioms' containing unique FOL axioms.
    """

    unique_axioms_set = set()

    for i in range(len(df)):
        class_name1 = df.loc[i, 'concept']
        antonyms1 = set(df.loc[i, 'extended_antonyms'])

        if class_name1 not in embeddings_dict:
            continue

        for j in range(i + 1, len(df)):
            class_name2 = df.loc[j, 'concept']
            antonyms2 = set(df.loc[j, 'extended_antonyms'])

            if class_name2 not in embeddings_dict:
                continue

            if class_name2 in antonyms1 or class_name1 in antonyms2:

                embedding1 = embeddings_dict[class_name1]
                embedding2 = embeddings_dict[class_name2]
                similarity = 1 - cosine(embedding1, embedding2)

                # Generate the axiom only if the similarity is below the threshold
                if similarity < similarity_threshold:
                    fol_statement = f"∀x,y (¬{class_name1}(x,y) ∨ ¬{class_name2}(x,y))"

                    # Add the FOL statement to the set to ensure uniqueness
                    unique_axioms_set.add(fol_statement)

    # Convert the unique axioms set to a DataFrame with a single column
    result_df = pd.DataFrame({'negative_fol_axioms': list(unique_axioms_set)})

    return result_df


def generate_negative_axioms_with_predicates(df, embeddings_dict, similarity_threshold=0):
    """
    Generate mutual exclusivity (negative) FOL axioms between predicates with low similarity.

    :param df: DataFrame with a column 'concept' representing predicates.
    :param embeddings_dict: Dictionary of embeddings with keys matching the 'concept' column.
    :param similarity_threshold: Cosine similarity threshold below which axioms are generated.
    :return: DataFrame containing unique negative FOL axioms.
    """

    unique_axioms_set = set()

    for i in range(len(df)):
        class_name1 = df.loc[i, 'concept']

        if class_name1 not in embeddings_dict:
            continue

        for j in range(i + 1, len(df)):
            class_name2 = df.loc[j, 'concept']

            if class_name2 not in embeddings_dict:
                continue

            embedding1 = embeddings_dict[class_name1]
            embedding2 = embeddings_dict[class_name2]
            similarity = 1 - cosine(embedding1, embedding2)

            if similarity < similarity_threshold:
                fol_statement = f"∀x,y (¬{class_name1}(x,y) ∨ ¬{class_name2}(x,y))"

                unique_axioms_set.add(fol_statement)

    result_df = pd.DataFrame({'negative_fol_axioms': list(unique_axioms_set)})

    return result_df



def generate_hypernyms_fol_axioms(df, admissible_hypernyms, left_side_axioms_column = 'concept', right_side_axioms_column = 'Hypernyms'):
    """
    Generates a DataFrame with ontological FOL axioms for each concept based on the 'concept' and 'Hypernyms' columns.

    :param df (pd.DataFrame): DataFrame with 'concept' (str) and 'Hypernyms' (list of str) columns.
    :param admissible_hypernyms (list): List of hypernyms to include in the FOL implications.
    :return: DataFrame with columns 'concept' and 'FOL_axioms_list', where each row has a concept and a list of FOL axioms.
    """

    concept_axioms = {}

    for _, row in df.iterrows():
        concept = row[left_side_axioms_column]
        hypernyms = row[right_side_axioms_column]

        axioms = []

        # Generate FOL implications for each admissible hypernym
        # if isinstance(hypernyms, list):
        for hypernym in hypernyms:
            if hypernym in admissible_hypernyms:
                axiom = f"{concept}(z) → {hypernym}(z)"
                axioms.append(axiom)

        # Assign the list of axioms to the concept in the dictionary
        concept_axioms[concept] = axioms

    result_df = pd.DataFrame(list(concept_axioms.items()), columns=[left_side_axioms_column, 'FOL_axioms_list'])

    return result_df


def generate_positive_domain_fol_axioms(df, column_name, rhs_predicates_counts=4):
    """
    Generates a DataFrame with positive domain FOL axioms based on 'concept' and the given column_name column.

    :param df (pd.DataFrame): DataFrame with 'concept' and the specified column containing hypernyms.
    :param column_name (str): The column name with hypernym lists or comma-separated strings.
    :param rhs_predicates_counts (int): Number of hypernyms to use in the axiom.
    :return: pd.DataFrame: DataFrame with 'concept', 'has_axiom', and 'axiom' columns.
    """
    concepts = []
    has_axiom_flags = []
    axioms = []

    for _, row in df.iterrows():
        concept = row['concept']
        hypernyms = row[column_name]

        if isinstance(hypernyms, list):
            hypernyms_list = hypernyms[:rhs_predicates_counts]
        elif isinstance(hypernyms, str):
            hypernyms_list = hypernyms.split(", ")[:rhs_predicates_counts]
        else:
            hypernyms_list = []

        if hypernyms_list:
            axiom = f"{concept}(x, y) → " + " ∨ ".join([f"{hyp}(x)" for hyp in hypernyms_list])
            has_axiom = True
        else:
            axiom = None
            has_axiom = False

        concepts.append(concept)
        has_axiom_flags.append(has_axiom)
        axioms.append(axiom)

    result_df = pd.DataFrame({
        'concept': concepts,
        'has_axiom': has_axiom_flags,
        'axiom': axioms
    })

    return result_df


def generate_positive_range_fol_axioms(df, column_name, rhs_predicates_counts=4):
    """
    Generates a DataFrame with positive range FOL axioms based on 'concept' and the given column_name column.

    :param df (pd.DataFrame): DataFrame with 'concept' and the specified column containing hypernyms.
    :param column_name (str): The column name with hypernym lists or comma-separated strings.
    :param rhs_predicates_counts (int): Number of hypernyms to use in the axiom.
    :return: pd.DataFrame: DataFrame with 'concept', 'has_axiom', and 'axiom' columns.
    """
    concepts = []
    has_axiom_flags = []
    axioms = []

    for _, row in df.iterrows():
        concept = row['concept']
        hypernyms = row[column_name]

        if isinstance(hypernyms, list):
            hypernyms_list = hypernyms[:rhs_predicates_counts]
        elif isinstance(hypernyms, str):
            hypernyms_list = hypernyms.split(", ")[:rhs_predicates_counts]
        else:
            hypernyms_list = []

        if hypernyms_list:
            axiom = f"{concept}(x, y) → " + " ∨ ".join([f"{hyp}(y)" for hyp in hypernyms_list])
            has_axiom = True
        else:
            axiom = None
            has_axiom = False

        concepts.append(concept)
        has_axiom_flags.append(has_axiom)
        axioms.append(axiom)

    result_df = pd.DataFrame({
        'concept': concepts,
        'has_axiom': has_axiom_flags,
        'axiom': axioms
    })

    return result_df


def generate_negative_domain_fol_axioms(df, column_name, rhs_predicates_counts=4):
    """
    Generates a DataFrame with negative domain FOL axioms based on 'concept' and the given column_name column.

    :param df (pd.DataFrame): DataFrame with 'concept' and the specified column containing hypernyms.
    :param column_name (str): The column name with hypernym lists or comma-separated strings.
    :param rhs_predicates_counts (int): Number of hypernyms to use in the axiom.
    :return: pd.DataFrame: DataFrame with 'concept', 'has_axiom', and 'axiom' columns.
    """
    concepts = []
    has_axiom_flags = []
    axioms = []

    for _, row in df.iterrows():
        concept = row['concept']
        hypernyms = row[column_name]

        if isinstance(hypernyms, list):
            hypernyms_list = hypernyms[:rhs_predicates_counts]
        elif isinstance(hypernyms, str):
            hypernyms_list = hypernyms.split(", ")[:rhs_predicates_counts]
        else:
            hypernyms_list = []

        if hypernyms_list:
            axiom = f"{concept}(x, y) → " + " ∧ ".join([f"¬ {hyp}(x)" for hyp in hypernyms_list])
            has_axiom = True
        else:
            axiom = None
            has_axiom = False

        concepts.append(concept)
        has_axiom_flags.append(has_axiom)
        axioms.append(axiom)

    result_df = pd.DataFrame({
        'concept': concepts,
        'has_axiom': has_axiom_flags,
        'axiom': axioms
    })

    return result_df



def generate_negative_range_fol_axioms(df, column_name, rhs_predicates_counts=4):
    """
    Generates a DataFrame with negative range FOL axioms based on 'concept' and the given column_name column,
    handling both list and string types in column_name.

    :param df (pd.DataFrame): DataFrame with 'concept' and the specified column containing hypernyms.
    :param column_name (str): The column name with hypernym lists or comma-separated strings.
    :param rhs_predicates_counts (int): Number of hypernyms to use in the axiom.
    :return: pd.DataFrame: DataFrame with 'concept', 'has_axiom', and 'axiom' columns.
    """
    concepts = []
    has_axiom_flags = []
    axioms = []

    for _, row in df.iterrows():
        concept = row['concept']
        hypernyms = row[column_name]

        if isinstance(hypernyms, list):
            hypernyms_list = hypernyms[:rhs_predicates_counts]
        elif isinstance(hypernyms, str):
            hypernyms_list = hypernyms.split(", ")[:rhs_predicates_counts]
        else:
            hypernyms_list = []

        if hypernyms_list:
            axiom = f"{concept}(x, y) → " + " ∧ ".join([f"¬ {hyp}(y)" for hyp in hypernyms_list])
            has_axiom = True
        else:
            axiom = None
            has_axiom = False

        concepts.append(concept)
        has_axiom_flags.append(has_axiom)
        axioms.append(axiom)

    result_df = pd.DataFrame({
        'concept': concepts,
        'has_axiom': has_axiom_flags,
        'axiom': axioms
    })

    return result_df



def generate_hypernyms_fol_axioms_extended(df, axiom_list, axioms_with_hypernyms_column_name, admissible_hypernyms, left_side_axioms_column='concept', right_side_axioms_column='Hypernyms'):
    """
    Generates a DataFrame with ontological FOL axioms for each concept based on the given columns. Also includes
    a list of axioms from the input axiom_list that contain at least one hypernym or the concept itself,
    along with a count of the matching axioms.

    :param df (pd.DataFrame): DataFrame with columns for concepts and their hypernyms.
    :param axiom_list (list): List of FOL axioms (strings) to search for hypernyms or concepts.
    :param left_side_axioms_column (str): Column name for the concepts.
    :param right_side_axioms_column (str): Column name for lists of hypernyms.
    :param admissible_hypernyms (list): List of hypernyms to include in the FOL implications.
    :return pd.DataFrame: DataFrame with columns 'concept', 'FOL_axioms_list', 'axioms_with_hypernyms', and 'matching_axioms_count',
                      where each row has a concept, a list of FOL axioms, a list of matching axioms
                      from the input axiom_list, and the count of matching axioms.
    """
    if admissible_hypernyms is None:
        admissible_hypernyms = []

    concepts = []
    axioms_list = []
    axioms_with_hypernyms = []
    matching_axioms_count = []

    for _, row in df.iterrows():
        concept = row[left_side_axioms_column]
        hypernyms = row[right_side_axioms_column]

        axioms = []
        matching_axioms = []

        if isinstance(hypernyms, list):
            for hypernym in hypernyms:
                if hypernym in admissible_hypernyms:
                    axiom = f"{concept}(z) → {hypernym}(z)"
                    axioms.append(axiom)

            for axiom_str in axiom_list:
                if any(get_concept_name(hyp) in axiom_str for hyp in hypernyms) or get_concept_name(concept) in axiom_str:
                    matching_axioms.append(axiom_str)

        concepts.append(concept)
        axioms_list.append(axioms)
        axioms_with_hypernyms.append(matching_axioms)
        matching_axioms_count.append(len(matching_axioms))

    result_df = pd.DataFrame({
        left_side_axioms_column: concepts,
        'FOL_ontological_axioms': axioms_list,
        axioms_with_hypernyms_column_name: axioms_with_hypernyms,
        'matching_' + axioms_with_hypernyms_column_name + '_count': matching_axioms_count
    })

    df_merged = df.merge(result_df, left_on=left_side_axioms_column, right_on=left_side_axioms_column, how='left')

    return df_merged


def check_negative_fol_axioms_for_predicates(df, axiom_list, axioms_with_concepts_column_name,left_side_axioms_column='concept'):
    """
    Generates a DataFrame where each concept is associated with matching axioms from
    the input axiom_list that contain the concept itself, along with a count of the matching axioms.

    :param df (pd.DataFrame): DataFrame with a column for concepts.
    :param axiom_list (list): List of FOL axioms (strings) to search for concepts.
    :param  left_side_axioms_column (str): Column name for the concepts.
    :return pd.DataFrame: DataFrame with columns 'concept', 'axioms_with_concepts', and 'matching_axioms_count',
                      where each row has a concept, a list of matching axioms from the input axiom_list,
                      and the count of matching axioms.
    """

    concepts = []
    axioms_with_concepts = []
    matching_axioms_count = []

    for _, row in df.iterrows():
        concept = row[left_side_axioms_column]

        matching_axioms = []

        for axiom_str in axiom_list:
            if f"{concept}(x,y)" in axiom_str:
                matching_axioms.append(axiom_str)


        concepts.append(concept)
        axioms_with_concepts.append(matching_axioms)
        matching_axioms_count.append(len(matching_axioms))


    result_df = pd.DataFrame({
        left_side_axioms_column: concepts,
        axioms_with_concepts_column_name: axioms_with_concepts,
        'matching_' + axioms_with_concepts_column_name + '_count': matching_axioms_count
    })


    df_merged = df.merge(result_df, on=left_side_axioms_column, how='left')

    return df_merged


def check_equivalence_fol_axioms_for_predicates(df, axiom_list, axioms_with_concepts_column_name, left_side_axioms_column='concept'):
    """
    Generates a DataFrame where each concept is associated with matching equivalence axioms from
    the input axiom_list that contain the concept itself, along with a count of the matching axioms.

    :param df (pd.DataFrame): DataFrame with a column for concepts.
    :param axiom_list (list): List of FOL axioms (strings) to search for equivalences.
    :param axioms_with_concepts_column_name (str): Column name for storing matching axioms.
    :param left_side_axioms_column (str): Column name for the concepts.
    :return pd.DataFrame: DataFrame with columns 'concept', 'axioms_with_concepts', and 'matching_axioms_count',
                      where each row has a concept, a list of matching equivalence axioms from the input axiom_list,
                      and the count of matching axioms.
    """

    concepts = []
    axioms_with_concepts = []
    matching_axioms_count = []

    for _, row in df.iterrows():
        concept = row[left_side_axioms_column]

        matching_axioms = []

        for axiom_str in axiom_list:
            if f"{get_concept_name(concept)}(x, y) ↔" in axiom_str or f"↔ {get_concept_name(concept)}(x, y)" in axiom_str:
                matching_axioms.append(axiom_str)

        concepts.append(concept)
        axioms_with_concepts.append(matching_axioms)
        matching_axioms_count.append(len(matching_axioms))

    result_df = pd.DataFrame({
        left_side_axioms_column: concepts,
        axioms_with_concepts_column_name: axioms_with_concepts,
        'matching_' + axioms_with_concepts_column_name + '_count': matching_axioms_count
    })

    df_merged = df.merge(result_df, on=left_side_axioms_column, how='left')

    return df_merged
