import pandas as pd
import sqlite3
import joblib
from collections import Counter
import numpy as np
import re
import textwrap



# Division of Conceptnet relationships in categories
macro_categories = {
    "Semantic Relationships": [
        '/r/Antonym', '/r/DefinedAs',  '/r/DistinctFrom',
        '/r/EtymologicallyDerivedFrom', '/r/EtymologicallyRelatedTo', '/r/FormOf',
          '/r/MannerOf',
        '/r/RelatedTo', '/r/SimilarTo', '/r/SymbolOf', '/r/Synonym'
    ],
    "Spatial Relationships": ['/r/AtLocation', '/r/LocatedNear'],
    "Functional Relationships": [
        '/r/Desires', '/r/NotDesires', '/r/CapableOf', '/r/NotCapableOf', '/r/Causes', '/r/CausesDesire', '/r/HasFirstSubevent',
        '/r/HasLastSubevent', '/r/HasPrerequisite', '/r/HasSubevent', '/r/MotivatedByGoal',
        '/r/ReceivesAction', '/r/UsedFor', '/r/HasProperty', '/r/NotHasProperty'
    ],
    "Ontological Relationships": [
        '/r/DerivedFrom', '/r/InstanceOf', '/r/IsA', '/r/HasA', '/r/PartOf'
    ],
    "Creation or Origin Relationships": ['/r/CreatedBy', '/r/MadeOf'],
    "External Information Links": [
        '/r/ExternalURL', '/r/dbpedia/capital', '/r/dbpedia/field', '/r/dbpedia/genre',
        '/r/dbpedia/genus', '/r/dbpedia/influencedBy', '/r/dbpedia/knownFor',
        '/r/dbpedia/language', '/r/dbpedia/leader', '/r/dbpedia/occupation', '/r/dbpedia/product'
    ]
}


def load_joblib_file(file_path):
    try:
        data = joblib.load(file_path)
        print(f"File loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def standardize_most_similar_concept(df, column_name = 'most_similar_concept'):
    """
    Standardize the 'column_name' column in the DataFrame by ensuring that each term
    is formatted as /c/en/{term}. Handles NoneType values.

    :param df (pd.DataFrame): DataFrame with a 'column_name' column containing terms to standardize.
    :param column_name: name of the column in df containing terms to standardize.
    :return pd.DataFrame: DataFrame with the standardized 'column_name' column.
    """
    def format_concept(concept):
        if concept is None:
            return None

        if isinstance(concept, list):
            terms = concept
        else:
            terms = concept.split(', ')
            
        # Format each term as '/c/en/{term}' if it doesn't start with '/c/en/'
        formatted_terms = [term if term.startswith('/c/en/') else f'/c/en/{term}' for term in terms]

        return ', '.join(formatted_terms)

    df[column_name] = df[column_name].apply(format_concept)
    
    return df


def fetch_relationship_counts_optimized(db_path, concepts_df):
    """
    Fetches relationship counts from the ConceptNet database for the most similar concepts
    in the concepts_df DataFrame.

    :param db_path (str): Path to the SQLite database.
    :param concepts_df (pd.DataFrame): DataFrame containing a 'most_similar_concept' column.
    :return pd.DataFrame: A DataFrame where each row corresponds to a concept and each column corresponds to a relationship type.
    """
    conn = sqlite3.connect(db_path)

    concepts = concepts_df['most_similar_concept'].explode().unique()  # Handle lists and strings
    concepts = [concept for concept in concepts if concept]  # Remove any None or empty strings

    # Use SQL parameterization to prevent injection and ensure proper formatting
    query = f"""
    SELECT 
        CASE 
            WHEN start IN ({','.join(['?'] * len(concepts))}) THEN start 
            ELSE end 
        END AS concept,
        relation,
        COUNT(*) as count
    FROM conceptnet
    WHERE start IN ({','.join(['?'] * len(concepts))}) 
       OR end IN ({','.join(['?'] * len(concepts))})
    GROUP BY concept, relation
    """

    params = concepts * 3  # We need to provide the concepts list three times (start, end, and GROUP BY)
    relationship_df = pd.read_sql_query(query, conn, params=params)

    relationship_df = relationship_df.pivot_table(index='concept', columns='relation', values='count',
                                                  fill_value=0).reset_index()

    conn.close()

    return relationship_df


def get_total_counts(relationship_df):
    """
    Calculate total counts for each relationship type across the entire dataset,
    eliminating rows with duplicate concepts, and transpose the result.

    :param  (pd.DataFrame): DataFrame containing relationship counts with concepts as rows.
    :return pd.DataFrame: A transposed DataFrame with total counts for each relationship type.
    """
    unique_df = relationship_df.drop_duplicates(subset=['concept'])
    counts_df = unique_df.drop(columns=['concept'])
    counts_df = counts_df.apply(pd.to_numeric, errors='coerce')

    # Calculate total counts for each relationship type across the entire dataset
    total_counts = counts_df.sum(axis=0)


    # Create a DataFrame for total counts
    total_counts_df = pd.DataFrame(total_counts).T
    total_counts_df = total_counts_df.transpose()

    total_counts_df.columns = ['Total Count']

    return total_counts_df



def categorize_relationships(df):
    """
    Categorizes relationship types into macro-categories and sums their counts.

    :param df (pd.DataFrame): A DataFrame where the index contains relationship types and the 'Total Count' column holds their respective counts.
    :return pd.DataFrame: A DataFrame where the index contains macro-categories and the 'Count' column represents the sum of counts for relationships in each category.
    """
    macrocategory_counts = {category: 0 for category in macro_categories.keys()}

    for relationship, count in df['Total Count'].items():
        for category, relationships in macro_categories.items():
            if relationship in relationships:
                macrocategory_counts[category] += count

    # Convert the macrocategory counts to a DataFrame with macrocategories as row names (index)
    result_df = pd.DataFrame.from_dict(macrocategory_counts, orient='index', columns=['Count'])

    return result_df


def safe_convert_to_int(val):
    if pd.isna(val):
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None




def count_strings_in_df_column(df, column_name):
    """
    Count each string in the 'column_name' column where values are list of strings.
    :param df: the dataframe with the 'column_name' column
    :param column_name:name of the column containing the strings to count
    :return: a dataframe with the 'column_name' column ad the 'count' column
    """
    all_strings = []

    for item in df[column_name]:
        if isinstance(item, list):
            all_strings.extend(item)
        else:
            all_strings.append(item)

    string_counts = Counter(all_strings)

    count_df = pd.DataFrame(string_counts.items(), columns=[column_name, 'count'])
    count_df = count_df.sort_values(by='count', ascending=False).reset_index(drop=True)

    return count_df



def count_strings_with_frequency(df, column_name, frequency_column='Frequency'):
    """
    Count each string in the 'column_name' column where values are list of strings with frequency.
    :param df: the dataframe with the 'column_name' column
    :param column_name:name of the column containing the strings to count
    :return: a dataframe with the 'column_name' column ad the 'count' column
    """
    weighted_strings = []

    for item, frequency in zip(df[column_name], df[frequency_column]):
        if isinstance(item, list):
            weighted_strings.extend(item * frequency)
        else:
            weighted_strings.extend([item] * frequency)

    string_counts = Counter(weighted_strings)

    count_df = pd.DataFrame(string_counts.items(), columns=[column_name, 'count_in_Visual_Genome'])
    count_df = count_df.sort_values(by='count_in_Visual_Genome', ascending=False).reset_index(drop=True)

    return count_df



def calculate_alignment_metrics(df, top_k=5):
    metrics = {
        'mean_similarity': [],
        # 'std_similarity': [],
        'ratio_to_mean': [],
        'difference_to_mean': [],
        'similarity_dropoff': [],
    }



    for similarities in df['top_5_similarities']:
        similarities = sorted(similarities, reverse=True)

        # Mean Similarity
        mean_similarity = np.mean(similarities)
        metrics['mean_similarity'].append(mean_similarity)

        # Ratio between the top similarity and the mean of the others
        ratio_to_mean = similarities[0] / np.mean(similarities[1:])
        metrics['ratio_to_mean'].append(ratio_to_mean)

        # Difference between the top similarity and the botton similarity
        similarity_dropoff = similarities[0] - similarities[-1] if len(similarities) > 1 else 0
        metrics['similarity_dropoff'].append(similarity_dropoff)

        # Difference  between top similarity and the mean of the others
        difference_to_mean = similarities[0] - np.mean(similarities[1:])
        metrics['difference_to_mean'].append(difference_to_mean)

        # Standard Deviation of Similarities
        # std_similarity = np.std(similarities)
        # metrics['std_similarity'].append(std_similarity)

    metrics_df = pd.DataFrame(metrics)

    df = pd.concat([df, metrics_df], axis=1)
    
    return df


def calculate_mean_std_alignment_metrics(df):
    return pd.DataFrame({
    'Mean': df[['mean_similarity', 'ratio_to_mean', 'similarity_dropoff', 'difference_to_mean']].mean(),
    'Standard Deviation': df[['mean_similarity', 'ratio_to_mean', 'similarity_dropoff', 'difference_to_mean']].std(),
            '25th Percentile': df[['mean_similarity', 'ratio_to_mean', 'similarity_dropoff', 'difference_to_mean']].quantile(0.25),
        '75th Percentile': df[['mean_similarity', 'ratio_to_mean', 'similarity_dropoff', 'difference_to_mean']].quantile(0.75)
    })



def count_predicate_in_lhs(predicate, df):
    if not isinstance(predicate, str) or df.empty:
        return 0  # Ensure the predicate is a string and the DataFrame is not empty

    # Ensure 'axiom' column is treated as a string and drop NaNs
    df = df[df['axiom'].notna()].copy()
    df['axiom'] = df['axiom'].astype(str)

    # Extract LHS using vectorized split
    lhs = df['axiom'].str.split(r'→|->').str[0].str.strip()

    # Remove arguments in parentheses using optimized regex
    lhs_cleaned = lhs.map(lambda x: re.sub(r'\([^)]+\)', '', x))

    # Count occurrences of the predicate in LHS
    return (lhs_cleaned == predicate).sum()

# def count_predicate_in_lhs(predicate, df):
#     # Initialize an empty pandas Series to store predicate counts
#     predicate_counts = pd.Series(dtype=int)
#
#     count = 0
#     if isinstance(predicate, str):  # Ensure the predicate is a string before processing
#         for index, row in df.iterrows():
#             axiom = row['axiom']
#             if isinstance(axiom, str):  # Ensure the axiom is a string
#                 # Split the axiom into LHS and RHS parts
#                 if '→' in axiom:
#                     lhs = axiom.split('→')[0].strip()  # Get LHS part of the axiom
#                 elif '->' in axiom:
#                     lhs = axiom.split('->')[0].strip()
#                 else:
#                     continue
#
#                 # Remove arguments in parentheses from each predicate in the LHS
#                 lhs_cleaned = re.sub(r'\([^)]+\)', '', lhs).strip()
#
#                 # Debug: Print cleaned LHS and compare it with the predicate
#                 # print(f"Row {index} - Original LHS: '{lhs}'")
#                 # print(f"Row {index} - Cleaned LHS: '{lhs_cleaned}'")
#                 # print(f"Comparing: '{lhs_cleaned}' == '{predicate}'")
#
#                 # If the predicate is not in the series, initialize the count to 0
#                 if predicate not in predicate_counts:
#                     predicate_counts[predicate] = 0
#
#                 # Check if we found a match and increment the count
#                 if lhs_cleaned == predicate:
#                     # print(f"Found exact match for '{predicate}' in Row {index}: '{lhs_cleaned}'")
#                     predicate_counts[predicate] += 1
#
#     # Retrieve and return the count for the given predicate
#     return predicate_counts.get(predicate, 0)


def count_negative_axioms_for_df(predicate, negative_fol_axioms):
    count = 0

    if isinstance(predicate, str):  # Ensure the predicate is a string before processing
        for axiom in negative_fol_axioms['negative_fol_axioms']:
            if isinstance(axiom, str):  # Ensure the axiom is a string
                # Check if the axiom has the negation symbol (¬) and contains the predicate in negated form
                negated_predicate_pattern = f"¬{predicate}\([^)]+\)"

                # If the axiom contains a negated predicate and it's in the form of a disjunction (∨)
                if re.search(negated_predicate_pattern, axiom) and '∨' in axiom:
                    count += 1
    return count



def count_equivalence_axioms_for_df(predicate, fol_axioms_df):
    count = 0
    predicate_base = predicate.split('/')[-1]  # Extract the predicate name, e.g., 'climb' from '/c/en/climb'

    if isinstance(predicate_base, str):  # Ensure the predicate base is a string before processing
        for axiom in fol_axioms_df[0]:
            if isinstance(axiom, str):  # Ensure the axiom is a string
                # Pattern to find biconditionals of form "predicate(x, y) ↔ another_predicate(x, y)"
                biconditional_pattern = rf"\b{predicate_base}\(x, y\)\s*↔|\s*↔\s*\b{predicate_base}\(x, y\)"

                # If the axiom contains a biconditional statement with the predicate on either side
                if re.search(biconditional_pattern, axiom):
                    count += 1
    return count


def process_row(subject_axioms, object_axioms, predicate_axiom=None):
    axioms_set = set(subject_axioms + object_axioms)

    if predicate_axiom is not None:
        if isinstance(predicate_axiom, str):
            axioms_set.add(predicate_axiom)
        elif isinstance(predicate_axiom, list):
            axioms_set.update(predicate_axiom)

    return list(axioms_set)



columns = [
        'positive_domain_axioms_in_triple',
        'positive_range_axioms_in_triple',
        'positive_domain_using_capable_of_fol_axioms',
        'negative_domain_using_not_capable_of_fol_axioms',
        'ontological_fol_axioms',
        'negative_fol_axioms',
        'equivalence_fol_axioms',
        'total_axioms'

]

def clean_nan_from_lists(df, columns):
    """
    Removes NaN values from lists in specified DataFrame columns.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the columns.
    columns (list): List of column names to process.

    Returns:
    pd.DataFrame: DataFrame with NaN values removed from lists in the specified columns.
    """
    for column in columns:
        if column in df.columns:
            df[column] = df[column].map(lambda x: [item for item in x if pd.notna(item)] if isinstance(x, list) else x)
    return df






def format_list_for_latex(lst, list_name):
    formatted_items = "\n".join(lst)  # Format each item on a new line
    latex_output = f"\\begin{{lstlisting}}\n{list_name}:\n{formatted_items}\n\\end{{lstlisting}}\n"
    return latex_output

def wrap_labels(labels, max_width):
    return [
        '\n'.join(textwrap.wrap(label, width=max_width, break_long_words=False, break_on_hyphens=False))
        for label in labels
    ]


