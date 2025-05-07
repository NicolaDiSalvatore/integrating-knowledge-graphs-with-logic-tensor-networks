import pandas as pd
import sqlite3
import time
import joblib
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re



#
def count_strings_in_df_column(df, column_name):
    """
    Count the number of occurrences of each string in the specified column of a DataFrame.
    The column is expected to contain lists of strings or a string.

    :param df: A pandas DataFrame containing the data.
    :param column_name: The name of the column in the DataFrame to analyze.
    :return: A DataFrame with two columns: the strings and their counts, sorted by count in descending order.
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
    Count the occurrences of strings in the specified column of a DataFrame, taking into account their frequency.
    The column is expected to contain lists of strings, and the frequency column provides the count for each list.

    :param df: A pandas DataFrame containing the data.
    :param column_name: The name of the column in the DataFrame containing lists of strings.
    :param frequency_column: The name of the column in the DataFrame that contains the frequency of each list.
    :return: A DataFrame with two columns: the strings and their weighted counts, sorted by count in descending order.
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




def calculate_alignment_metrics(df):
    """
    Calculate various metrics to evaluate the alignment of similarities in a DataFrame.

    :param df: A pandas DataFrame containing a column 'top_5_similarities' with lists of similarity scores.
    :return: The original DataFrame concatenated with a new DataFrame containing the calculated metrics.
    """
    metrics = {
        'mean_similarity': [],
        # 'std_similarity': [],
        'ratio_to_mean': [],
        'difference_to_mean': [],
        'similarity_dropoff': []
    }

    for similarities in df['top_5_similarities']:
        similarities = sorted(similarities, reverse=True)

        # 1. Mean Similarity
        mean_similarity = np.mean(similarities)
        metrics['mean_similarity'].append(mean_similarity)

        # 2. Standard Deviation of Similarities
        # std_similarity = np.std(similarities)
        # metrics['std_similarity'].append(std_similarity)

        # 3. ratio between top similarity and the mean of the others
        ratio_to_mean = similarities[0] / np.mean(similarities[1:])
        metrics['ratio_to_mean'].append(ratio_to_mean)

        # 4. Similarity Drop-Off
        dropoff = similarities[0] - similarities[-1] if len(similarities) > 1 else 0
        metrics['similarity_dropoff'].append(dropoff)


    metrics_df = pd.DataFrame(metrics)

    df = pd.concat([df, metrics_df], axis=1)
    
    return df


def calculate_mean_std_alignment_metrics(df):
    """
    Calculate statistical metrics to evaluate the alignment of similarity scores in a DataFrame.
    The metrics include mean, standard deviation, 25th percentile, and 75th percentile for each alignment metric.

    :param df: A pandas DataFrame containing columns with alignment metrics such as 'mean_similarity', 'ratio_to_mean',
               'similarity_dropoff', and 'difference_to_mean'.
    :return: A new DataFrame summarizing the statistical metrics for each alignment metric.
    """
    return pd.DataFrame({
    'Mean': df[['mean_similarity', 'ratio_to_mean', 'similarity_dropoff', 'difference_to_mean']].mean(),
    'Standard Deviation': df[['mean_similarity', 'ratio_to_mean', 'similarity_dropoff', 'difference_to_mean']].std(),
    '25th Percentile': df[['mean_similarity', 'ratio_to_mean', 'similarity_dropoff', 'difference_to_mean']].quantile(0.25),
    '75th Percentile': df[['mean_similarity', 'ratio_to_mean', 'similarity_dropoff', 'difference_to_mean']].quantile(0.75)
    })



def count_predicate_in_lhs(predicate, df):

    """
    Count the occurrences of a given predicate in the left-hand side (LHS) of logical axioms within a DataFrame.

    :param predicate: The predicate to search for, expected to be a string.
    :param df: A pandas DataFrame containing a column 'FOL_axioms' with logical axioms as strings.
    :return: The count of occurrences of the predicate in the LHS of the axioms.
    """
    # Initialize an empty pandas Series to store predicate counts
    predicate_counts = pd.Series(dtype=int)

    count = 0
    if isinstance(predicate, str):  # Ensure the predicate is a string before processing
        for index, row in df.iterrows():
            axiom = row['FOL_axioms']
            if isinstance(axiom, str):  # Ensure the axiom is a string
                # Split the axiom into LHS and RHS parts
                if '→' in axiom:
                    lhs = axiom.split('→')[0].strip()  # Get LHS part of the axiom
                elif '->' in axiom:
                    lhs = axiom.split('->')[0].strip()
                else:
                    continue

                # Remove arguments in parentheses from each predicate in the LHS
                lhs_cleaned = re.sub(r'\([^)]+\)', '', lhs).strip()

                # Debug: Print cleaned LHS and compare it with the predicate
                # print(f"Row {index} - Original LHS: '{lhs}'")
                # print(f"Row {index} - Cleaned LHS: '{lhs_cleaned}'")
                # print(f"Comparing: '{lhs_cleaned}' == '{predicate}'")

                # If the predicate is not in the series, initialize the count to 0
                if predicate not in predicate_counts:
                    predicate_counts[predicate] = 0

                # Check if we found a match and increment the count
                if lhs_cleaned == predicate:
                    # print(f"Found exact match for '{predicate}' in Row {index}: '{lhs_cleaned}'")
                    predicate_counts[predicate] += 1

    # Retrieve and return the count for the given predicate
    return predicate_counts.get(predicate, 0)



def count_negative_axioms_for_df(predicate, negative_fol_axioms):
    """
    Counts the number of negative first-order logic (FOL) axioms that involve
    the given predicate in a DataFrame.

    Parameters:
    predicate (str): The predicate to search for in the negative FOL axioms.
    negative_fol_axioms (pd.DataFrame): A DataFrame containing a column
                                        'negative_fol_axioms' with FOL axioms as strings.

    Returns:
    int: The count of negative axioms where the given predicate appears
         in a negated form (¬predicate) and is part of a disjunction (∨).
    """
    count = 0
    
    if isinstance(predicate, str):
        for axiom in negative_fol_axioms['negative_fol_axioms']:
            if isinstance(axiom, str):
                negated_predicate_pattern = f"¬{predicate}\([^)]+\)"

                if re.search(negated_predicate_pattern, axiom) and '∨' in axiom:
                    count += 1
    return count



def count_equivalence_axioms_for_df(predicate, fol_axioms_df):
    """
    Counts the number of equivalence (biconditional) axioms in a DataFrame
    that involve the given predicate.

    Parameters:
    predicate (str): The predicate whose equivalence axioms need to be counted.
                     It is expected to be in the format '/c/en/predicate'.
    fol_axioms_df (pd.DataFrame): A DataFrame where the first column (index 0)
                                  contains FOL axioms as strings.

    Returns:
    int: The count of axioms where the given predicate appears in a
         biconditional (↔) statement.
    """
    count = 0
    predicate_base = predicate.split('/')[-1]
    
    if isinstance(predicate_base, str):
        for axiom in fol_axioms_df[0]:
            if isinstance(axiom, str):
                biconditional_pattern = rf"\b{predicate_base}\(x, y\)\s*↔|\s*↔\s*\b{predicate_base}\(x, y\)"
                if re.search(biconditional_pattern, axiom):
                    count += 1
    return count

