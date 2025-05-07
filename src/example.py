import joblib
import pandas as pd
import re
# Replace 'your_file.csv' with the path to your CSV file
# example_for_image = pd.read_csv('example_for_image.csv')
# fol_axioms = pd.read_csv('fol_axioms.csv')
# aligned_predicates = joblib.load( "aligned_predicates_extended.joblib")
# predicates_fol_axioms = joblib.load("predicates_fol_axioms2.joblib")
# aligned_predicates = aligned_predicates[aligned_predicates['image_id']<=10]

aligned_predicates_example = joblib.load( "aligned_predicates_example.joblib")

aligned_predicates_example = aligned_predicates_example.rename(columns={
    'positive_domain_axioms_in_triple': 'positive_domain_axioms',
    'positive_range_axioms_in_triple': 'positive_range_axioms',
})

columns = [
    # 'subject_concept',
    # 'object_concept',
    # 'predicate_concept',
    'positive_domain_axioms',
    'positive_range_axioms',
    'positive_domain_using_capable_of_fol_axioms',
    'negative_domain_using_not_capable_of_fol_axioms',
    'ontological_fol_axioms',
    'negative_fol_axioms',
    'equivalence_fol_axioms',
    'total_axioms'
]


objects_and_attributes_hierarchies = joblib.load("objects_and_attributes_hierarchies3.joblib")



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
            df[column] = df[column].apply(lambda x: [item for item in x if pd.notna(item)] if isinstance(x, list) else x)
    return df

fol_axioms_in_images_example = aligned_predicates_example.groupby('image_id')[columns].apply(
    lambda df: df.apply(lambda x: list(set(sum(x.dropna(), []))))  # Drop NaN values before processing
).reset_index()


# fol_axioms_in_images_example = clean_nan_from_lists(fol_axioms_in_images_example, columns)

string_columns = ['subject_concept', 'object_concept', 'predicate_concept']

def get_unique_elements(col_values):
    return col_values.dropna().unique().tolist()

# Group by 'image_id' and apply the function to each column
result = aligned_predicates_example.groupby('image_id').agg(
    {
        'subject_concept': lambda x: get_unique_elements(x),
        'object_concept': lambda x: get_unique_elements(x),
        'predicate_concept': lambda x: get_unique_elements(x)
    }
).reset_index()


fol_axioms_in_images_example = fol_axioms_in_images_example.merge(result, on='image_id', how='left')
# Clean NaN values from the result (if necessary)
fol_axioms_in_images_example = clean_nan_from_lists(fol_axioms_in_images_example, columns)


def filter_ontological_axioms(row):
    # Flatten both subject_concept and object_concept into one set of valid terms
    valid_terms = set(row['subject_concept']) | set(row['object_concept'])  # Union of both lists
    
    # Filter ontological_axiom: Keep strings containing at least one valid term
    return [axiom for axiom in row['ontological_fol_axioms'] if any(term in axiom for term in valid_terms)]

# Apply the filter function row by row
fol_axioms_in_images_example['ontological_fol_axioms_filtered'] = fol_axioms_in_images_example.apply(filter_ontological_axioms, axis=1)

def map_hypernyms_to_objects_per_row(axioms):
    hypernym_to_objects = {}

    # Iterate through each axiom for this row
    for axiom in axioms:
        try:
            # Split the axiom into object and hypernym parts
            left, right = axiom.split(' → ')
            object_concept = left.split('(')[0]  # Extract the object part
            hypernym_concept = right.split('(')[0]  # Extract the hypernym part

            # Add the object to the list of objects for the hypernym
            if hypernym_concept not in hypernym_to_objects:
                hypernym_to_objects[hypernym_concept] = []
            if object_concept not in hypernym_to_objects[hypernym_concept]:
                hypernym_to_objects[hypernym_concept].append(object_concept)
        except ValueError:
            # If splitting fails (invalid axiom format), skip this axiom
            continue

    # Convert all object and hypernym concepts to their full form with '/c/en/'
    final_mapping = {}
    for hypernym, objects in hypernym_to_objects.items():
        final_mapping[f'{hypernym}'] = [f'{obj}' for obj in objects]
    
    return final_mapping

# Function to process the entire dataframe and build a dictionary for each row
def build_row_dictionaries(df, column_name):
    row_mappings = {}
    
    for _, row in df.iterrows():
        # Map hypernyms to objects for each image (row)
        row_mapping = map_hypernyms_to_objects_per_row(row[column_name])
        row_mappings[row['image_id']] = row_mapping
    
    return row_mappings

# # Apply the function to build row dictionaries
hypernyms_objects_dictionaries = build_row_dictionaries(fol_axioms_in_images_example, 'ontological_fol_axioms_filtered')


# def map_hypernyms_in_positive_domain(row, hypernym_dict):
#     image_id = row['image_id']
#     positive_domain = row['positive_domain_axioms']
#     row_hypernym_dict = hypernym_dict.get(image_id, {})
    
#     # List to store the transformed strings
#     transformed_axioms = []
    
#     for axiom in positive_domain:
#         # Replace each hypernym in the axiom with its corresponding objects
#         transformed_axiom = axiom
#         for hypernym, objects in row_hypernym_dict.items():
#             # Join multiple objects with " \lor " if needed
#             object_repr = ' ∨ '.join(objects)
            
#             # Replace all occurrences of the hypernym with the objects
#             transformed_axiom = re.sub(re.escape(hypernym) + r'\((\w+)\)', 
#                                        lambda m: f"{object_repr}({m.group(1)})", 
#                                        transformed_axiom)
        
#         # Append the transformed axiom to the list
#         transformed_axioms.append(transformed_axiom)
    
#     return transformed_axioms



def map_hypernyms_in_positive_domain(row, hypernym_dict, columns_to_map):
    image_id = row['image_id']
    positive_domain = row[columns_to_map]
    row_hypernym_dict = hypernym_dict.get(image_id, {})
    
    # List to store the transformed strings
    transformed_axioms = []
    
    for axiom in positive_domain:
        # Replace each hypernym in the axiom with its corresponding objects
        transformed_axiom = axiom
        
        # For each hypernym in the image's dictionary
        for hypernym, objects in row_hypernym_dict.items():
            # Define a lambda function to process the match and format the object terms with the correct variable
            def format_objects(m):
                return ' ∨ '.join([f"{obj}({m.group(1)})" for obj in objects])
            
            # Use re.sub to replace the hypernym with the formatted object terms
            transformed_axiom = re.sub(re.escape(hypernym) + r'\((\w+)\)', 
                                       format_objects, 
                                       transformed_axiom)
        
        # Append the transformed axiom to the list
        transformed_axioms.append(transformed_axiom)
    
    return transformed_axioms

# Apply the function to each row of the DataFrame
fol_axioms_in_images_example['positive_domain_axioms_mapped'] = fol_axioms_in_images_example.apply(
    lambda row: map_hypernyms_in_positive_domain(row, hypernym_dict = hypernyms_objects_dictionaries, columns_to_map = 'positive_domain_axioms'), axis=1
)

fol_axioms_in_images_example['positive_range_axioms_mapped'] = fol_axioms_in_images_example.apply(
    lambda row: map_hypernyms_in_positive_domain(row, hypernym_dict = hypernyms_objects_dictionaries, columns_to_map = 'positive_range_axioms'), axis=1
)

fol_axioms_in_images_example['positive_domain_using_capable_of_axioms_mapped'] = fol_axioms_in_images_example.apply(
    lambda row: map_hypernyms_in_positive_domain(row, hypernym_dict = hypernyms_objects_dictionaries, columns_to_map = 'positive_domain_using_capable_of_fol_axioms'), axis=1
)

fol_axioms_in_images_example['negative_domain_using_not_capable_of_axioms_mapped'] = fol_axioms_in_images_example.apply(
    lambda row: map_hypernyms_in_positive_domain(row, hypernym_dict = hypernyms_objects_dictionaries, columns_to_map = 'negative_domain_using_not_capable_of_fol_axioms'), axis=1
)


def remove_duplicates_in_implication(axiom):
    # Split the axiom into left and right parts at the implication symbol
    if '→' in axiom:
        left_part, right_part = map(str.strip, axiom.split('→'))
        
        # Split right part by ' ∨ ' to get individual terms
        terms = right_part.split(' ∨ ')
        
        # Remove duplicates by converting terms to a dictionary to preserve order
        unique_terms = list(dict.fromkeys(terms))
        
        # Join the unique terms back with ' ∨ '
        deduplicated_right_part = ' ∨ '.join(unique_terms)
        
        # Reconstruct the axiom with deduplicated right part
        return f"{left_part} → {deduplicated_right_part}"
    
    # If there's no implication symbol, return the axiom as is
    return axiom

# Example usage
# axiom = '/c/en/tire(x, y) → /c/en/building(x) ∨ /c/en/car(x) ∨ /c/en/car(x) ∨ /c/en/van(x) ∨ /c/en/van(x)'
# print(remove_duplicates_in_implication(axiom))


def process_positive_domain_mapped(axioms_list):
    return [remove_duplicates_in_implication(axiom) for axiom in axioms_list]

# Apply the processing function to each row in `positive_domain_mapped`
fol_axioms_in_images_example['positive_domain_axioms_mapped'] = fol_axioms_in_images_example['positive_domain_axioms_mapped'].apply(process_positive_domain_mapped)
fol_axioms_in_images_example['positive_range_axioms_mapped'] = fol_axioms_in_images_example['positive_range_axioms_mapped'].apply(process_positive_domain_mapped)
fol_axioms_in_images_example['positive_domain_using_capable_of_axioms_mapped'] = fol_axioms_in_images_example['positive_domain_using_capable_of_axioms_mapped'].apply(process_positive_domain_mapped)
fol_axioms_in_images_example['negative_domain_using_not_capable_of_axioms_mapped'] = fol_axioms_in_images_example['negative_domain_using_not_capable_of_axioms_mapped'].apply(process_positive_domain_mapped)

def filter_axioms_by_predicates(row, column_name):
    # Extract the list of axioms from the column 'positive_domain_axioms_mapped'
    axioms = row[column_name]
    
    # Extract the list of predicates from the column 'predicate_concept'
    predicates = row['predicate_concept']
    
    # Filter axioms by checking if any predicate is in the axiom string
    filtered_axioms = [axiom for axiom in axioms if any(pred + "(x, y)" in axiom for pred in predicates)]
    
    return filtered_axioms

# Apply the function to each row to create the new column
fol_axioms_in_images_example['positive_domain_axioms_mapped_filtered'] = fol_axioms_in_images_example.apply(
    lambda row: filter_axioms_by_predicates(row, column_name='positive_domain_axioms_mapped'), axis=1
)
fol_axioms_in_images_example['positive_range_axioms_mapped_filtered'] = fol_axioms_in_images_example.apply(
    lambda row: filter_axioms_by_predicates(row, column_name='positive_range_axioms_mapped'), axis=1
)
fol_axioms_in_images_example['positive_domain_using_capable_of_axioms_mapped_filtered'] = fol_axioms_in_images_example.apply(
    lambda row: filter_axioms_by_predicates(row, column_name='positive_domain_using_capable_of_axioms_mapped'), axis=1
)
fol_axioms_in_images_example['negative_domain_using_not_capable_of_axioms_mapped_filtered'] = fol_axioms_in_images_example.apply(
    lambda row: filter_axioms_by_predicates(row, column_name='negative_domain_using_not_capable_of_axioms_mapped'), axis=1
)

# fol_axioms_in_images_example.to_csv('fol_axioms_in_images_example.csv', index=False)
# fol_axioms_in_images_example = pd.read_csv('fol_axioms_in_images_example.csv')
print("Finished")