import json
import pandas as pd
import joblib
import os
from src.flattening import load_json, flatten_attributes, flatten_objects, flatten_attribute_synsets, flatten_object_synsets, flatten_qa_to_region_mapping, flatten_question_answers, flatten_region_descriptions, flatten_region_graphs, flatten_relationships, flatten_relationship_synsets, flatten_scene_graphs, flatten_synsets
from pathlib import Path
import logging


logger = logging.getLogger("scripts.process_flattening")


# Retrieve the path containing the Visual Genome json files downloaded from https://homes.cs.washington.edu/~ranjay/visualgenome/api.html
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data"
logger.info(f"Data directory: {data_dir}")


## Flattening part (transfrom json files in Visual Genome in
flattening_functions = {
    'attributes.json': flatten_attributes,
    'attribute_synsets.json': flatten_attribute_synsets,
    'objects.json': flatten_objects,
    'object_synsets.json': flatten_object_synsets,
    'qa_to_region_mapping.json': flatten_qa_to_region_mapping,
    'question_answers.json': flatten_question_answers,
    'region_descriptions.json': flatten_region_descriptions,
    'region_graphs.json': flatten_region_graphs,
    'relationships.json': flatten_relationships,
    'relationship_synsets.json': flatten_relationship_synsets,
    'scene_graphs.json': flatten_scene_graphs,
    'synsets.json': flatten_synsets
}


attributes_data = []
attribute_synsets_data = []
objects_data = []
object_synsets_data = []
qa_to_region_mapping_data = []
question_answers_data = []
region_descriptions_data = []
region_graphs_data = []
relationships_data = []
relationship_synsets_data = []
scene_graphs_data = []
synsets_data = []




for json_file, flatten_func in flattening_functions.items():
    json_filepath = os.path.join(data_dir, json_file)

    data = load_json(json_filepath)

    flattened_data = flatten_func(data)

    logger.info(f"Flattened DataFrame for {json_file}:")
    (logger.info(flattened_data.head(), "\n"))

    joblib_filepath = os.path.join(data_dir, json_file.replace('.json', '.joblib'))

    # Save the flattened DataFrame to a .joblib file with compression
    joblib.dump(flattened_data, joblib_filepath, compress=3)

    # Store the loaded DataFrame in the dictionary
    var_name = json_file.replace('.json', '_data')
    locals()[var_name] = flattened_data

    logger.info(f"Successfully saved flattened {json_file} to {joblib_filepath}")

logger.info("All JSON files have been flattened and saved as .joblib files.")




# # Example usage
# attributes_data = loaded_data_dict['attributes_data']
#
# attribute_synsets_data = loaded_data_dict['attribute_synsets_data']
#
# objects_data = loaded_data_dict['objects_data']
#
# object_synsets_data = loaded_data_dict['object_synsets_data']
#
# qa_to_region_mapping_data = loaded_data_dict['qa_to_region_mapping_data']
#
# question_answers_data = loaded_data_dict['question_answers_data']
#
# region_descriptions_data = loaded_data_dict['region_descriptions_data']
#
# region_graphs_data = loaded_data_dict['region_graphs_data']
#
# relationships_data = loaded_data_dict['relationships_data']
#
# relationship_synsets_data = loaded_data_dict['relationship_synsets_data']
#
# scene_graphs_data = loaded_data_dict['scene_graphs_data']
#
# synsets_data = loaded_data_dict['synsets_data']
#




#MERGE SECTION



# Merge and Save Objects with Attributes
objects_attributes = pd.merge(objects_data, attributes_data, on='object_id', how='left',
                              suffixes=('_object', '_attribute'))

joblib.dump(objects_attributes, data_dir, 'objects_attributes.joblib')

# Merge relationships_data with objects_data
relationships_subject = pd.merge(
    relationships_data[["relationship_id", "predicate", "synsets", "subject_id", "object_id"]],
    objects_data,
    left_on='subject_id',
    right_on='object_id',
    how='left'
)

#   Rename columns to indicate they belong to the subject
relationships_subject.rename(columns={
    'object_id_x': 'object_id',
    'synsets_x': 'relationship_synsets',
    'subject_image_id': 'image_id',
    'object_id_y': 'subject_id',
    'synsets_y': 'subject_synsets',
    'h': 'subject_h',
    'w': 'subject_w',
    'x': 'subject_x',
    'y': 'subject_y',
    'names': 'subject_names',
    'merged_object_ids': 'subject_merged_object_ids'
}, inplace=True)

# Merge the result with objects_data again on object_id
relationships_full = pd.merge(
    relationships_subject,
    objects_data,
    on='object_id',
    how='left'
)

# Rename columns to indicate they belong to the object
relationships_full.rename(columns={
    'synsets': 'object_synsets',
    'h': 'object_h',
    'w': 'object_w',
    'x': 'object_x',
    'y': 'object_y',
    'names': 'object_names',
    'image_id_x': 'image_id',
    'merged_object_ids': 'object_merged_object_ids'
}, inplace=True)

relationships_full = relationships_full.drop('image_id_y', axis=1)

# Display and save the fully merged dataframe
joblib_filepath = os.path.join(data_dir, "relationships_full.joblib")
joblib.dump(relationships_full, joblib_filepath)
# relationships_full.head()



# Merge scene_graph_data with objects_data
scene_graphs_subject = pd.merge(
    scene_graphs_data,
    objects_data,
    left_on='subject_id',
    right_on='object_id',
    how='left'
)

# Rename columns to indicate they belong to the subject
scene_graphs_subject.rename(columns={
    'object_id_x': 'object_id',
    'synsets_x': 'relationship_synsets',
    'subject_image_id': 'image_id',
    'object_id_y': 'subject_id',
    'synsets_y': 'subject_synsets',
    'h': 'subject_h',
    'w': 'subject_w',
    'x': 'subject_x',
    'y': 'subject_y',
    'names': 'subject_names',
    'merged_object_ids': 'subject_merged_object_ids'
}, inplace=True)

# Merge the result with objects_data again on object_id
scene_graphs_full = pd.merge(
    relationships_subject,
    objects_data,
    on='object_id',
    how='left'
)

# Rename columns to indicate they belong to the object
scene_graphs_full.rename(columns={
    'synsets': 'object_synsets',
    'h': 'object_h',
    'w': 'object_w',
    'x': 'object_x',
    'y': 'object_y',
    'image_id_x': 'image_id',
    'names': 'object_names',
    'merged_object_ids': 'object_merged_object_ids'
}, inplace=True)

scene_graphs_full = scene_graphs_full.drop(['image_id_y', 'subject_id'], axis=1)

# Display and Save the fully merged dataframe
joblib_filepath = os.path.join(data_dir, "scene_graphs_full.joblib")
joblib.dump(scene_graphs_full, joblib_filepath)
# scene_graphs_full.head()


# Link Region Graphs
region_graphs_full = pd.merge(region_graphs_data,
                              objects_data[['object_id', 'names', 'synsets', 'h', 'w', 'y', 'x', 'merged_object_ids']],
                              left_on='relationship_object_id', right_on='object_id', how='left')
region_graphs_full.rename(columns={
    'width': 'region_w',
    'height': 'region_h',
    'x_x': 'region_x',
    'y_x': 'region_y',
    'object_id_x': 'object_id',
    'names': 'relationship_object_names',
    'synsets': 'relationship_object_synsets',
    'h': 'relationship_object_h',
    'w': 'relationship_object_w',
    'y_y': 'relationship_object_y',
    'x_y': 'relationship_object_x',
    'merged_object_ids': 'relationship_object_merged_object_ids'

}, inplace=True)
region_graphs_full.drop('object_id_y', axis=1)
region_graphs_full = pd.merge(region_graphs_full,
                              objects_data[['object_id', 'names', 'synsets', 'h', 'w', 'y', 'x', 'merged_object_ids']],
                              left_on='relationship_subject_id', right_on='object_id', how='left')
region_graphs_full.rename(columns={
    'object_id_x': 'object_id',
    'names': 'relationship_subject_names',
    'synsets': 'relationship_subject_synsets',
    'h': 'relationship_subject_h',
    'w': 'relationship_subject_w',
    'y': 'relationship_subject_y',
    'x': 'relationship_subject_x',
    'merged_object_ids': 'relationship_subject_merged_object_ids'
}, inplace=True)
region_graphs_full.drop('object_id_y', axis=1)
joblib_filepath = os.path.join(data_dir, "region_graphs_full.joblib")
joblib.dump(region_graphs_full, joblib_filepath)

# Connect QA to Regions
qa_region_mapping_full = pd.merge(qa_to_region_mapping_data, region_descriptions_data, on='region_id', how='left')
qa_region_mapping_full = pd.merge(qa_region_mapping_full, question_answers_data, on='qa_id', how='left')

joblib_filepath = os.path.join(data_dir, "qa_region_mapping_full.joblib")
joblib.dump(qa_region_mapping_full, joblib_filepath)
# qa_region_mapping_full.head()


logger.info("Data have been merged and saved as .joblib files.")