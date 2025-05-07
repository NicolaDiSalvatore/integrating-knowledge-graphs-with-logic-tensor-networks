import json
import pandas as pd
import joblib
import os


data_dir = 'C:/Users/nicol/PycharmProjects/VisualGenomeProject'


def flatten_attributes(data):

    rows = []

    for entry in data:
        image_id = entry['image_id']
        for attr in entry['attributes']:
            row = {
                'image_id': image_id,
                'object_id': attr.get('object_id'),
                'synsets': attr.get('synsets'),
                'h': attr.get('h'),
                'w': attr.get('w'),
                'x': attr.get('x'),
                'y': attr.get('y'),
                'names': attr.get('names')
            }
            rows.append(row)
    return pd.DataFrame(rows)


def flatten_attribute_synsets(data):

    flattened_data = [{'attribute_name': key, 'synset': value} for key, value in data.items()]
    df = pd.DataFrame(flattened_data)
    return df




def flatten_objects(data):

    flattened_data = []

    for entry in data:
        image_id = entry['image_id']
        for obj in entry['objects']:
            flattened_obj = {
                'image_id': image_id,
                'object_id': obj.get('object_id', None),
                'synsets': obj.get('synsets', []),
                'h': obj.get('h', None),
                'w': obj.get('w', None),
                'y': obj.get('y', None),
                'x': obj.get('x', None),
                'names': obj.get('names', []),
                'merged_object_ids': obj.get('merged_object_ids', [])
            }
            flattened_data.append(flattened_obj)

    df = pd.DataFrame(flattened_data)
    return df


def flatten_object_synsets(data):
    flattened_data = [{'object_name': key, 'synset': value} for key, value in data.items()]
    df = pd.DataFrame(flattened_data)
    return df


def flatten_qa_to_region_mapping(data):
    flattened_data = [{'qa_id': int(key), 'region_id': int(value)} for key, value in data.items()]
    df = pd.DataFrame(flattened_data)
    return df

def flatten_question_answers(data):

    flattened_data = []

    for entry in data:
        for qa in entry['qas']:
            flattened_qa = {
                'id': entry.get('id', None),
                'question': qa.get('question', None),
                'image_id': qa.get('image_id', None),
                'qa_id': qa.get('qa_id', None),
                'answer': qa.get('answer', None),
                'q_objects': qa.get('q_objects', []),
                'a_objects': qa.get('a_objects', [])
            }
            flattened_data.append(flattened_qa)

    df = pd.DataFrame(flattened_data)
    return df


def flatten_region_descriptions(data):

    flattened_data = []

    for entry in data:
        for region in entry['regions']:
            flattened_region = {
                'region_id': region.get('region_id', None),
                'width': region.get('width', None),
                'height': region.get('height', None),
                'image_id': region.get('image_id', None),
                'phrase': region.get('phrase', None),
                'y': region.get('y', None),
                'x': region.get('x', None),
            }
            flattened_data.append(flattened_region)
    df = pd.DataFrame(flattened_data)
    return df



def flatten_relationships(data):

    rows = []

    for entry in data:
        relationships = entry.get('relationships', [])
        for rel in relationships:
            row = {
                'relationship_id': rel.get('relationship_id'),
                'predicate': rel.get('predicate'),
                'object_id': rel.get('object', {}).get('object_id'),
                'object_name': rel.get('object', {}).get('names', [None])[0],
                # Assuming names list has at least one entry
                'object_synsets': rel.get('object', {}).get('synsets'),
                'object_h': rel.get('object', {}).get('h'),
                'object_w': rel.get('object', {}).get('w'),
                'object_x': rel.get('object', {}).get('x'),
                'object_y': rel.get('object', {}).get('y'),
                'subject_id': rel.get('subject', {}).get('object_id'),
                'subject_name': rel.get('subject', {}).get('name'),
                'subject_synsets': rel.get('subject', {}).get('synsets'),
                'subject_h': rel.get('subject', {}).get('h'),
                'subject_w': rel.get('subject', {}).get('w'),
                'subject_x': rel.get('subject', {}).get('x'),
                'subject_y': rel.get('subject', {}).get('y'),
                'synsets': rel.get('synsets'),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df

def flatten_relationship_synsets(data):
    flattened_data = [{'relationship_name': key, 'synset': value} for key, value in data.items()]
    df = pd.DataFrame(flattened_data)
    return df


def flatten_region_graphs(data):

    flat_data = []

    for graph in data:
        for region in graph['regions']:
            base_data = {
                'region_id': region['region_id'],
                'width': region['width'],
                'height': region['height'],
                'image_id': region['image_id'],
                'phrase': region['phrase'],
                'x': region['x'],
                'y': region['y']
            }

            for synset in region['synsets']:
                synset_data = base_data.copy()
                synset_data.update({
                    'synset_entity_idx_start': synset['entity_idx_start'],
                    'synset_entity_idx_end': synset['entity_idx_end'],
                    'synset_entity_name': synset['entity_name'],
                    'synset_name': synset['synset_name']
                })
                flat_data.append(synset_data)

            for obj in region['objects']:
                object_data = base_data.copy()
                object_data.update({
                    'object_id': obj['object_id'],
                    'object_name': obj['name'],
                    'object_h': obj['h'],
                    'object_w': obj['w'],
                    'object_x': obj['x'],
                    'object_y': obj['y'],
                    'object_synsets': obj['synsets']
                })
                flat_data.append(object_data)

            for rel in region['relationships']:
                relationship_data = base_data.copy()
                relationship_data.update({
                    'relationship_id': rel['relationship_id'],
                    'relationship_predicate': rel['predicate'],
                    'relationship_synsets': rel['synsets'],
                    'relationship_subject_id': rel['subject_id'],
                    'relationship_object_id': rel['object_id']
                })
                flat_data.append(relationship_data)


    df = pd.DataFrame(flat_data)
    return df


def flatten_scene_graphs(data):

    flattened_data = []

    for entry in data:
        for relationship in entry['relationships']:
            flattened_relationship = {
                'relationship_id': relationship.get('relationship_id', None),
                'predicate': relationship.get('predicate', None),
                'object_id': relationship.get('object_id', None),
                'subject_id': relationship.get('subject_id', None),
                'synsets': relationship.get('synsets', [])
            }
            flattened_data.append(flattened_relationship)

    df = pd.DataFrame(flattened_data)
    return df


def flatten_synsets(data):
    flattened_data = [{'synset_name': item['synset_name'], 'definition': item['synset_definition']} for item in data]
    df = pd.DataFrame(flattened_data)
    return df

# Example usage
# synsets_df = flatten_synsets('path/to/synsets.json')
# print(synsets_df.head())


flattening_functions = {
    # 'attributes.json': flatten_attributes,
    'attribute_synsets.json': flatten_attribute_synsets,
    # 'objects.json': flatten_objects,
    'object_synsets.json': flatten_object_synsets,
    # 'qa_to_region_mapping.json': flatten_qa_to_region_mapping,
    # 'question_answers.json': flatten_question_answers,
    # 'region_descriptions.json': flatten_region_descriptions,
    # 'region_graphs.json': flatten_region_graphs,
    # 'relationships.json': flatten_relationships,
    # 'relationship_synsets.json': flatten_relationship_synsets,
    # 'scene_graphs.json': flatten_scene_graphs,
    # 'synsets.json': flatten_synsets
}



def load_json(json_filepath):
    """
    Loads a JSON file and returns its content.
    """
    with open(json_filepath, 'r') as f:
        data = json.load(f)
    return data















