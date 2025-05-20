import requests
from neo4j import GraphDatabase
import json
import re
import argparse
from bs4 import BeautifulSoup
from tqdm import tqdm
import logging
import numpy as np
import os
import requests
from difflib import SequenceMatcher

def is_similar(name1, name2, threshold=0.65):
    """
    Check if two strings are similar
    :param name1: First string
    :param name2: Second string
    :param threshold: Similarity threshold
    :return: True if similarity is greater than or equal to threshold, False otherwise
    """
    similarity = SequenceMatcher(None, name1, name2).ratio()
    return similarity >= threshold

def recall(recommended_items, actual_items):
    """
    Calculate Recall.
    Calculate how many of the actually clicked items were recommended.
    """
    hits = sum(1 for item in actual_items if item in recommended_items)
    return hits / len(actual_items) if actual_items else 0

def precision(recommended_items, actual_items):
    """
    Calculate Precision.
    Calculate the proportion of recommended items that were actually clicked.
    """
    hits = sum(1 for item in actual_items if item in recommended_items)
    return hits / len(recommended_items) if recommended_items else 0

def ndcg(prediction, ground_truth):
    """
    Calculate NDCG
    :param prediction: List of predicted books
    :param ground_truth: List of ground truth books
    :return: NDCG value
    """
    def dcg(relevance_scores):
        """Calculate DCG"""
        return np.sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores)])

    def idcg(ground_truth):
        """Calculate IDCG under ideal conditions"""
        # Calculate ideal ranking (books in ground truth should be ranked higher than non-ground truth books)
        relevance_scores = [1 if item in ground_truth else 0 for item in ground_truth]
        return dcg(relevance_scores)
    
    # Calculate relevance scores for each book in the predicted list
    relevance_scores = [1 if book in ground_truth else 0 for book in prediction]
    
    # Calculate DCG
    dcg_value = dcg(relevance_scores)
    
    # Calculate IDCG
    idcg_value = idcg(ground_truth)
    
    # Prevent division by zero
    if idcg_value == 0:
        return 0.0
    
    return dcg_value / idcg_value

def ftr(response):
    """ 
    Failed to recommend
    Input is the raw model output
    If there is no [SEP], it means the model did not output recommendations in the correct format, return False
    """
    if '[SEP]' not in response:
        return True
    else:
        return False

def name2id(name):
    """
    Find the book in the database corresponding to the name.
    """
    name = name.replace("'", "\\'")
    name = re.sub(r'\s\(\d{4}\)$', '', name)
    query = f"MATCH (n:Book) WHERE n.name = '{name}' RETURN n.name AS name LIMIT 1"
    with DRIVER.session(database=DATABASE) as session:
        result = session.run(query)
        record = result.single()
    if record:
        return record["name"]
    elif len(name.split()) >= 5:       
        query = f"MATCH (b:Book) WHERE b.name CONTAINS '{name}' RETURN b.name AS name LIMIT 1"
        with DRIVER.session(database=DATABASE) as session:
            result = session.run(query)
            record = result.single()
        if record:
            return record["name"]
        return None

def check_by_KG(name, conditions):
    """
    Use knowledge graph to check if a book satisfies the query conditions.

    :param name: Book name
    :param conditions: Query conditions
    :return: True if the book satisfies all query conditions, False otherwise
    """
    try:
        # Escape special characters in book name and wrap in single quotes
        escaped_name = name.replace("'", "\\'")
        cypher_query = f"MATCH (b:Book {{name: '{escaped_name}'}})"
        
        # Build WHERE clause
        where_clauses = []
        for condition in conditions:
            relationship_type, target_node_name = condition
            target_node_name = target_node_name.replace("'", "\\'")  # Escape special characters
            source_node_label, target_node_label = RELATION_SCHEMA[relationship_type]
            where_clauses.append(
                f"(b)-[:{relationship_type}]->(:{target_node_label} {{name: '{target_node_name}'}})"
            )

        # Concatenate query
        if where_clauses:
            cypher_query += " WHERE " + " AND ".join(where_clauses)
        cypher_query += " RETURN b"

        # Execute query
        with DRIVER.session(database=DATABASE) as session:
            result = session.run(cypher_query).data()

        return len(result) > 0

    except Exception as e:
        # Print error message for debugging
        print(f"Error occurred while running Cypher query: {str(e)}")
        print(f"Generated Cypher query: {cypher_query}")
        return False
   

def get_predicted_book_titles(prediction_response):
    """
    Extract recommended book titles from the raw model output.
    """
    predicted_book_titles = prediction_response.strip().strip("Recommended books (separated by [SEP]):").split("[SEP]")
    return predicted_book_titles

def preprocess_matching(recommended_items, actual_items, threshold=0.8):
    """
    Preprocess matching items.
    Output the matching results of predicted list and ground truth list into a new list,
    replacing books in the predicted list with matched results.
    """
    new_recommended_items = []

    for i, recommended_item in enumerate(recommended_items):
        for actual_item in actual_items:
            if is_similar(recommended_item, actual_item, threshold):
                new_recommended_items.append(actual_item)
                break
        else:
            new_recommended_items.append(recommended_item)
    assert len(new_recommended_items) == len(recommended_items)
    return new_recommended_items


def eval(prediction_response,groundtrue_data, query_type='condition'):
    '''
    :param groundtrues: Ground truth results (bookIDs) for a user
    :param predicted_movie_titles: Predicted results (book titles) for a user
    :param conditions: Query conditions
    :param metrics: List of metrics to evaluate
    '''
    
    results = {}
    # Calculate failed to recommend ratio
    results['ftr'] = ftr(prediction_response)

    predicted_book_titles = get_predicted_book_titles(prediction_response)
    matched_book_titles = preprocess_matching(predicted_book_titles, groundtrue_data['bookSubset'])
    # print(predicted_movie_titles)
    # print(groundtrue_data['movieSubset'])
    # print(matched_movie_titles) 
    
    # Print book proportion in kg

    # logging.info(f"Ratio of Books in KG: {len([m for m in predicted_book_ids if m is not None])/len(predicted_book_titles)}")
    
    recall_score = recall(matched_book_titles, groundtrue_data['bookSubset'])
    precision_score = precision(matched_book_titles, groundtrue_data['bookSubset'])
    ndcg_score = ndcg(matched_book_titles, groundtrue_data['bookSubset'])

    if query_type == 'condition':
        predicted_book_ids = [name2id(book) for book in predicted_book_titles] 
        recall_score = max(recall_score, recall(predicted_book_ids, groundtrue_data['bookSubset']))
        precision_score = max(precision_score, precision(predicted_book_ids, groundtrue_data['bookSubset']))
        ndcg_score = max(ndcg_score, ndcg(predicted_book_ids, groundtrue_data['bookSubset']))

    results["recall"] = recall_score
    results["precision"] = precision_score
    results["ndcg"] = ndcg_score
    
    # Calculate existence ratio
    
    # titles_need_check = [predicted_book_titles[i] for i, book in enumerate(predicted_book_ids) if not book] if query_type == 'condition' else predicted_book_titles
    # existence_for_titles_need_check = existence(titles_need_check)
    # if query_type == 'condition':
    #     existence_ratio_score = (sum(existence_for_titles_need_check) + len([book for book in predicted_book_ids if book is not None])) / len(predicted_book_titles)
    # else:
    #     existence_ratio_score = sum(existence_for_titles_need_check) / len(predicted_book_titles)
    # results["existence_ratio"] = existence_ratio_score
    # logging.info(f"Existence ratio: {existence_ratio_score}")


    if query_type == 'condition':
        satisfied_count = 0
        unsatisfied_books = []
        exist_in_KG_ratio = len([m for m in predicted_book_ids if m is not None])/len(predicted_book_titles)
        for i, book in enumerate(predicted_book_ids):
            if predicted_book_ids[i] not in groundtrue_data['bookSubset']:
                book_name = predicted_book_titles[i]
                book_id = predicted_book_ids[i]
                if book_id:  # If book exists in database
                    if check_by_KG(book_id, groundtrue_data['sharedRelationships']):   
                        satisfied_count += 1
                        # logging.info(f"{book_name} checked by KG: True")
                    else:
                        unsatisfied_books.append(book_name)
                        # logging.info(f"{book_name} checked by KG: False")
        satisfied_count += len([book for book in groundtrue_data['bookSubset'] if book in predicted_book_ids])
        satisfied_ratio = satisfied_count / len([m for m in predicted_book_ids if m is not None]) if exist_in_KG_ratio != 0 else -1
        results["satisfied_ratio"] = satisfied_ratio
        results["unsatisfied_books"] = unsatisfied_books
        results["existence_in_KG_ratio"] = exist_in_KG_ratio

    return results




def get_relations_from_schema(schema_file):
    with open(schema_file, "r") as file:
        schema = json.load(file)
    relations = schema["relations"]
    return {rel["type"]: (rel["source"], rel["target"]) for rel in relations}

def readPredictions(predictions_file):
    predictions = []
    with open(predictions_file, "r") as file:
        for line in file:
            predictions.append(json.loads(line))
    return predictions

def readGroundTruths(groundtruths_file):
    with open(groundtruths_file, "r") as file:
        groundtruths = json.load(file)
    return groundtruths

def calculate_avg(metric, evaluation_results):
    result = [result[metric] for result in evaluation_results if result[metric] != -1]
    # print(result)
    return sum(result) / len(result)

def eval_batch(args):
    # Read predictions
    predictions = readPredictions(args.predictions)
    # Read groundtruths
    groundtruths = readGroundTruths(args.groundtruths)
    # if args.query_type == 'condition':
    #     assert len(predictions) == len(groundtruths)
    
    # Check if there is a intermediate result file
    output_path = os.path.join(args.output_dir, args.predictions.split("/")[-1].replace(".jsonl", ".json"))
    if os.path.exists(output_path):
        with open(output_path, "r") as file:
            evaluation_results = json.load(file)
        start_idx = len(evaluation_results)
    else:
        evaluation_results = []
        start_idx = 0

    for i in tqdm(range(len(predictions)), desc="Evaluating predictions"):
        if i < start_idx:
            continue
        if args.query_type == 'condition':
            prediction_response = predictions[i]['response']
            id = predictions[i]['id']
            # Find corresponding groundtruth
            groundtruth_data = [data for data in groundtruths if data['data_idx'] == int(id)][0]
            eval_res = {"id": predictions[i]['id'],"condition_num": groundtruth_data['condition_num']}

            eval_res.update(eval(prediction_response, groundtruth_data, args.query_type))
            evaluation_results.append(eval_res)
        else:
            prediction_response = predictions[i]['response']
            idx1,idx2 = predictions[i]['id'].split('-')
            idx1 = int(idx1)
            idx2 = int(idx2)
            groundtruth_data = groundtruths[idx1]['query & ground true'][idx2]
            # try:
            groundtruth_data['bookSubset'] = groundtruth_data['book subset']
            # except:
            #     continue
            groundtruth_data['bookSubsetId'] = ['none']
            eval_res = {"id": predictions[i]['id']}
            eval_res.update(eval(prediction_response, groundtruth_data, args.query_type))
            evaluation_results.append(eval_res)

        # Write to file periodically
        if i % 20 == 0:
            with open(output_path, "w") as file:
                json.dump(evaluation_results, file, indent=4)
    # Write to file one last time
    with open(output_path, "w") as file:
        json.dump(evaluation_results, file, indent=4)
    metrics = [m for m in list(evaluation_results[0].keys()) if m not in ['id', 'unsatisfied_books','Condition Num']]
    avgs = {metric: calculate_avg(metric, evaluation_results) for metric in metrics}
    for metric, avg in avgs.items():
        print(f" {metric.replace('_', ' ').title()}: {avg}")
    
    
    
    return evaluation_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the performance of a recommendation system")
    parser.add_argument("--uri", type=str, default="bolt://localhost:7687", help="URI for Neo4j database")
    parser.add_argument("--username", type=str, default="neo4j", help="Username for Neo4j database")
    parser.add_argument("--password", type=str, default="", help="Password for Neo4j database")
    parser.add_argument("--database", type=str, default="", help="Target database name")
    parser.add_argument("--schema", type=str, default="book-schema.json", help="Path to schema file")
    parser.add_argument("--query_type", type=str, default="collaborative", choices=["condition", "collaborative"], help="query type")
    parser.add_argument("--groundtruths", type=str, default="../dataset/book/ItemBasedQuery.json", help="Path to ground truths file")   
    parser.add_argument("--predictions", type=str, default="../llm_results/gpt-4o-mini/book-ItemBasedQuery_gpt-4o-mini-prediction.jsonl", help="Path to predictions file")
    parser.add_argument("--output_dir", type=str, default="../eval_results", help="Path to output folder")
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args() 

    DRIVER = GraphDatabase.driver(args.uri, auth=(args.username, args.password))
    DATABASE = args.database

    # Test connection success
    try:
        with DRIVER.session(database=DATABASE) as session:
            session.run("RETURN 1")
        logging.info("Successfully connected to the Neo4j database.")
    except Exception as e:
        logging.info(f"Failed to connect to the Neo4j database: {e}")
        exit(1)

    RELATION_SCHEMA = get_relations_from_schema(args.schema)

    # Evaluate
    evaluation_results = eval_batch(args)
    
    # Close driver
    DRIVER.close()
