# Check if items exist
import requests
from neo4j import GraphDatabase
import json
import re
import argparse
from bs4 import BeautifulSoup
from bs4 import BeautifulSoup
from openai import OpenAI
from tqdm import tqdm
import logging
import numpy as np
import os

from difflib import SequenceMatcher

def is_similar(name1, name2, threshold=0.8):
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
    :param prediction: List of predicted movies
    :param ground_truth: List of ground truth movies
    :return: NDCG value
    """
    def dcg(relevance_scores):
        """Calculate DCG"""
        return np.sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores)])

    def idcg(ground_truth):
        """Calculate IDCG under ideal conditions"""
        # Calculate ideal ranking (movies in ground truth should be ranked higher than non-ground truth movies)
        relevance_scores = [1 if item in ground_truth else 0 for item in ground_truth]
        return dcg(relevance_scores)
    
    # Calculate relevance scores for each movie in the predicted list
    relevance_scores = [1 if movie in ground_truth else 0 for movie in prediction]
    
    # Calculate DCG
    dcg_value = dcg(relevance_scores)
    
    # Calculate IDCG
    idcg_value = idcg(ground_truth)
    
    # Prevent division by zero
    if idcg_value == 0:
        return 0.0
    
    return dcg_value / idcg_value


def existence(movie_titles):
    """
    Check if a set of movie titles exist in the OMDb database.

    :param movie_titles: List of movie titles
    :param api_key: OMDb API key
    :return: Dictionary indicating whether each movie title exists
    """
    url = "http://www.omdbapi.com/"
    
    res = []
    for title in movie_titles:
        # If more than 10 words, consider it does not exist
        if len(title.split(' '))>15 or len(title)>30:
            res.append(False)
            continue
        # Send request to OMDb API
        try:
            response = requests.get(url, params={"t": title, "apikey": "57d1ffe2"},timeout=3)
            data = response.json()
        except Exception as e:  
            logging.info(f"OMDb API request failed: {e}")
            res.append(False)
            continue

        # print(data)
        # Check if movie exists
        if data.get("Response") == "True":
            
            res.append(True)
        else:
            res.append(False)
    return res
            
        # print(data)
    
    # Calculate existence ratio
    # print(exists_count)

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
    Find the ID in the database corresponding to the name.
    """
    # Need to escape name
    # name = name.replace("'", "\\'").strip()
    # Remove year from name
    name = re.sub(r'\s\(\d{4}\)$', '', name).strip()
    name = name.replace('"', "'")
    query = f"MATCH (n:Movie) WHERE n.Title = \"{name}\" RETURN n.movieId AS movieId"
    with DRIVER.session(database=DATABASE) as session:
        result = session.run(query)
        record = result.single()
    if record:
        return record["movieId"]
    
    # If movie is not found, calculate fuzzy match
    else:       
        query = f"MATCH (m:Movie) WHERE m.Title CONTAINS \"{name}\" RETURN m.movieId AS movieId LIMIT 1"
        with DRIVER.session(database=DATABASE) as session:
            result = session.run(query)
            record = result.single()
        if record:
            return record["movieId"]
        return None

def check_by_KG(movieID, conditions):
    """
    Use knowledge graph to check if a movie satisfies the query conditions.

    :param movieID: Movie ID
    :param conditions: Query conditions
    Example: [
            [
                "Cinematography",
                "Dean Cundey"
            ],
            [
                "Produced_by",
                "Amblin Entertainment"
            ]
        ]
    :return: True if the movie satisfies all query conditions, False otherwise
    """
    # Basic MATCH query
    cypher_query = f"MATCH (m:Movie {{movieId: {movieID}}})"

    # Build WHERE clause
    where_clauses = []
    for condition in conditions:
        relationship_type, target_node_name = condition
        # Escape target_node_name
        target_node_name = target_node_name.replace("'", "\\'")
        source_node_label, target_node_label = RELATION_SCHEMA[relationship_type]
        # Generate independent relationship match for each condition
        where_clauses.append(
            f"(m)-[:{relationship_type}]->(:{target_node_label} {{name: '{target_node_name}'}})"
        )

    # If there are conditions, build WHERE clause and add AND connection
    if where_clauses:
        cypher_query += " WHERE " + " AND ".join(where_clauses)

    # Add RETURN clause
    cypher_query += " RETURN m"

    # print(cypher_query)
    with DRIVER.session(database=DATABASE) as session:
        result = session.run(cypher_query).data()
        # print(result)


    # If query result is not empty, it means the movie satisfies all conditions
    return len(result) > 0      

def get_predicted_movie_titles(prediction_response):
    """
    Extract recommended movie titles from the raw model output.
    """
    # Extract recommended movie titles from raw model output
    predicted_movie_titles = prediction_response.split("[SEP]")
    predicted_movie_titles = [movie.strip().strip('\'').strip('"') for movie in predicted_movie_titles]
    # Remove year from each
    predicted_movie_titles = [re.sub(r'\s\(\d{4}\)$', '', movie.strip()).strip() for movie in predicted_movie_titles]
    return predicted_movie_titles

def preprocess_matching(recommended_items, actual_items, threshold=0.8):
    """
    Preprocess matching items.
    Output the matching results of predicted list and ground truth list into a new list,
    replacing movies in the predicted list with matched results.
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

def eval(prediction_response,groundtrue_data, query_type='condition',k=0):
    '''
    :param groundtrues: Ground truth results (movieIDs) for a user
    :param predicted_movie_titles: Predicted results (movie titles) for a user
    :param conditions: Query conditions
    :param metrics: List of metrics to evaluate
    '''
    
    results = {}
    # Calculate failed to recommend ratio
    results['ftr'] = ftr(prediction_response)

    predicted_movie_titles = get_predicted_movie_titles(prediction_response)
    if k!=0:
        predicted_movie_titles = predicted_movie_titles[:k]
        if len(predicted_movie_titles)<k:
            predicted_movie_titles += ['none']*(k-len(predicted_movie_titles))
    # print(len(predicted_movie_titles))
    matched_movie_titles = preprocess_matching(predicted_movie_titles, groundtrue_data['movieSubset'])
    # print(predicted_movie_titles)
    # print(groundtrue_data['movieSubset'])
    # print(matched_movie_titles) 
    
    # Print movie proportion in kg

    # logging.info(f"Ratio of Movies in KG: {len([m for m in predicted_movie_ids if m is not None])/len(predicted_movie_titles)}")
    
    recall_score = recall(matched_movie_titles, groundtrue_data['movieSubset'])
    precision_score = precision(matched_movie_titles, groundtrue_data['movieSubset'])
    ndcg_score = ndcg(matched_movie_titles, groundtrue_data['movieSubset'])

    if query_type == 'condition':
        predicted_movie_ids = [name2id(movie) for movie in predicted_movie_titles] 
        recall_score = max(recall_score, recall(predicted_movie_ids, groundtrue_data['movieSubsetId']))
        precision_score = max(precision_score, precision(predicted_movie_ids, groundtrue_data['movieSubsetId']))
        ndcg_score = max(ndcg_score, ndcg(predicted_movie_ids, groundtrue_data['movieSubsetId']))

    results["recall"] = recall_score
    results["precision"] = precision_score
    results["ndcg"] = ndcg_score
    
    # Calculate existence ratio
    
    # titles_need_check = [predicted_movie_titles[i] for i, movie in enumerate(predicted_movie_ids) if not movie] if query_type == 'condition' else predicted_movie_titles
    # existence_for_titles_need_check = existence(titles_need_check)
    # if query_type == 'condition':
    #     existence_ratio_score = (sum(existence_for_titles_need_check) + len([movie for movie in predicted_movie_ids if movie is not None])) / len(predicted_movie_titles)
    # else:
    #     existence_ratio_score = sum(existence_for_titles_need_check) / len(predicted_movie_titles)
    # results["existence_ratio"] = existence_ratio_score
    # logging.info(f"Existence ratio: {existence_ratio_score}")


    if query_type == 'condition':
        satisfied_count = 0
        unsatisfied_movies = []
        exist_in_KG_ratio = len([m for m in predicted_movie_ids if m is not None])/len(predicted_movie_titles)
        for i, movie in enumerate(predicted_movie_ids):
            if predicted_movie_ids[i] not in groundtrue_data['movieSubsetId']:
                movie_name = predicted_movie_titles[i]
                movie_id = predicted_movie_ids[i]
                if movie_id:  # If movie exists in database
                    if check_by_KG(movie_id, groundtrue_data['sharedRelationships']):   
                        satisfied_count += 1
                        # logging.info(f"{movie_name} checked by KG: True")
                    else:
                        unsatisfied_movies.append(movie_name)
                        # logging.info(f"{movie_name} checked by KG: False")
        satisfied_count += len([movie for movie in groundtrue_data['movieSubsetId'] if movie in predicted_movie_ids])
        satisfied_ratio = satisfied_count / len([m for m in predicted_movie_ids if m is not None]) if exist_in_KG_ratio != 0 else -1
        results["satisfied_ratio"] = satisfied_ratio
        results["unsatisfied_movies"] = unsatisfied_movies
        results["existence_in_KG_ratio"] = exist_in_KG_ratio

    return results

# Read relations definition from schema.json, get label of each relation's head and tail nodes
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
    # Remove data with value -1
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
    
    # Check if there is intermediate result file
    output_path = os.path.join(args.output_dir, args.predictions.split("/")[-1].replace(".jsonl", ".json"))
    if os.path.exists(output_path):
        with open(output_path, "r") as file:
            evaluation_results = json.load(file)
        start_idx = len(evaluation_results)
    else:
        evaluation_results = []
        start_idx = 0

    for i in tqdm(range(start_idx, len(predictions)), desc="Evaluating predictions"):
        if args.query_type == 'condition':
            prediction_response = predictions[i]['response']
            id = predictions[i]['id']
            # Find corresponding groundtruth
            # print(id)
            try:
                groundtruth_data = [data for data in groundtruths if data['data_idx'] == int(id)][0]
                eval_res = {"id": predictions[i]['id'],"condition_num": groundtruth_data['condition_num']}

                eval_res.update(eval(prediction_response,groundtruth_data, args.query_type,args.k))
            except:
                print(id," not found")   
            evaluation_results.append(eval_res)
        else:
            prediction_response = predictions[i]['response']
            idx1,idx2 = predictions[i]['id'].split('-')
            idx1 = int(idx1)
            idx2 = int(idx2)
            groundtruth_data = [d for d in groundtruths[idx1]['query & ground true'] if d!={}][idx2]
            # try:
            if 'movie subset' not in groundtruth_data:
                print(idx1,idx2)
            groundtruth_data['movieSubset'] = groundtruth_data['movie subset']
            # except:
            #     continue
            groundtruth_data['movieSubsetId'] = ['none']
            eval_res = {"id": predictions[i]['id']}
            eval_res.update(eval(prediction_response, groundtruth_data, args.query_type,args.k))
            evaluation_results.append(eval_res)

        # Write to file periodically
        if i % 20 == 0:
            with open(output_path, "w") as file:
                json.dump(evaluation_results, file, indent=4)
    # Write to file once at the end
    assert len(evaluation_results) == len(predictions)
    with open(output_path, "w") as file:
        json.dump(evaluation_results, file, indent=4)

    metrics = [m for m in list(evaluation_results[0].keys()) if m not in ['id', 'unsatisfied_movies']]
    avgs = {metric: calculate_avg(metric, evaluation_results) for metric in metrics}
    for metric, avg in avgs.items():
        logging.info(f" {metric.replace('_', ' ').title()}: {avg}")
    
    
    return evaluation_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the performance of a recommendation system")
    parser.add_argument("--uri", type=str, default="bolt://localhost:7687", help="URI for Neo4j database")
    parser.add_argument("--username", type=str, default="neo4j", help="Username for Neo4j database")
    parser.add_argument("--password", type=str, default="", help="Password for Neo4j database")
    parser.add_argument("--database", type=str, default="", help="Target database name")
    parser.add_argument("--schema", type=str, default="movie-schema.json", help="Path to schema file")
    parser.add_argument("--query_type", type=str, default="condition", choices=["condition", "collaborative"], help="query type")
    parser.add_argument("--groundtruths", type=str, default="../dataset/movie/MisinformedQuery.json", help="Path to ground truths file")   
    parser.add_argument("--predictions", type=str, default="../llm_results/gpt-4o-mini/movie-MisinformedQuery_gpt-4o-mini-prediction.jsonl", help="Path to predictions file")
    parser.add_argument("--output_dir", type=str, default="../eval_results", help="Path to output folder")
    parser.add_argument("--k", type=int, help="Number of recommendations to evaluate", default=0)
    
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args() 

    # Create driver
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

    # Read schema.json file
    RELATION_SCHEMA = get_relations_from_schema(args.schema)

    # Evaluate
    evaluation_results = eval_batch(args)
    
    # Close driver
    DRIVER.close()

   


    




    
