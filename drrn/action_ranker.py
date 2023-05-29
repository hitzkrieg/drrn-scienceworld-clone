"""
Assumes that an embedding server is already running in Localhost. This file can be modified to handle case if server is hosted at another IP address. 
Current design gets embeddings from the server. If serializing embeddings, sending via gRPC and deserializing embeddings takes a lot of time, calculate the cosine similarities 
at serve
"""
import numpy as np 
import grpc
import embeddingservice_pb2
import embeddingservice_pb2_grpc
import time
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

class ActionScorer():
    """ 
    """
    def __init__(self, embedding_server_port, taskIdx, threshold_strategy, threshold_file):
        """
        tuning_strategy: can be one of similarity_threshold, top_k, top_p  
        """
        self.embedding_server_port = embedding_server_port
        self.taskIdx = taskIdx
        self.threshold_strategy = threshold_strategy
        self.temperature = 100

        # if threshold_strategy == 'similarity_threshold', threshold is the minimum admissable cosine similarity 
        # if threshold_strategy == 'top_k', top k actions are returned 
        # if threshold_strategy == 'top_p', Top Actions are returned such that their cumulative probability adds upto threshold. Probabilities of actions are calculated by softmax(scores/temperature) 

        self.threshold = self.load_threshold(threshold_file)

    def load_threshold(self, threshold_file):
        threshold_dict = json.load(threshold_file)
        return threshold_dictp[self.taskIdx]
        
    def score_actions(self, actions, task_description):
        """
        parameters:
        actions:
        task_description 

        Returns:
        cosine_similarities (ndarray)
        normalized_scores (ndarray)
        desc_sorting_indices (ndarray)
        size_of_pruned_set (int)   
        """
        with grpc.insecure_channel(f'localhost:{self.embedding_server_port}') as channel:
            client = embeddingservice_pb2_grpc.EmbeddingServiceStub(channel)
            response = client.GetActionScores(embeddingservice_pb2.GetActionScoresRequest(actions = actions, task_description = task_description))
            cosine_similarities = list(response.cosine_similarities)
        normalized_scores = softmax(cosine_similarities / self.temperature)

        # Get descending order indices
        desc_sorting_indices = np.argsort(-cosine_similarities) 
        
        if self.threshold_strategy == 'top_k':
            size_of_pruned_set = k
        elif self.threshold_strategy == 'similarity_threshold':
            size_of_pruned_set = np.sum(cosine_similarities > self.threshold)

        if self.threshold_strategy == 'top_p':
            cumsum_probs = np.cumsum(normalized_scores[desc_sorting_indices])
            #say normalized_scores =  [0.8, 0.1, 0.05, 0.05],  np.cumsum(normalized_scores) = [0.8 , 0.9 , 0.95, 1. ]
            size_of_pruned_set = np.sum(cumsum_probs<self.threshold)+1
        
        return cosine_similarities, normalized_scores, desc_sorting_indices, size_of_pruned_set