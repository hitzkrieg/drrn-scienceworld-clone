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
    def __init__(self, embedding_server_port, taskIdx, threshold_strategy, pruning_strategy, threshold_file):
        """
        Args:
        embedding_server_port, 
        taskIdx, 
        threshold_strategy: tuning_strategy: can be one of similarity_threshold, top_k, top_p   
        threshold_file
        

        """
        self.embedding_server_port = embedding_server_port
        self.taskIdx = taskIdx
        self.pruning_strategy = pruning_strategy
        self.threshold_strategy = threshold_strategy
        # TO DO: replace by the temperature used in training the embeddings
        self.temperature = 1

        # if threshold_strategy == 'similarity_threshold', threshold is the minimum admissable cosine similarity, and all actions having similarity greater than threshold are returned 
        # if threshold_strategy == 'top_k', top k actions are returned  
        # if threshold_strategy == 'top_p', Top Actions are returned such that their cumulative probability adds upto p = threshold. Probabilities of actions are calculated by softmax(scores/temperature) 

        if self.pruning_strategy == 'hard' or self.pruning_strategy == 'hybrid':
            if self.threshold_strategy!= 'top_k':
                # be a little conservative by increasing threshold by 0.1
                self.threshold = self.load_threshold(threshold_file) + 0.1
            else:
                # TO DO:
                self.k = 50
        
        print("Testing embedding server:")
        temp_embedding = self.get_embedding(['hello world'])
        print(type(temp_embedding), temp_embedding.shape)
        print("Action Scorer ready!")

    def load_threshold(self, threshold_file):
        threshold_dict = json.load(threshold_file)
        return threshold_dictp[self.taskIdx]
    
    def get_embedding(self, sentences):
        """
        Get embeddings for a list of sentences from the embedding server
        """
        with grpc.insecure_channel(f'localhost:{self.embedding_server_port}') as channel:
            client = embeddingservice_pb2_grpc.EmbeddingServiceStub(channel)
            response = client.GetEmbedding(embeddingservice_pb2.EmbeddingRequest(sentences = sentences))
            embeddings_bytes = BytesIO(response.embeddings)
            embeddings = np.load(embeddings_bytes, allow_pickle=False)
        
        return embeddings

        
    def score_actions(self, actions, task_description):
        """
        Args
        actions: list of action strs
        task_description: task description str

        Returns:
        cosine_similarities (ndarray)
        normalized_scores (ndarray)
        desc_sorting_indices (ndarray)
        size_of_pruned_set (int)   
        """
        with grpc.insecure_channel(f'localhost:{self.embedding_server_port}') as channel:
            client = embeddingservice_pb2_grpc.EmbeddingServiceStub(channel)
            response = client.GetActionScores(embeddingservice_pb2.GetActionScoresRequest(actions = actions, task_description = task_description))
            cosine_similarities = np.asarray(list(response.cosine_similarities))
        normalized_scores = softmax(cosine_similarities / self.temperature)

        # Get descending order indices
        desc_sorting_indices = np.argsort(-cosine_similarities) 
        if self.pruning_strategy == 'soft':
            size_of_pruned_set = len(actions)
        else:            
            if self.threshold_strategy == 'top_k':
                size_of_pruned_set = self.k
            elif self.threshold_strategy == 'similarity_threshold':
                size_of_pruned_set = np.sum(cosine_similarities > self.threshold)

            if self.threshold_strategy == 'top_p':
                cumsum_probs = np.cumsum(normalized_scores[desc_sorting_indices])
                #say normalized_scores =  [0.8, 0.1, 0.05, 0.05],  np.cumsum(normalized_scores) = [0.8 , 0.9 , 0.95, 1. ]
                size_of_pruned_set = np.sum(cumsum_probs<self.threshold)+1
        
        return cosine_similarities, normalized_scores, desc_sorting_indices, size_of_pruned_set