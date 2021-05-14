"""
This is a more complex example on performing clustering on large scale dataset.
This examples find in a large set of sentences local communities, i.e., groups of sentences that are highly
similar. You can freely configure the threshold what is considered as similar. A high threshold will
only find extremely similar sentences, a lower threshold will find more sentence that are less similar.
A second parameter is 'min_community_size': Only communities with at least a certain number of sentences will be returned.
The method for finding the communities is extremely fast, for clustering 50k sentences it requires only 5 seconds (plus embedding comuptation).
In this example, we download a large set of questions from Quora and then find
similar questions in this set.
https://www.sbert.net/examples/applications/clustering/README.html
https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/clustering/fast_clustering.py
"""
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import csv
import pickle
import time


def pair_metric(embeddings, threshold=0.8):


    cos_scores = util.pytorch_cos_sim(embeddings, embeddings)
    print(cos_scores.shape)
    print(cos_scores[0])
    print(cos_scores[1])

def community_detection(embeddings, threshold=0.75, min_community_size=25, init_max_size=1000):
    """
    Function for Fast Community Detection
    Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
    Returns only communities that are larger than min_community_size. The communities are returned
    in decreasing order. The first element in each list is the central point in the community.
    # Slow, deprecated
    """

    # Compute cosine similarity scores
    start_time = time.time()
    cos_scores = util.pytorch_cos_sim(embeddings, embeddings)
    print("Compare after {:.2f} sec".format(time.time() - start_time))    
    print(cos_scores.shape)
    print(cos_scores[0])
    print(cos_scores[1])
    # Minimum size for a community
    top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)

    # Filter for rows >= min_threshold
    extracted_communities = []
    for i in range(len(top_k_values)):
        if top_k_values[i][-1] >= threshold:
            new_cluster = []

            # Only check top k most similar entries
            top_val_large, top_idx_large = cos_scores[i].topk(k=init_max_size, largest=True)
            top_idx_large = top_idx_large.tolist()
            top_val_large = top_val_large.tolist()

            if top_val_large[-1] < threshold:
                for idx, val in zip(top_idx_large, top_val_large):
                    if val < threshold:
                        break

                    new_cluster.append(idx)
            else:
                # Iterate over all entries (slow)
                for idx, val in enumerate(cos_scores[i].tolist()):
                    if val >= threshold:
                        new_cluster.append(idx)

            extracted_communities.append(new_cluster)

    # Largest cluster first
    extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

    # Step 2) Remove overlapping communities
    unique_communities = []
    extracted_ids = set()

    for community in extracted_communities:
        add_cluster = True
        for idx in community:
            if idx in extracted_ids:
                add_cluster = False
                break

        if add_cluster:
            unique_communities.append(community)
            for idx in community:
                extracted_ids.add(idx)

    return unique_communities

# Model for computing sentence embeddings. We use one trained for similar questions detection
#model = SentenceTransformer('protbert...')

# We donwload the Quora Duplicate Questions Dataset (https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs)
# and find similar question in it
#url = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"
#dataset_path = "quora_duplicate_questions.tsv"
#max_corpus_size = 50000 # We limit our corpus to only the first 50k questions
#embedding_cache_path = 'quora-embeddings-size-{}.pkl'.format(max_corpus_size)

embedding_cache_path = '/scratch/gpfs/cmcwhite/qfo_2020/qfo_2020.pkl'

#dataset_path = 


print("Load pre-computed embeddings from disk")
with open(embedding_cache_path, "rb") as fIn:
    cache_data = pickle.load(fIn)
    corpus_sentences = cache_data['sequences']
    corpus_embeddings = cache_data['embeddings']

print("Start clustering")
start_time = time.time()

#Two parameter to tune:
#min_cluster_size: Only consider cluster that have at least 25 elements (30 similar sentences)
#threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
clusters = community_detection(corpus_embeddings, min_community_size=25, threshold=0.95)




#Print all cluster / communities
with open("/scratch/gpfs/cmcwhite/qfo_2020/qfo_clusters.txt", "w") as outfile:

    for i, cluster in enumerate(clusters):
        print(i)
        for sentence_id in cluster:
            outfile.write("{}\t{}\n".format(i, corpus_sentences[sentence_id]))



print("Clustering done after {:.2f} sec".format(time.time() - start_time))
