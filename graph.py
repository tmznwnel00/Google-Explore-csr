import os
import numpy as np
import pandas as pd

import firebase_admin
from firebase_admin import credentials, db
import networkx as nx

cred = credentials.Certificate(os.getcwd() + "/explorecsr-3178b-firebase-adminsdk-7kg05-7ed0ddb01b.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://explorecsr-3178b-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

ref = db.reference("papers")
papers = ref.get()
feature_vectors = pd.read_csv('feature_vectors.csv')

# all_author_ids = set()
# for paper_data in papers.values():
#     for author in paper_data.get('authors', []):
#         if "authorId" not in author:
#             continue
#         all_author_ids.add(author['authorId'])

# author_id_to_index = {author_id: idx for idx, author_id in enumerate(all_author_ids)}

G = nx.DiGraph()

for paper_id, paper_data in papers.items():
    year = paper_data.get('year', 1970)
    label = paper_data.get('label', None)
    
    # author_vector = np.zeros(len(all_author_ids))
    # for author in paper_data.get('authors', []):
    #     if "authorId" not in author:
    #         continue
    #     author_id = author['authorId']
    #     if author_id in author_id_to_index:
    #         author_vector[author_id_to_index[author_id]] = 1
    
    feature_vector = feature_vectors.loc[feature_vectors['paper_id'] == paper_id].iloc[:, 1:].values.flatten()
    
    # combined_features = np.concatenate(([year], author_vector, feature_vector))

    # node_attributes = {f'feature_{i}': feature for i, feature in enumerate(combined_features)}
    node_attributes = {f'feature_{i}': feature for i, feature in enumerate(feature_vector)}
    node_attributes['label'] = label
    
    G.add_node(paper_id, **node_attributes)

# Step 5: Add edges based on citations
for paper_id, paper_data in papers.items():
    for cited_paper_id in paper_data.get('citation_list', []):
        if cited_paper_id in papers:  # Ensure the cited paper is in the database
            G.add_edge(paper_id, cited_paper_id)

print("finished networkX")