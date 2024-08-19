import os

import firebase_admin
from firebase_admin import credentials, db
import networkx as nx
import matplotlib.pyplot as plt

cred = credentials.Certificate(os.getcwd() + "/explorecsr-3178b-firebase-adminsdk-7kg05-7ed0ddb01b.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://explorecsr-3178b-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

ref = db.reference("papers")

papers = ref.get()

G = nx.Graph()

for paper_id, paper_data in papers.items():
    G.add_node(paper_id, title=paper_data['title'])

    for ref_id in paper_data.get('reference_list', []):
        G.add_edge(paper_id, ref_id)  # A -> B

plt.figure(figsize=(30, 30))  
pos = nx.spring_layout(G, k=0.1)
nx.draw(G, pos, node_size=10, node_color="skyblue", with_labels=False, edge_color="black", alpha=0.7)

# plt.title("Citation and Reference Network")
plt.savefig("citation_reference_network.png")

plt.close()