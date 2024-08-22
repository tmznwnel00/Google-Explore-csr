import os
import numpy as np

import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, request, jsonify, render_template

import search_engine

cred = credentials.Certificate(os.getcwd() + "/explorecsr-3178b-firebase-adminsdk-7kg05-7ed0ddb01b.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://explorecsr-3178b-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

ref = db.reference("papers")

node_embeddings_array = np.load('final_embeddings.npy')
node_ids = node_embeddings_array[:, 0]

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_papers():
    query = request.json.get('query', '')
    results = search_engine.search(query)
    papers = []
    for result in results:
        papers.append(ref.child(node_ids[result]).get())
    return jsonify(papers)

@app.route('/paper')
def get_paper():
    paper_id = request.args.get('paperId')
    paper = ref.child(paper_id).get()
    paper_details = {
        "id": paper_id,
        "title": paper["title"],
        "authors": paper["authors"],
        "year": paper["year"],
        "abstract": paper["abstract"]
    }
    return jsonify(paper_details)

if __name__ == "__main__":
    app.run(debug=True)
