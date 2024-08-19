import os

import firebase_admin
from firebase_admin import credentials, db
from semanticscholar import SemanticScholar

cred = credentials.Certificate(os.getcwd() + "/explorecsr-3178b-firebase-adminsdk-7kg05-7ed0ddb01b.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://explorecsr-3178b-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

author_dict = {
    "박현우": "https://www.semanticscholar.org/author/Hyunwoo-Park/1869386",
    "이준석": "https://www.semanticscholar.org/author/Joonseok-Lee/2119006",
    "김태섭": "https://www.semanticscholar.org/author/Taesup-Kim/3307885",
    "김형신": "https://www.semanticscholar.org/author/Hyung-Sin-Kim/2042630",
    "성효진": "https://www.semanticscholar.org/author/Hyojin-Sung/2946535",
    "오민환": "https://www.semanticscholar.org/author/Min-hwan-Oh/2683817",
    "이상원": "https://www.semanticscholar.org/author/Sang-Won-Lee/2108243387",
    "이상학": "https://www.semanticscholar.org/author/Sanghack-Lee/2144568286", 
    "이승근": "https://www.semanticscholar.org/author/Seunggeun-Lee/37780942",
    "이재윤": "https://www.semanticscholar.org/author/Jay-Yoon-Lee/11073942",
    "이재진": "https://www.semanticscholar.org/author/Jaejin-Lee/2108395388",
    "정형수": "https://www.semanticscholar.org/author/Hyungsoo-Jung/1692265",
    "조요한": "https://www.semanticscholar.org/author/Yohan-Jo/39947629",
    "wen": "https://www.semanticscholar.org/author/Wen-Syan-Li/2108717697"
}

author_id_list = []

for name, link in author_dict.items():
    id = link.split('/')[-1]
    author_id_list.append(id)

sch = SemanticScholar()

def upload_paper(p, origin_id=None, edge_type=None):
    if "fieldsOfStudy" in p:
        pass
    elif "citedPaper" in p:
        p = p["citedPaper"]
    else:
        p = p["citingPaper"]

    fieldsOfStudy = p["fieldsOfStudy"]
    if fieldsOfStudy and "Computer Science" in fieldsOfStudy and p["abstract"]:
        data = {
            "paperId": p["paperId"],
            "title": p["title"],
            "year": p["year"],
            "authors": p["authors"],
            "abstract": p["abstract"],
            "reference_list": [],
            "citation_list": []
        }
        ref = db.reference(f"papers/{data['paperId']}")

        if ref.get() is None:
            ref.set(data)
            print(data["title"]) 
        
        if origin_id:
            if edge_type == 'r':
                add_edge(origin_id, p["paperId"])
            elif edge_type == 'c':
                add_edge(p["paperId"], origin_id)


def add_edge(p1, p2):
    ref1 = db.reference(f"papers/{p1}")
    paper1 = ref1.get()

    ref2 = db.reference(f"papers/{p2}")
    paper2 = ref2.get()

    if not paper2:
        pass
    else:
        if "reference_list" in paper1 and p2 in paper1["reference_list"]:
            pass
        else:
            paper1.setdefault("reference_list", []).append(p2)
            paper2.setdefault("citation_list", []).append(p1)
            ref1.set(paper1)
            ref2.set(paper2)
            print("add edge", paper1["title"], paper2["title"])
            
            

results = sch.get_authors(author_id_list)
for result in results:
    print(result.name)
    for paper in result["papers"]:
        if paper["abstract"] is None:
            continue
        if paper["fieldsOfStudy"] and "Computer Science" in paper["fieldsOfStudy"]:
            if db.reference(f"papers/{paper["paperId"]}").get():
                continue
            upload_paper(dict(paper))

            references = sch.get_paper_references(paper["paperId"])
            citations = sch.get_paper_citations(paper["paperId"])

            for reference in references:
                upload_paper(dict(reference), paper["paperId"], 'r')
            for citation in citations:
                upload_paper(dict(citation), paper["paperId"], 'c')
