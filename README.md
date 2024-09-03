# Custom Search Engine with APPNP Model and SNU Data Science Citation Network

This project is a search engine built using the APPNP model and a custom citation network based on SNU Data Science papers.

## Installation

1. **Install Required Packages**

   Before running the project, install the necessary packages by executing the following command:

   ```bash
   pip install -r requirements.txt
   ```
2. **Database Configuration**

    - Create your own database.
    - Replace the existing database credentials with your own in the relevant files.

## Data Collection

3. **Paper Collection**

    To populate the database with papers (SNU DS papers, their citations, and references), run the collect_papers.py script:

    ```bash
    python collect_paper.py
    ```
    This script uses the Semantic Scholar API to gather the required data.

## Optional Steps

4. **Feature Vector (Optional)**
    
    To generate feature vectors, run the feature_vector.py script:

    ```bash
    python feature_vector.py
    ```
    This will create a feature_vectors.csv file. Alternatively, you can use the existing feature_vectors.csv file provided.

5. **Citation Network Visualization (Optional)**
    
    You can visualize the citation network by running the networkX.py script:

    ```bash
    python networkX.py
    ```
6. **Train APPNP Model (Optional)**
    
    To train the APPNP model and generate feature embeddings, run the ppnp_pytorch.py script:

    ```bash
    python ppnp_pytorch.py
    ```
    - This script will first convert the citation network to a NetworkX graph, then to a sparse graph for training the APPNP model.
    - It outputs the node embedding vectors to final_embeddings.npy.
    - You can modify hyper-parameters in the script as needed.
    - If preferred, you can use the existing trained final_embeddings.npy file.

## Feature Embedding

7. **Generate Feature Embedding**
    
    To create feature embeddings from the final node embeddings, run the feature_embedding.py script:
    
    ```bash
    python feature_embedding.py
    ```
    This will generate the feature_embedding.npy file.

## Run the Search Engine

8. **Run Flask Server**
    
    Finally, to start the search engine, run the Flask server using:

    ```bash
    python app.py
    ```

This `README.md` file provides a structured guide for users to set up and run your search engine project, with clear instructions for each step and optional enhancements.



