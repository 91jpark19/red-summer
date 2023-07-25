import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(texts):
    # Tokenize the input texts and get their embeddings
    embeddings = model.encode(texts)
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix 

def space_newline(input_string):
    words = input_string.replace('\n', ' ').split(' ')
    filtered_words = [word for word in words if word]
    return ' '.join(filtered_words)

if __name__ == "__main__":
    model_name = "bert-base-nli-mean-tokens"
    model = SentenceTransformer(model_name)

    # Prompt for city
    city = input("Enter the city (e.g., chicago, elaine, washington): ")
    
    # Prompt for year
    year = input("Enter the year (e.g., 1917-01-01): ")

    # Prompt for user input
    user_input = input("Enter your text: ")
    input_vector=model.encode(user_input)
    
    # Path to the folder containing CSV files
    csv_folder_path = "/Volumes/T7/chroniclingamerica/redsummer-keyword/"
    new_csv_folder_path = os.path.abspath(os.path.join(csv_folder_path, os.pardir, "cosine_similarity", city))

    # Create the new folder if it doesn't exist
    os.makedirs(new_csv_folder_path, exist_ok=True)

    # Iterate through CSV files in the folder
    for file_name in tqdm(os.listdir(csv_folder_path + city), desc="Processing files", position=0, leave=True):
        if file_name.endswith(".csv"):
            file_path = os.path.join(csv_folder_path + city, file_name)
            df = pd.read_csv(file_path, encoding='latin1')
            df['date'] = pd.to_datetime(df['date'])
            newdf = df[df['date'] >= year]
            
            texts=[]
            for text in newdf['text']:
                texts.append(space_newline(text))
            if len(texts) == 0:
                continue
            else:
                embeddings=model.encode(texts)
                cosine_similarity_scores = cosine_similarity([input_vector], embeddings)
                newdf["similarity"] = list(cosine_similarity_scores[0])
                new_file_path = os.path.join(new_csv_folder_path, file_name.replace(".csv", "_cosine_similarity.csv"))
                newdf.to_csv(new_file_path, index=False)
