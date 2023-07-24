import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(texts, model):
    # Tokenize the input texts and get their embeddings
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Use mean pooling for sentence embeddings
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

if __name__ == "__main__":
    # Load the pre-trained BERT model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    #Prompt for city
    city = input("Enter the city (e.g., chicago, elaine, washington): ")
    
    #Prompt for year
    year = input("Enter the year (e.g., 1917-01-01): ")

    # Prompt for user input
    user_input = input("Enter your text: ")

    # Path to the folder containing CSV files
    csv_folder_path = "/Volumes/T7/chroniclingamerica/redsummer-keyword/"
    new_csv_folder_path = os.path.abspath(os.path.join(csv_folder_path, os.pardir, "cosine_similarity_", city))

    # Create the new folder if it doesn't exist
    os.makedirs(new_csv_folder_path, exist_ok=True)

    # Iterate through CSV files in the folder
    for file_name in tqdm(os.listdir(csv_folder_path + city)):
        if file_name.endswith(".csv"):
            file_path = os.path.join(csv_folder_path + city, file_name)
            df = pd.read_csv(file_path)
            df['date']=pd.to_datetime(df['date'])
            newdf=df[df['date']>=year]

            # Assuming the column containing text is named 'text'
            texts = newdf['text'].tolist()

            # Append the user input to the list of texts
            texts.append(user_input)

            # Calculate cosine similarity
            similarity_matrix = calculate_cosine_similarity(texts, model)

            # Get the similarity score between user input and each text in the CSV
            similarity_scores = similarity_matrix[-1][:-1]  # Exclude the last row (user input)
            max_similarity_score = np.max(similarity_scores)

            # Get the index of the most similar text
            most_similar_idx = np.argmax(similarity_scores)

            # Get the text corresponding to the most similar index
            most_similar_text = texts[most_similar_idx]

            # Append cosine similarity values to a new column in the DataFrame
            newdf["Cosine_Similarity"] = similarity_scores

            # Save the DataFrame back to the CSV file with the appended column
            new_file_path = os.path.join(new_csv_folder_path, file_name.replace(".csv", "_cosine_similarity.csv"))
            newdf.to_csv(new_file_path, index=False)

            # Output the results
            print(f"File: {file_name}")
            print(f"Most similar text: {most_similar_text}")
            print(f"Cosine similarity score: {max_similarity_score}")
            print(f"CSV file with cosine similarity saved at: {new_file_path}")
            print("=" * 50)
