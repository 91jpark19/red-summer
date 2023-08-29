import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm

if __name__ == "__main__":
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    device = torch.device('mps')  # Change to the appropriate device if needed
    model.to(device)

    # Input text
    city = input("Enter the city (e.g., chicago, elaine, washington): ")
    input_text = input("Enter your text: ")

    # Preprocess the input text
    input_tokens = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
    input_tokens = input_tokens.to(device)
    with torch.no_grad():
        input_output = model(**input_tokens)
    input_embedding = input_output.last_hidden_state.mean(dim=1)  # Average pooling

    # Folder containing Feather files
    folder_path = '/Volumes/T7/chroniclingamerica/redsummer-keyword-article/'
    feather_folder_path = os.path.abspath(os.path.join(folder_path, city))
    # Get list of Feather files in the folder
    feather_files = [f for f in os.listdir(feather_folder_path) if f.endswith('.feather')]

    # Create the output folder if it doesn't exist
    output_folder = os.path.join('/Volumes/T7/chroniclingamerica/redsummer-keyword-article/', city+'_cosine_similarity')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each Feather file and calculate cosine similarity
    for file_name in tqdm(feather_files, desc="Processing files"):
        file_path = os.path.join(feather_folder_path, file_name)

        # Read Feather file into a DataFrame
        df = pd.read_feather(file_path)

        similarities = []

        # Calculate cosine similarity for each row in the DataFrame
        for idx, row in df.iterrows():
            text = row['sent']
            text_tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            text_tokens = text_tokens.to(device)
            with torch.no_grad():
                text_output = model(**text_tokens)
            text_embedding = text_output.last_hidden_state.mean(dim=1)  # Average pooling
            cosine_sim = cosine_similarity(input_embedding.cpu(), text_embedding.cpu()).item()
            similarities.append((row['date'], row['sent'], cosine_sim))

        # Create a DataFrame to store results
        result_df = pd.DataFrame(similarities, columns=['date', 'sent', 'sim'])

        # Save the result_df DataFrame to a Feather file in the output folder
        output_file = os.path.join(output_folder, file_name.replace('.feather', '_sim.feather'))
        result_df.to_feather(output_file)

        # print("Results saved to:", output_file)
