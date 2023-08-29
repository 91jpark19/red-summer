import os
import warnings
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
import transformers
# from sentence_transformers import SentenceTransformer, util
import torch
from transformers import BertTokenizer, BertModel, util
# from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize

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

# def grouping_sentences(text):
#     text=str(text)
#     text=sent_tokenize(text)
#     group_size=5
#     grouped_lists=[]
#     for i in range(0, len(text), group_size):
#         group=text[i:i+group_size]
#         group=[string for string in group if any(c.isalpha() for c in string)]
#         grouped_lists.append(group)
#     return grouped_lists   

def grouping_sentences(text):
    sentences=sent_tokenize(text)
    grouped=[sentences[i:i+5] for i in range(0, len(sentences), 5)]
    return grouped


if __name__ == "__main__":
    # Ignore warnings
    warnings.filterwarnings("ignore")
    model_name = "bert-base-uncased"
    model = transformers.BertModel.from_pretrained(model_name)
    device = torch.device('mps')
    model.to(device)

    # Prompt for city
    city = input("Enter the city (e.g., chicago, elaine, washington): ")
    
    # Prompt for year
    year = input("Enter the year (e.g., 1919-07-27, 1919-09-30, 1919-07-19): ")

    # Prompt for user input
    user_input = input("Enter your text: ")
    input_vector = model.encode(user_input)
    
    # Path to the folder containing CSV files
    csv_folder_path = "/Volumes/T7/chroniclingamerica/redsummer-keyword/"
    new_csv_folder_path = os.path.abspath(os.path.join(csv_folder_path, os.pardir, "cosine_similarity_5sentence", city))

    # Create the new folder if it doesn't exist
    os.makedirs(new_csv_folder_path, exist_ok=True)

    # Iterate through CSV files in the folder
    for file_name in tqdm(os.listdir(csv_folder_path + city), desc="Processing files", position=0, leave=True):
        if file_name.endswith(".csv") and file_name[0].isalnum():
            file_path = os.path.join(csv_folder_path + city, file_name)
            df = pd.read_csv(file_path, encoding='latin1', usecols=['date', 'text'])
            df['pub_date'] = pd.to_datetime(df['date'])
            newdf = df[df['pub_date'] >= year]
            string_to_analyze=' '.join(newdf['text'])
            # grouped_texts = grouping_sentences(newdf['text'])
            if newdf.shape[0]==0:
                continue
            else:
                # grouped_cosine=[]
                # grouped_article=[]
                # grouped_date=[]
                tokenized_data=[]
                for idx, val in newdf.iterrows():
                    # interim_date=[]
                    # grouped_texts=grouping_sentences(val['text'])
                    # interim_date=[val['pub_date']]*len(grouped_texts)
                    date_for_df=val['pub_date']
                    text_for_df=val['text']
                    cosine_for_df=[]
                    grouped_texts=grouping_sentences(text_for_df)
                    for texts in grouped_texts:
                        try:
                            texts_cleaned = [space_newline(text) for text in texts]
                            combined_sentences=' '.join(texts_cleaned)
                            # embeddings = model.encode(combined_sentences)
                            # cosine_similarity_scores = cosine_similarity([input_vector], embeddings)
                            # cosine_for_df.append(cosine_similarity_scores[0])
                            tokenized_data.append({'sent': combined_sentences, 'date':date_for_df})
                            # grouped_cosine.append(cosine_similarity_scores[0])
                            # grouped_article.append(texts)
                            # grouped_date.append([val['date']]*len(grouped_texts))
                            # grouped_date.append(interim_date)
                            # grouped_date.append(val['date'])
                        except:
                            continue
                grouped_df=pd.DataFrame(tokenized_data)
                sentence_embedding=model.encode(grouped_df['sent'], convert_to_tensor=True)
                cosine_scores = util.pytorch_cos_sim(input_vector.reshape(1, -1), sentence_embedding)
                grouped_df['sim']=cosine_scores[0]
                # cosine_scores=cosine_similarity(input_vector, sentence_embedding)
                # print(len(grouped_article), len(grouped_cosine), len(grouped_date))
                # article_list = [item for sublist in grouped_article for item in sublist]
                # cosine_list = [item for sublist in grouped_cosine for item in sublist]
                # date_list = [item for sublist in grouped_date for item in sublist]
                # print(len(article_list), len(cosine_list), len(date_list))
                # grouped_df=pd.DataFrame.from_dict({'article': article_list})
                # grouped_df['sim']=cosine_list
                # grouped_df['date']=date_list
                new_file_path = os.path.join(new_csv_folder_path, file_name.replace(".csv", "_bert_sentence_cosine_similarity.csv"))
                grouped_df.to_csv(new_file_path, index=False)
