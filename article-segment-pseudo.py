import os
import pandas as pd
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from nltk.tokenize import word_tokenize

def space_newline(input_string):
    words = input_string.replace('\n', ' ').split(' ')
    filtered_words = [word for word in words if word]
    return ' '.join(filtered_words)

def grouping_sentences(text):
    sentences = word_tokenize(text)
    grouped = [sentences[i:i+30] for i in range(0, len(sentences), 30)]
    return grouped

def process_file(file_path, new_folder_path, year, end_year):
    df = pd.read_feather(file_path)#, encoding='latin1', usecols=['date', 'text'])
    df['pub_date'] = pd.to_datetime(df['date'])
    newdf = df[(df['pub_date'] >= year) & (df['pub_date'] < end_year)]
    
    if newdf.shape[0] == 0:
        return
    
    tokenized_data = []
    for _, val in newdf.iterrows():
        date_for_df = val['pub_date']
        text_for_df = val['text']
        grouped_texts = grouping_sentences(text_for_df)
        for texts in grouped_texts:
            try:
                texts_cleaned = [space_newline(text) for text in texts]
                combined_sentences = ' '.join(texts_cleaned)
                tokenized_data.append({'sent': combined_sentences, 'date': date_for_df})
            except:
                continue
    grouped_df = pd.DataFrame(tokenized_data)
    new_file_path = os.path.join(new_folder_path, os.path.basename(file_path).replace(".feather", "_pseudo.feather"))
    grouped_df.reset_index(drop=True, inplace=True)  # Reset index for feather
    grouped_df.to_feather(new_file_path)


if __name__ == "__main__":
    city = input("Enter the city (e.g., chicago, elaine, washington): ")
    year = input("Enter the year (e.g., 1919-07-27, 1919-09-30, 1919-07-19): ")
    end_year = input("Enter the end year (e.g., 1920-07-27, 1920-09-30, 1920-07-19): ")
    csv_folder_path = "/Volumes/T7/chroniclingamerica/redsummer-keyword/"
    new_folder_path = os.path.abspath(os.path.join(csv_folder_path, os.pardir, "redsummer-keyword-pseudo", city))
    os.makedirs(new_folder_path, exist_ok=True)
    
    partial_process_file = partial(process_file, new_folder_path=new_folder_path, year=year, end_year=end_year)
    
    csv_files = [os.path.join(csv_folder_path + city, file_name) for file_name in os.listdir(csv_folder_path + city) if file_name.endswith(".feather") and file_name[0].isalnum()]
    
    num_processes = min(cpu_count(), len(csv_files))
    
    with Pool(num_processes) as pool:
        list(tqdm(pool.imap(partial_process_file, csv_files), total=len(csv_files), desc="Processing files"))
