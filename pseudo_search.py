import pandas as pd
import os
from tqdm import tqdm
from multiprocessing import Pool
from nltk.tokenize import word_tokenize

directory = '/Volumes/T7/chroniclingamerica/redsummer-1917-1921/'


# def keyword_search_wrapper(args):
#     return keyword_search(*args)

def search_dataframe(args):
    file, folder, keywords = args
    date=[]
    sent_list = []
    # context_list = []
    file_path = os.path.join(directory, file)
    df=pd.read_feather(file_path)
    for idx, val in df.iterrows():
        tokenized_word = word_tokenize(val['text'].lower())
        matching_items = [(keywords, idx + i) for i, item in enumerate(tokenized_word) if keywords in item]
        to_context = []

        for num, sent in enumerate(tokenized_word):
            interim_to_context = []
            if keywords in sent:
                start_idx = max(0, num - 10)
                end_idx = min(len(tokenized_word), num + 11)
                context = [tokenized_word[i] for i in range(start_idx, end_idx)]
                interim_to_context.extend(context)
                to_context.append(interim_to_context)

        date.append(val['date'])
        sent_list.append(to_context)  # Store the surrounding tokens in sent_list
        # context_list.append(to_context)

    newdf = pd.DataFrame({'date': date, 'sent': sent_list})#, 'context': context_list})
    newdf.reset_index(drop=True, inplace=True)  # Reset index for Feather
    newdf.to_feather(os.path.join(folder, os.path.splitext(file)[0] + '.feather'))


# def search_dataframe(args):
#     file, folder, keywords = args

#     # Read the feather file
#     file_path = os.path.join(directory, file)
#     df = pd.read_feather(file_path)

#     # Subset the DataFrame based on the entered keywords
#     df['text'] = df['text'].astype(str)
#     subset = df[df['text'].str.contains('|'.join(keywords), case=False)]

#     # Save the subset as a Feather file in the specified folder
#     subset.reset_index(drop=True, inplace=True)  # Reset index for Feather
#     subset.to_feather(os.path.join(folder, os.path.splitext(file)[0] + '.feather'))

if __name__ == '__main__':
    os.chdir('/Volumes/T7/chroniclingamerica/redsummer-keyword/')
    save_folder = input("Enter the folder path to save the result: ")
    os.makedirs(save_folder, exist_ok=True)
    keywords = input("Enter keywords (comma-separated): ").split(',')
    keywords = [keyword.strip() for keyword in keywords]

    # Create a pool of worker processes
    pool = Pool()

    # Prepare the arguments for search_dataframe function
    args_list = [(file, save_folder, keywords) for file in os.listdir(directory) if file.endswith('.feather')]

    # Apply multiprocessing
    for _ in tqdm(pool.imap_unordered(search_dataframe, args_list), total=len(args_list)):
        pass

    # Close the pool of worker processes
    pool.close()
    pool.join()

    print("Search result saved successfully.")

# if __name__ == '__main__':
#     os.chdir('/Volumes/T7/chroniclingamerica/redsummer-pseudo/')
#     save_folder = input("Enter the folder path to save the result: ")
#     os.makedirs(save_folder, exist_ok=True)
#     keywords = input("Enter keywords (comma-separated): ").split(',')
#     keywords = [keyword.strip() for keyword in keywords]

#     # Create a pool of worker processes
#     pool = Pool()

#     # Prepare the arguments for keyword_search function
#     args_list = [(file, keywords, save_folder) for file in os.listdir(directory) if file.endswith('.feather')]

#     # Apply multiprocessing using the wrapper function
#     for _ in tqdm(pool.imap_unordered(keyword_search_wrapper, args_list), total=len(args_list)):
#         pass

#     # Close the pool of worker processes
#     pool.close()
#     pool.join()

#     print("Search result saved successfully.")
