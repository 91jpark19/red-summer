import pandas as pd
import os
from tqdm import tqdm
from multiprocessing import Pool

directory = '/Volumes/T7/chroniclingamerica/redsummer-1917-1921/'

def search_dataframe(args):
    file, folder, keywords = args

    # Read the feather file
    file_path = os.path.join(directory, file)
    df = pd.read_feather(file_path)

    # Subset the DataFrame based on the entered keywords
    df['text'] = df['text'].astype(str)
    subset = df[df['text'].str.contains('|'.join(keywords), case=False)]

    # Save the subset as a Feather file in the specified folder
    subset.reset_index(drop=True, inplace=True)  # Reset index for Feather
    subset.to_feather(os.path.join(folder, os.path.splitext(file)[0] + '.feather'))

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
