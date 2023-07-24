import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

def get_bert_embeddings(texts, model_name='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()

    # Tokenize input texts and convert to tensors
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state

def calculate_cosine_similarity(df, query_string, text_column):
    # Get BERT embeddings for the dataframe column and the query string
    df_embeddings = get_bert_embeddings(df[text_column])
    query_embeddings = get_bert_embeddings([query_string], text_column)

    # Calculate cosine similarity
    similarities = cosine_similarity(query_embeddings, df_embeddings)

    # Convert similarity scores to a pandas DataFrame
    result_df = pd.DataFrame({'Similarity': similarities.squeeze()})

    # Concatenate the similarity scores with the original DataFrame
    df_result = pd.concat([df, result_df], axis=1)

    return df_result

# Example usage
if __name__ == "__main__":
    # Sample DataFrame
    data = {
        'Text': ['I love coding with Python',
                 'Machine learning is fascinating',
                 'Data science is the future'],
    }
    df = pd.DataFrame(data)

    # Query string
    query_string = 'Python is a powerful language for coding'

    # Calculate similarity
    result = calculate_cosine_similarity(df, query_string, 'Text')
    print(result)
