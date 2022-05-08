import pandas as pd
import numpy as np
import ast
import gensim
import nltk
import re
import pickle
from itertools import product
from nltk.corpus import stopwords
# nltk.download('stopwords')
stop_words = stopwords.words('english')

def load():
    
    md = pd.read_csv('data/movies_metadata.csv')
    print("load md")
    df_keywords = pd.read_csv('data/keywords.csv')
    print('keywords')
    df_credits = pd.read_csv('data/credits.csv')
    print('credits')
    return md, df_keywords, df_credits

def convert(text):
    
    text = ast.literal_eval(text)
    return text

def convert_md_id(mid):
    
    try:
        mid = int(mid)
        return mid
    except:
        return mid
    
def extract_keywords(keyword_dict):
    
    keyword_dict = [i['name'] for i in  keyword_dict]  
    return keyword_dict


def clean_overview(text):
    
#     print(text)
    text = text.lower()
    text = re.sub(r'\-', " ",text)
    text = re.sub(r'\.', " ",text).replace(","," ").replace("\'","'")
    text = text.replace("(","",).replace(")","").replace('"',"").split()
    text = [w for w in text if w not in stop_words]
    
    return text

def lower_title(title):
    title= title.lower()
    return title

def credits_preprocessing(df_credits):
    
    df_credits['crew'] = df_credits['crew'].apply(convert)
    df_credits['Director'] = None
    for index, val in enumerate(df_credits['crew']):
        print(index)
        if len(val)!= 0:
            if val[0]['job'] == 'Director':
                print(val[0])
                df_credits['Director'].iloc[index] = val[0]['name']

    df_credits.dropna(subset = [ 'Director' ], inplace = True)
    df_credits.reset_index(inplace = True)
    df_credits.drop(columns = ['index'], inplace = True)

    print('credits preprocessing done')
    return df_credits

def keyword_preprocessing(df_keywords):
    
    keywords_duplicate_id = df_keywords[df_keywords['id'].duplicated(keep = 'first')]['id']
    keywords_duplicate_id_index = keywords_duplicate_id.index.tolist()
    df_keywords.drop(keywords_duplicate_id_index, inplace = True)
    
    df_keywords['keywords'] = df_keywords['keywords'].apply(convert)
    df_keywords["id"] = df_keywords["id"].astype(int).astype(object)
    print('keywords preprocessing done')
    return df_keywords
    
def md_preprocessing(md):
    
    x = md[md['id'].duplicated(keep='first')]['id']
    duplicate_id_index = x.index.tolist()
    md.drop(duplicate_id_index, inplace=True)
    
    md['id'] = md['id'].apply(convert_md_id)
    
    drop_indices = []
    for index, val  in enumerate(md['id']):
        if type(val)  == str:
            drop_indices.append(index)
            
    md.drop(drop_indices, axis = 0, inplace =True)    
    md.reset_index(inplace= True)
    md.drop(columns =['index'], inplace = True)    
    print('md preprocessing done')
    return md
    
    
def merge_keywords(df_keywords, md):        
    
    md_updated = md.merge(df_keywords, on='id', how ='left')
    md_updated.dropna(subset=['keywords'], inplace = True)
    md_updated.reset_index(inplace = True)
    md_updated.drop(columns=['index'], inplace = True)
    
    md_updated['keywords'] =md_updated['keywords'].apply(extract_keywords)
    
    drop_index=[]
    for i in range(len(md_updated)):
        if len(md_updated['keywords'].iloc[i]) == 0:
            drop_index.append(i)
    print(len(drop_index))
    md_updated.drop(drop_index, inplace = True)
    md_updated.reset_index(inplace = True)
    md_updated.drop(columns=['index'], inplace =True)
    print('merged keywords with md dataframe')
    return md_updated
    
    
def create_embedding_matrix(md_updated):    
    
    em_md = md_updated.dropna(subset=['overview', 'title', 'keywords'])
    em_md['overview'] = em_md['overview'].apply(clean_overview)
    em_md['title'] = em_md['title'].apply(lower_title)
    em_md.reset_index(inplace= True)
        
    model = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format("glove840B.bin", 
        binary = True)

    documents = list(em_md['overview'])
    dictionary = gensim.corpora.Dictionary(documents)
    print('document created')

    doc_list = []
    for i in em_md['overview']:
        doc_list.append(dictionary.doc2bow(i))
    print('doc_dict created')

    tfidf = gensim.models.TfidfModel(doc_list)
    tfidf_list = [tfidf[i] for i in doc_list]
    print('tfidf_list created')

    termsim_index = gensim.similarities.WordEmbeddingSimilarityIndex(model)
    print('embedding index created')
    termsim_matrix = gensim.similarities.SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf)
    print('matrix created')
       
    return em_md, tfidf_list, termsim_matrix


def collection_recommendations(em_md, movie_index, movie_collection, score_list):
    
    collection_list = []
    print('entered')

    for i, collection in enumerate(em_md['belongs_to_collection']):

        if type(collection) == dict:
            if collection == movie_collection:
                print(collection['name'])
                collection_list.append(i)
    
    collection_list = collection_list.remove(movie_index)
            
    top10_collection_list = sorted(collection_list[1:11], 
                                   key = lambda x: em_md['popularity'].astype(float).iloc[x],  
                                   reverse = True)

    top10_collection_list = [em_md['title'].iloc[i] for i in top10_collection_list][:10]
#         print("movies from same collection: ", top10_collection_list)
#         print('\n')
    return top10_collection_list
    

def director_recommendations(df_credits, em_id, movie_id):
    
    director_movie_ids = []
    director_movies = []
    movie_director = df_credits[df_credits['id'] == movie_id]['Director'].tolist()
    
    if len(movie_director) != 0:
        for mid, director in enumerate(df_credits['Director']):
            if director == movie_director[0]:
                director_movie_ids.append(df_credits['id'].iloc[mid])
                
        director_movie_ids = director_movie_ids.remove(movie_id)
#         print(director_movie_ids)
        
        if len(director_movie_ids) != 0:
             director_movies = [em_md[em_md['id'] == i]['title'].tolist()[0] 
                                for i in director_movie_ids
                                if i in em_md['id']]
#             director_movies = sorted(director_movies, key = lambda x: em_md[em_md['title'] == x]['popularity'].tolist()[0], reverse = True)                      

#         if len(director_movies)!=0:
#             print("MOVIES FROM SAME DIRECTOR: ", director_movies)    
    return director_movies[:10]


def get_recommendations(em_md, df_credits, termsim_matrix, tfidf_list, movie_name):
    
    movie_index = em_md[em_md['title'] == movie_name].index[0]
    try:
        movie_index = em_md[em_md['title'] == movie_name].index[0]
        movie_index = em_md[em_md['title'] == movie_name].index[0]
        movie_id = em_md['id'].iloc[movie_index]
        movie_collection = em_md['belongs_to_collection'].iloc[movie_index]

        score_list = []
        top10_collection_list = []

        
        director_movies = director_recommendations(df_credits, em_md, movie_id)

        for i in range(len(em_md)):
            score_list.append((i,
                termsim_matrix.inner_product(tfidf_list[movie_index], tfidf_list[i], 
                                                          normalized=(True, True))))
            
        score_list = sorted(score_list, key = lambda x : x[1], reverse= True)

        print(movie_collection)
        print(len(score_list))
        print(len(em_md))

        if type(movie_collection) == str:
            top10_collection_list = collection_recommendations(em_md, movie_collection, score_list)
            print(top10_collection_list)
        
        top50_movie_list = sorted(score_list[1:51], 
                                  key = lambda x: em_md['popularity'].astype(float).iloc[x[0]], 
                                  reverse = True)
        top20_movie_list = [em_md['title'].iloc[i[0]] for i in top50_movie_list][:20]

        print(top20_movie_list)
        
        print(director_movies)
    
    except:
        print("Movie not found")
    
    
def main():
    
    md, df_keywords, df_credits = load()
    df_credits = credits_preprocessing(df_credits)
    df_keywords = keyword_preprocessing(df_keywords)
    md = md_preprocessing(md)
    md_updated = merge_keywords(df_keywords, md)
    em_md, tfidf_list, termsim_matrix = create_embedding_matrix(md_updated)
    
    movie_name = input("Enter movie name: ").lower()
    get_recommendations(em_md, df_credits, termsim_matrix, tfidf_list, movie_name)
    
    
if __name__ == '__main__':
    main()
    
    
