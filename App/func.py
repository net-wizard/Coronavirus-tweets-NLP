from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import warnings
import pickle
filename="finalized_model.pkl"
model=pickle.load(open(filename,"rb"))
import numpy as np
warnings.filterwarnings('ignore')

def clean(df):

    #custom cleaning

    #urls
    df['t'] = df['t'].replace(r'http\S+', ' ', regex=True).replace(r'www\S+', ' ', regex=True)

    #@mention
    df['t'] = df['t'].replace(r'@[A-Za-z0-9]+', ' ', regex=True)

    #Emoji
    df['t'] = df['t'].apply(lambda x: [y.encode('ascii', 'ignore').decode('ascii') for y in x.split()])

    #regular cleaning

    # remove punctuation
    df['t'] = df['t'].astype(
    str).str.replace(r'[^\w\d\s]', ' ')

    # replace multiple spaces with single space in between text
    df['t'] = df['t'].str.replace(r'\s+', ' ')

    # change to lower case
    df['t'] = df['t'].astype(str).str.lower()

    #remove numbers
    df['t'] = df['t'].astype(str).str.replace(r'\d+', ' ')

    # remove stopwords
    stop = stopwords.words('english')
    stop.append('new')
    stop.append('yorkers')
    stop.append('york')
    stop.append('newyork')
    df['t'] = df['t'].apply(lambda x: [item for item in x.split() if item not in stop])

    # Lemmatizing
    lemmatizer = WordNetLemmatizer()
    df['t'] = df['t'].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])

    return df['t']


def prediction(text):
    df = pd.DataFrame([[text]], columns=list('t'))
    tweet=clean(df)
    tweet = tweet.astype(str).str.strip()
    print(tweet) 
    result = model.predict(tweet)
    return result

while True:
    var = input("Text: ")
    print(prediction(var))
    print("press ctrl+c to stop else give input after Text:")
