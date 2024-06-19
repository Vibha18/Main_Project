from flask import Flask,redirect,url_for,render_template,request,jsonify
import mysql.connector

app=Flask(__name__)
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import random
import csv

def get_mysql_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root@123",
        database="youtube"
    )

import pandas as pd
df=pd.read_csv('training_comments.csv')
X=df['text'].values.astype('U')
y=df['sentiment']

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=6)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

text_clf=Pipeline([('vect',TfidfVectorizer()),('clf',MultinomialNB())])
text_clf.fit(X_train,y_train)

from sklearn.metrics import accuracy_score
predicted=text_clf.predict(X_test)
print("Accuracy ",accuracy_score(y_test,predicted))

API_KEY = 'AIzaSyAGH6Zrs1fMPijKyPWRxyVXiVCuBpfK3Eg'

def comments(video_id):
    from googleapiclient.discovery import build
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    video_id=video_id[-11:]
    video_response = youtube.videos().list(
        part='snippet',
        id=video_id
    ).execute()
    video_snippet = video_response['items'][0]['snippet']
    uploader_channel_id = video_snippet['channelId']
    comments = []
    nextPageToken = None
    while len(comments) < 20000:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100,
            pageToken=nextPageToken
        )
        response = request.execute()
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            if comment['authorChannelId']['value'] != uploader_channel_id:
                comments.append(comment['textDisplay'])
        nextPageToken = response.get('nextPageToken')

        if not nextPageToken:
            break
    import pandas as pd
    df1 = pd.DataFrame(comments, columns=['text'])
    df1.to_csv("fetched_comments.csv")
    v1=df1.iloc[:,0:]
    import nltk
    import re
    data_file=v1
    nltk.download('stopwords')
    from nltk.corpus import stopwords,wordnet
    stop_words = stopwords.words('english')
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    wlt = WordNetLemmatizer()
    import pandas as pd
    def remove_URL(text):
        url = re.compile(r'https*')
        return url.sub(r'', text)
    
    data_file['text'] = data_file['text'].apply(lambda x: remove_URL(x))
    def remove_html(text):
        html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        return re.sub(html, '', text)

    data_file['text'] = data_file['text'].apply(lambda x: remove_html(x))
    import string

    def remove_punct(text):
        table = str.maketrans('', '', string.punctuation)
        return text.translate(table)

    data_file['text'] = data_file['text'].apply(lambda x: remove_punct(x))

    import nltk
    nltk.download('punkt')
    data_file['text'] = data_file['text'].apply(word_tokenize)

    data_file['text'] = data_file['text'].apply(lambda x: [word.lower() for word in x])

    data_file['text'] = data_file['text'].apply(lambda x: [word for word in x if word not in stop_words])

    import nltk
    nltk.download('averaged_perceptron_tagger')
    data_file['text'] = data_file['text'].apply(nltk.tag.pos_tag)

    import nltk
    nltk.download('wordnet')
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    data_file['text'] = data_file['text'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])

    data_file['text'] = data_file['text'].apply(lambda x: [wlt.lemmatize(word, tag) for word, tag in x])
    data_file['text'] = [' '.join(map(str, l)) for l in data_file['text']]

    data_file.to_csv('preprocessed_comments.csv', index=False)
    p1=[]
    c1=0
    c2=0
    c3=0
    d={}
    l=[]
    for c in data_file['text']:
        new_comment = [c]
        predicted_sentiment = text_clf.predict(new_comment)
        p1.append(predicted_sentiment[0])

        d[c]=predicted_sentiment[0]
        if(predicted_sentiment[0]=='Negative'):
            l.append([c,"Negative"])
            c1=c1+1
        if(predicted_sentiment[0]=='Positive'):
            l.append([c,"Positive"])
            c2=c2+1
        if(predicted_sentiment[0]=='Neutral'):
            l.append([c,"Neutral"])
            c3=c3+1
    with open('preprocessed_sentiment.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['text', 'sentiment'])
        writer.writerows(l)
    rating=(c2/(c1+c2+c3))*5
    return(rating)

def playlist_comments(playlist_id):
    from googleapiclient.discovery import build
    youtube = build('youtube', 'v3', developerKey=API_KEY)

    playlist_videos = youtube.playlistItems().list(
        part='snippet',
        playlistId=playlist_id,
        maxResults=50  
    ).execute()

    video_ids = [item['snippet']['resourceId']['videoId'] for item in playlist_videos['items']]

    def fetch_comments(video_id):
        comments = []
        nextPageToken = None
        while len(comments) < 20000:  
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100,  
                pageToken=nextPageToken
            )
            response = request.execute()
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)
            nextPageToken = response.get('nextPageToken')

            if not nextPageToken:
                break
        return comments

    all_comments = []
    for video_id in video_ids:
        video_comments = fetch_comments(video_id)
        all_comments.extend(video_comments)

    import pandas as pd
    df1 = pd.DataFrame(all_comments, columns=['text'])
    df1.to_csv("playlist.csv")
    v1=df1.iloc[:,0:]
   
    import nltk
    import re
    data_file=v1
    nltk.download('stopwords')
    from nltk.corpus import stopwords,wordnet
    stop_words = stopwords.words('english')
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    wlt = WordNetLemmatizer()
    import pandas as pd
    def remove_URL(text):
        url = re.compile(r'https*')
        return url.sub(r'', text)

    data_file['text'] = data_file['text'].apply(lambda x: remove_URL(x))

    def remove_html(text):
        html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        return re.sub(html, '', text)

    data_file['text'] = data_file['text'].apply(lambda x: remove_html(x))

    import string

    def remove_punct(text):
        table = str.maketrans('', '', string.punctuation)
        return text.translate(table)

    data_file['text'] = data_file['text'].apply(lambda x: remove_punct(x))

    import nltk
    nltk.download('punkt')
    data_file['text'] = data_file['text'].apply(word_tokenize)

    data_file['text'] = data_file['text'].apply(lambda x: [word.lower() for word in x])

    data_file['text'] = data_file['text'].apply(lambda x: [word for word in x if word not in stop_words])

    import nltk
    nltk.download('averaged_perceptron_tagger')
    data_file['text'] = data_file['text'].apply(nltk.tag.pos_tag)

    import nltk
    nltk.download('wordnet')
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    data_file['text'] = data_file['text'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])

    data_file['text'] = data_file['text'].apply(lambda x: [wlt.lemmatize(word, tag) for word, tag in x])
    data_file['text'] = [' '.join(map(str, l)) for l in data_file['text']]
  
    p1=[]
    c1=0
    c2=0
    c3=0
    d={}
    for c in data_file['text']:
        new_comment = [c]
        predicted_sentiment = text_clf.predict(new_comment)
        p1.append(predicted_sentiment[0])

        d[c]=predicted_sentiment[0]
        if(predicted_sentiment[0]=='Negative'):
            c1=c1+1
        if(predicted_sentiment[0]=='Positive'):
            c2=c2+1
        if(predicted_sentiment[0]=='Neutral'):
            c3=c3+1
    rating=(c2/(c1+c2+c3))*5
    return(rating)

@app.route('/')
def welcome():
    return render_template('front.html')

def process_youtube_url(youtube_url):
    video_id = youtube_url.split('=')[-1]  
    return video_id

@app.route('/submit', methods=['GET'])
def submit():
    youtube_url = request.args.get('url')
    video_id = process_youtube_url(youtube_url)
    if len(video_id)==11:
        k=comments(video_id)
    else:
        k=playlist_comments(video_id)
    k=round(k,1)
    m=str(k)
    return m

@app.route('/recommend', methods=['POST'])
def recommend():
    connection = get_mysql_connection()
    if connection.is_connected():
        selected_value = request.json.get('language')
        mycursor = connection.cursor(dictionary=True)
        query = "SELECT url, rating FROM {} ORDER BY rating DESC LIMIT 3".format(selected_value)
        mycursor.execute(query)
        top_videos = mycursor.fetchall()
        connection.close()
        return jsonify(result=top_videos)
    else:
        return "Error: Failed to connect to MySQL database"

    
if __name__=='__main__':  
    app.run(debug=True)