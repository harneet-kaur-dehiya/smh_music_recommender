__author__ = 'rbl'

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('test.html')

@app.route('/hello', methods=['POST'])
def hello():
    song_query = request.form['first_name']
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import warnings
    sns.set()
    data = pd.read_csv("spotipy.csv")
    df = data.drop(columns=['id', 'name','release_date'])
    from sklearn.preprocessing import MinMaxScaler
    datatypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    normarization = data.select_dtypes(include=datatypes)
    for col in normarization.columns:
        MinMaxScaler(col)
    normarization = normarization.sample(frac = 1)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=4)
    features = kmeans.fit_predict(normarization)
    data['features'] = features
    warnings.filterwarnings("ignore")
    class Spotify_Recommendation():
        def __init__(self, dataset):
            self.dataset = dataset
        def recommend(self, songs, amount=1):
            distance = []
            song = self.dataset[(self.dataset.name.str.lower()==songs.lower())].head(1).values[0]
            print("Artists: ",song[1])
            rec = self.dataset[self.dataset.name.str.lower() != songs.lower()]
            for songs in rec.values:
                d = 0
                for col in np.arange(len(rec.columns)):
                    if not col in [1, 5, 11, 13, 17]:
                        d = d + np.absolute(float(song[col]) - float(songs[col]))
                distance.append(d)
            rec['distance'] = distance
            rec = rec.sort_values('distance')
            columns = ['artists', 'name']
            print("Artists: ",song[1])
            return rec[columns][:amount],song[11],song[1]
    recommendations = Spotify_Recommendation(data)
    recs, songname, artistname=recommendations.recommend(song_query, 10)
    listtt=recs.values.tolist()
    table=f'<!DOCTYPE html><html><head><link rel="stylesheet" type="text/css" href="static/test.css"><title>Music Recommendation</title></head><body>'
    single_char="'"
    artistname=artistname[1:-1]
    artistname= ''.join([ch for ch in artistname if ch != single_char])
    table+=f'<h2>Song Recommendations for "{songname}" by {artistname}</h2><table class="center"><thead><tr><th>Song</th><th>Artist</th></tr></thead><tbody>'
    for i in range (0,10):
        abc=listtt[i][0][1:-1]
        abc = ''.join([ch for ch in abc if ch != single_char])
        xyz=listtt[i][1]
        table+=f'<td>{xyz}</td><td>{abc}</td></tr>'
    table+=f'</tbody></table><br/><a href="/">Back Home</a>'
    return table
if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 3000)