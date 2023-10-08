import pickle
from flask import Flask, request, Response, jsonify
from tmdbv3api import Movie, TMDb
import os
import dotenv
import pandas as pd

dotenv.load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
app = Flask(__name__)
movie = Movie()
tmdb = TMDb()
tmdb.api_key = TMDB_API_KEY
tmdb.language = 'ko-KR'

movies = pickle.load(open('movies.pickle', 'rb'))
cosine_sim = pickle.load(open('cosine_sim.pickle', 'rb'))

@app.route('/')
def index():
    hi = 'hello world'
    print(__file__)
    print("abd")
    print(pd.__version__)
    return hi

@app.route('/movie', methods=["post"])
def test():
    title = request.get_json()['title']
    movie_list = movies['title'].values
    images, titles = get_recommendations(title)
    print(titles)
    print(images)
    return jsonify({'title':titles, "image":images})

def get_recommendations(title):
    # 영화 제목을 통해서 전체 데이터 기준 그 영화의 index 값을 얻기
    idx = movies[movies['title'] == title].index[0]

    # 코사인 유사도 매트릭스 (cosine_sim) 에서 idx 에 해당하는 데이터를 (idx, 유사도) 형태로 얻기
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 코사인 유사도 기준으로 내림차순 정렬
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # 자기 자신을 제외한 10개의 추천 영화를 슬라이싱
    sim_scores = sim_scores[1:11]
    
    # 추천 영화 목록 10개의 인덱스 정보 추출
    movie_indices = [i[0] for i in sim_scores]
    
    # 인덱스 정보를 통해 영화 제목 추출
    images = []
    titles = []
    for i in movie_indices:
        id = movies['id'].iloc[i]
        details = movie.details(id)
        
        image_path = details['poster_path']
        if image_path:
            image_path = 'https://image.tmdb.org/t/p/w500' + image_path
        else:
            image_path = 'no_image.jpg'

        images.append(image_path)
        titles.append(details['title'])

    return images, titles


if __name__ == '__main__': # 모듈이 아니라면, 웹서버를 구동시켜라!
    app.run(host="127.0.0.1", port="4000", debug=False)

    