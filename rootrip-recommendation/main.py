import pickle
from flask import Flask, request, jsonify
import pandas as pd
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

tag_mapping = {
    "액티비티": 0,
    "여행": 1,
    "바다": 2,
    "산": 3,
    "관광": 4,
    "자연": 5,
    "음식": 6,
    "문화": 7,
    "도시": 8,
    "모험": 9,
    "휴양": 10,
    "풍경": 11,
    "체험": 12,
    "데이트": 13
}

def calculate_weights(tags, user_likes):
    weights = [user_likes[tag_mapping.get(tag['tag'], 0)] for tag in tags]
    return ' '.join([str(weight) for weight in weights])

def convert_to_python_type(value):
    """numpy.int64를 Python의 int로 변환"""
    if isinstance(value, np.int64):
        return int(value)
    return value

@app.route('/')
def index():
    hi = 'hello world'
    return hi

@app.route('/recommend', methods=["POST"])
def recommend_posts():
    user_likes = request.get_json()['user_likes']
    user_posts = request.get_json()['user_posts']

    # 사용자가 작성한 게시글을 데이터프레임으로 생성
    df_user_posts = pd.DataFrame(user_posts, columns=['board_id', 'title', 'hashtag'])

    recommended_posts = get_post_recommendations(user_likes, df_user_posts)
    # 'board_id' 값을 Python의 int로 변환한 후에 JSON으로 직렬화
    return jsonify({'recommended_posts': json.loads(json.dumps(recommended_posts, default=convert_to_python_type))})

def get_post_recommendations(user_likes, df_user_posts):
    # 각 게시물에 대해 가중치를 계산하여 'weights' 열에 저장
    df_user_posts['weights'] = df_user_posts['hashtag'].apply(calculate_weights, user_likes=user_likes)

    # CountVectorizer를 사용하여 단어 벡터화
    count = CountVectorizer()
    count_matrix = count.fit_transform(df_user_posts['weights'])

    # 사용자가 작성한 게시글에 대해 가중치를 계산하여 'weights' 열에 저장
    df_user_posts['weights'] = df_user_posts['hashtag'].apply(calculate_weights, user_likes=user_likes)

    # 코사인 유사도 매트릭스 계산
    user_weight_vector = count.transform([' '.join([str(user_likes[i]) for i in range(len(user_likes))])])
    similarities = cosine_similarity(user_weight_vector, count_matrix)
    similarities_with_titles = list(enumerate(similarities[0]))
    similarities_with_titles = sorted(similarities_with_titles, key=lambda x: x[1], reverse=True)
    recommended_posts = []
    for i in range(len(df_user_posts)):  # 상위 10개 게시글 추천
        post_idx = similarities_with_titles[i][0]
        recommended_posts.append({
            'board_id': df_user_posts['board_id'].iloc[post_idx]
            })
    return recommended_posts

if __name__ == '__main__':
    app.run(host="165.229.86.126", port="9500", debug=False)
