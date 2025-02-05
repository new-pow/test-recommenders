import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity  # Import the cosine similarity function

# Load the data
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    'movie_id': [1, 2, 3, 1, 2, 1, 2, 3, 1, 3],
    'rating': [5, 4, 3, 3, 2, 4, 3, 5, 1, 5]
}
df = pd.DataFrame(data)
print(df)
print('\n')

# 사용자-영화 행렬로 변환
user_item_matrix = df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
print(user_item_matrix)
print('\n')

def user_based_cf(user_id, user_item_matrix, n_similar_users=2):
    user_sim = cosine_similarity(user_item_matrix)

    # 올바른 인덱스 찾기 (user_id는 1부터 시작하지만, user_sim은 0-based index)
    user_index = user_id - 1

    # 사용자 유사도 기반으로 가장 유사한 사용자 찾기
    similar_users = user_sim[user_index].argsort()[::-1][1:n_similar_users+1]  
    print("similar users:", similar_users)

    # 추천 점수 계산
    recommendations = np.zeros(user_item_matrix.shape[1])
    for similar_user in similar_users:
        recommendations += user_item_matrix.iloc[similar_user]

    # 사용자가 이미 평가한 영화는 제외
    recommendations[np.array(user_item_matrix.iloc[user_index] > 0)] = 0  

    # 추천할 아이템이 없는 경우 빈 리스트 반환
    if np.all(recommendations == 0):
        return []

    # 추천 영화 정렬
    recommended_items = recommendations.argsort()[::-1]

    return recommended_items

user_id = 2
recommended_items = user_based_cf(user_id, user_item_matrix)
print(f"User {user_id}에게 추천하는 영화: {recommended_items + 1}")  # 영화 ID는 1부터 시작하므로 1을 더함
