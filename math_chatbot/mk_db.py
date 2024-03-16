import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

chroma_client = chromadb.HttpClient(host='localhost', port=8000)
print(chroma_client.list_collections())

# chroma_client.delete_collection("my_collection")


# collection = chroma_client.create_collection(name="math_collection")
collection = chroma_client.get_collection(name="math_collection")

df = pd.read_csv("problems.csv")
# print(df.sample(5))

# 모델 로드
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# ids = []
# metadatas =[]
# embeddings = []

# for row in tqdm(df.iterrows()):
#     index = row[0]
#     problem = row[1].problem
#     answer = row[1].answer
#     difficulty = row[1].difficulty

    
#     metadata = {
#         "problem": problem,
#         "answer": answer,
#         "difficulty": difficulty
#     }
    
#     embedding = model.encode(problem, normalize_embeddings=True)
    
#     ids.append(str(index))
#     metadatas.append(metadata)
#     embeddings.append(embedding)

# chunk_size = 12  # 한 번에 처리할 chunk 크기 설정
# total_chunks = len(embeddings) // chunk_size  # 전체 데이터를 chunk 단위로 나눈 횟수
# embeddings = [ e.tolist() for e in tqdm(embeddings)]  

# for chunk_idx in tqdm(range(total_chunks)):
#     start_idx = chunk_idx * chunk_size
#     end_idx = (chunk_idx + 1) * chunk_size
    
#     # chunk 단위로 데이터 자르기
#     chunk_embeddings = embeddings[start_idx:end_idx]
#     chunk_ids = ids[start_idx:end_idx]
#     chunk_metadatas = metadatas[start_idx:end_idx]
    
#     # chunk를 collection에 추가
#     collection.add(embeddings=chunk_embeddings, ids=chunk_ids, metadatas=chunk_metadatas)

text = "가게에서 300원짜리 과자 1개와 500원짜리 우유 1개를 사고 2000원을 냈습니다. 거스름돈은 얼마를 받아야 하나요?"

result = collection.query(
    query_embeddings=model.encode(text, normalize_embeddings=True).tolist(),
    n_results=3
)




print(result)

"""
{'ids': [['23', '59', '67']], 'distances': [[0.35854553065509875, 0.74035505208057, 0.9596338282709668]], 'embeddings': None, 'metadatas': [[{'answer': "'사탕 5개에 1500원인' 사탕 3개의 가격은 1500/5=900원입니다.\n850원짜리 우유 2개의 가격은 850×2=1700원입니다.\n1300원짜리 어묵 1개의 가격은 1300원입니다.\n따라서, 총 구매한 물건들의 가격은\n900+1700+1300=3900원입니다.\n고객은 5000원을 지불했으므로, 거스름돈은\n5000−3900=1100원입니다.\n따라서, 고객은 1100원의 거스름돈을 받아야 합니다.", 'difficulty': '중', 'problem': '가게에서 5개에 1500원인 사탕 3개와 850원짜리 우유 2개, 1300원짜리 어묵 1개를 사고 5000원을 냈습니다. 거스름돈은 얼마를 받아야 하나요?'}, {'answer': '판매 금액과 판매한 과자의 수 사이의 대응 관계를 식으로 나타내면 (판매 금액)÷700=(과자의 수)이므로 판매 금액이 8400원이 되려면 8400÷700=12(봉지)를 팔아야 합니다. 답: 12', 'difficulty': '상', 'problem': '어느 가게에서 과자 한 봉지를 700원에 판매하고 있습니다. 판매 금액이 8400원이 되려면 과자를 몇 봉지 팔아야 하는지 풀이 과정을 쓰고 답을 구하세요.'}, {'answer': '한 상자에 30개씩 담긴 사탕이 6상자 있으므로 총 사탕의 수는 30*6개입니다. 30×6/9=20, 한 사람에게 줄 수 있는 사탕은 20개입니다.', 'difficulty': '중', 'problem': '한 상자에 30개씩 담긴 사탕이 6상자 있습니다. 이 사탕을 9명에게 똑같이 나누어 주려면 한 사람에게 줄 수 있는 사탕은 몇 개인지 하나의 식으로 나타내고 구해 보세요.'}]], 'documents': [[None, None, None]], 'uris': None, 'data': None}
"""
