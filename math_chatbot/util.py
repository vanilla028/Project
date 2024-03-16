# ========================
# Few-Shot 구성
# 1. 유사한 문제 예시 찾기 
# 2. 문제-정답-난이도 예시 찾기
# ========================

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from typing import List


class GetFewshot:

    # ChromaDB 가져오기
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    print(chroma_client.list_collections())

    collection = chroma_client.get_collection(name="math_collection")

    # 임베딩 모델 로드
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')

    
def search_problems(self, text, n_results=3, threshold=0.5):
    examples = []
    result = self.collection.query(
        query_embeddings=self.model.encode(text, normalize_embeddings=True).tolist(),
        n_results=n_results
    )

    # 결과에서 'metadatas'를 찾고 각각의 'problem' 키를 리스트에 추가
    for distances_list, metadata_list in zip(result['distances'], result['metadatas']):
        for distance, metadata in zip(distances_list, metadata_list):
            # 유사도가 설정한 임계값 이상인 경우에만 데이터를 추가
            if distance >= threshold:
                examples.append(metadata['problem'])

    return examples

def search_fewshot(self, text, n_results=3, threshold=0.5):
    examples = []
    result = self.collection.query(
        query_embeddings=self.model.encode(text, normalize_embeddings=True).tolist(),
        n_results=n_results
    )

    # 결과에서 'metadatas'를 찾고 각각의 'problem, answer, difficulty' 키를 리스트에 추가
    for distances_list, metadata_list in zip(result['distances'], result['metadatas']):
        for distance, metadata in zip(distances_list, metadata_list):
            # 유사도가 설정한 임계값 이상인 경우에만 데이터를 추가
            if distance >= threshold:
                examples.append((metadata['problem'], metadata['answer'], metadata['difficulty']))

    return examples



class SetFewshot:
    def __init__(self):
        self.fewshot_selector = GetFewshot()


    # 사용자 입력에 대한 퓨샷 데이터 검색 수행
    def search_examples_with_user_input(self, user_text, data_to_search='search_fewshot', n_results=3):

        if data_to_search == 'search_fewshot':
            searched_data = self.fewshot_selector.search_fewshot
        elif data_to_search == 'search_problems':
            searched_data = self.fewshot_selector.search_problems
        elif data_to_search == 'user_data':
            searched_data = self.fewshot_selector.search_user_data
        else:
            raise ValueError("Invalid data_to_search option")

        few_shot_examples = searched_data(user_text, n_results)
        return few_shot_examples


    # 문제 추출
    def problem_examples(self, few_shot_examples: List):
        examples = []
        for problem in few_shot_examples:
            example = {"문제": problem}
            examples.append(example)
    
        return examples
    

    # 문제-정답 쌍
    def fewshot(self, few_shot_examples: List):
        examples = []
        for problem, answer, difficulty in few_shot_examples:
            example = {"문제": problem, "풀이": answer}
            examples.append(example)
        
        return examples


    # 문제-정답-난이도 쌍
    def fewshot_with_difficulty(self, few_shot_examples: List):
        examples = []
        for problem, answer, difficulty in few_shot_examples:
            example = {"문제": problem, "풀이": answer, "난이도": difficulty}
            examples.append(example)
        
        return examples
    
    # 구분 기호, 텍스트 변환
    def format_to_fewshot(self, my_prompt):
        formatted_text = ""

        for problem_set in my_prompt:
            for key, value in problem_set.items():
                formatted_text += f'{key}: {value}\n'
            formatted_text += "###\n"
        
        formatted_text = formatted_text + "문제: "

        return formatted_text

if __name__=='__main__':
    fewshot_selector = GetFewshot()
    p_examples = fewshot_selector.search_problems("민아는 10살이고, 언니는 민아보다 2살 더 많습니다. 할머니의 나이는 언니의 나이의 6배보다 3살이 적을 때, 어머니의 나이는 몇 살인지 풀이 과정을 쓰고, 답을 구하세요.")
    print(p_examples)

    f_examples = fewshot_selector.search_fewshot("민아는 10살이고, 언니는 민아보다 2살 더 많습니다. 할머니의 나이는 언니의 나이의 6배보다 3살이 적을 때, 어머니의 나이는 몇 살인지 풀이 과정을 쓰고, 답을 구하세요.")
    print(f_examples)


    user_text = "문제: 민아는 10살이고, 언니는 민아보다 2살 더 많습니다. 할머니의 나이는 언니의 나이의 6배보다 3살이 적을 때, 할머니의 나이는 몇 살인지 풀이 과정을 쓰고, 답을 구하세요."
    few_shot_prompt = SetFewshot()
    search_data = few_shot_prompt.search_examples_with_user_input(user_text=user_text, data_to_search='search_fewshot', n_results=3)
    my_prompt = few_shot_prompt.fewshot(search_data)
    final = few_shot_prompt.format_to_fewshot(my_prompt)
    print(final)



"""
['수현이는 12살이고, 형은 수현이보다 4살 더 많습니다. 아버지의 나이는 형의 나이의 3배보다 2살이 적을 때, 아버지의 나이는 몇 살인지 풀이 과정을 쓰고, 답을 구하세요.', '솔희, 지희, 민지가 숙제를 한 시간을 조사하였습니다. 세 명 중에서 숙제를 하는 데 가장 많은 시간을 사용한 사람을 찾으세요.\n솔희: 7/10시간, 지희: 5/8시간, 민지: 23/40시간', '4/9와 7/12을 통분할 때 공통분모가 될 수 있는 수 중에서 100보다 작은 수는 무엇이 있는지 풀이 과정을 쓰고 답을 모두 구하세요.']
"""

"""
[('수현이는 12살이고, 형은 수현이보다 4살 더 많습니다. 아버지의 나이는 형의 나이의 3배보다 2살이 적을 때, 아버지의 나이는 몇 살인지 풀이 과정을 쓰고, 답을 구하세요.', '수현이의 나이를 S로 표시하면, 형의 나이는 S+4입니다. 아버지의 나이는 형의 나이의 3배보다 2살이 적다고 했으므로, 아버지의 나이를 A로 표시하면 다음과 같은 식이 성립합니다: A=3×(S+4)−2\n문제에서 수현이의 나이를 12살로 주어졌으므로, 이를 대입하여 아버지의 나이를 구합니다:\nA=3×(12+4)−2\n=3×16−2\n=48−2\n=46\n하나의 식으로 나타내면 (12+4)×3-2=16×3-2=48-2=46(살)입니다. 답: 46\n따라서 아버지의 나이는 46입니다.', '상'), ('솔희, 지희, 민지가 숙제를 한 시간을 조사하였습니다. 세 명 중에서 숙제를 하는 데 가장 많은 시간을 사용한 사람을 찾으세요.\n솔희: 7/10시간, 지희: 5/8시간, 민지: 23/40시간', '우선 솔희와 지희가 숙제를 한 시간을 비교합니다. 솔희가 숙제를 한 시간은 7/10시간이고, 지희가 숙제를 한 시간은 5/8시간이므로 두 분수의 공통분모를 40으로 통분하면 각각 28/40시간, 25/40시간입니다. 따라서 숙제를 더 오래 한 사람은 솔희입니다\n이제 지희와 민지가 숙제를 한 시간을 비교합니다. 지희가 숙제를 한 시간은 5/8시간이고, 민지가 숙제를 한 시간은 23/40이므로 두 분수의 공통분모를 40으로 통분하면 각각 25/40, 23/40이므로 숙제를 더 오래 한 사람은 지희입니다.\n이제 솔희와 민지가 숙제를 한 시간을 비교합니다. 솔희가 숙제를 한 시간은 7/10시간이고, 민지가 숙제를 한 시간은 23/40시간이므로 두 분수의 공통분모를 40으로 통분하면 각각 28/40, 23/40이므로 숙제를 더 오래 한 사람은 솔희입니다.\n이제 종합해 보면, 솔희가 가장 숙제를 오래했고, 다음으로는 지희, 마지막으로는 민지라는 것을 알 수 있습니다. 따라서 답은 솔희, 지희, 민지입니다.', '중'), ('4/9와 7/12을 통분할 때 공통분모가 될 수 있는 수 중에서 100보다 작은 수는 무엇이 있는지 풀이 과정을 쓰고 답을 모두 구하세요.', ' 공통분모가 될 수 있는 수는 두 분모의 공배수이므로 최소공배수인 36의 배수를 구하면 36, 72, 108, ...이고 이 중에서 100보다 작은 수는 36, 72입니다. 답: 36, 72', '상')]
"""
 
"""
문제: 수현이는 12살이고, 형은 수현이보다 4살 더 많습니다. 아버지의 나이는 형의 나이의 3배보다 2살이 적을 때, 아버지의 나이는 몇 살인지 풀이 과정을 쓰고, 답을 구하세요.
정답: 수현이의 나이를 S로 표시하면, 형의 나이는 S+4입니다. 아버지의 나이는 형의 나이의 3배보다 2살이 적다고 했으므로, 아버지의 나이를 A로 표시하면 다음과 같은 식이 성립합니다: A=3×(S+4)−2
문제에서 수현이의 나이를 12살로 주어졌으므로, 이를 대입하여 아버지의 나이를 구합니다:
A=3×(12+4)−2
=3×16−2
=48−2
=46
하나의 식으로 나타내면 (12+4)×3-2=16×3-2=48-2=46(살)입니다. 답: 46
따라서 아버지의 나이는 46입니다.
###
문제: 솔희, 지희, 민지가 숙제를 한 시간을 조사하였습니다. 세 명 중에서 숙제를 하는 데 가장 많은 시간을 사용한 사람을 찾으세요.
솔희: 7/10시간, 지희: 5/8시간, 민지: 23/40시간
정답: 우선 솔희와 지희가 숙제를 한 시간을 비교합니다. 솔희가 숙제를 한 시간은 7/10시간이고, 지희가 숙제를 한 시간은 5/8시간이므로 두 분수의 공통분모를 40으로 통분하면 각각 28/40시간, 25/40시간입니다. 따라서 숙제를 더 오래 한 사람은 솔희입니다
이제 지희와 민지가 숙제를 한 시간을 비교합니다. 지희가 숙제를 한 시간은 5/8시간이고, 민지가 숙제를 한 시간은 23/40이므로 두 분수의 공통분모를 40으로 통분하면 각각 25/40, 23/40이므로 숙제를 더 오래 한 사람은 지희입니다.
이제 솔희와 민지가 숙제를 한 시간을 비교합니다. 솔희가 숙제를 한 시간은 7/10시간이고, 민지가 숙제를 한 시간은 23/40시간이므로 두 분수의 공통분모를 40으로 통분하면 각각 28/40, 23/40이므로 숙제를 더 오래 한 사람은 솔희입니다.
이제 종합해 보면, 솔희가 가장 숙제를 오래했고, 다음으로는 지희, 마지막으로는 민지라는 것을 알 수 있습니다. 따라서 답은 솔희, 지희, 민지입니다.
###
문제: 4/9와 7/12을 통분할 때 공통분모가 될 수 있는 수 중에서 100보다 작은 수는 무엇이 있는지 풀이 과정을 쓰고 답을 모두 구하세요.
정답:  공통분모가 될 수 있는 수는 두 분모의 공배수이므로 최소공배수인 36의 배수를 구하면 36, 72, 108, ...이고 이 중에서 100보다 작은 수는 36, 72입니다. 답: 36, 72
###
"""




