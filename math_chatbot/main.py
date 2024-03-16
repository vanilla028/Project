from dotenv import load_dotenv
import os
import time
# OpenAI API KEY 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# OpenAI LLM 모델 로드
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
chat = ChatOpenAI(temperature=0.5)

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from util import GetFewshot, SetFewshot

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# "민아는 10살이고, 언니는 민아보다 2살 더 많습니다. 할머니의 나이는 언니의 나이의 6배보다 3살이 적을 때, 할머니의 나이는 몇 살인지 풀이 과정을 쓰고, 답을 구하세요."
# "수진이는 10살이고, 민아는 11살입니다. 수진이와 민아의 나이를 더한 값에서 9를 뺀 값은 동현이의 나이입니다. 동현이는 몇 살입니까?"

from langchain.prompts import ChatPromptTemplate

# 실행:streamlit run main.py
import streamlit as st
import re

st.title('안녕하세요. 푸리푸리 수학도우미 챗봇입니다.:pencil2: :ledger:\n만나서 반가워요! :wave:')
st.text('저는 초등학교 고학년 수학 문제를 풀어주는 인공지능 챗봇입니다. 어려운 문제를 척척! 풀어 드릴게요.')
st.divider()


on_a = st.toggle('문제 풀기 :memo:')
if on_a:

    # # 캐시 사용하기
    # # DB 초기화 !rm -f .langchain.db 
    # from langchain.globals import set_llm_cache
    # from langchain.cache import SQLiteCache

    # set_llm_cache(SQLiteCache(database_path=".langchain.db"))
    user_text_st = st.text_area("수학 문제를 입력하세요.", key='수학 문제 풀이')

    
    if st.button('요청'):
        with st.spinner('문제 푸는 중...'):
            # time.sleep(1)
          
            # 퓨샷 구성
            user_text = user_text_st

            few_shot_prompt = SetFewshot()
            search_data = few_shot_prompt.search_examples_with_user_input(user_text=user_text, data_to_search='search_fewshot', n_results=3)
            my_prompt = few_shot_prompt.fewshot(search_data)
            temp1 = few_shot_prompt.format_to_fewshot(my_prompt)
            # streamlit에서 마크다운 적용
            temp2 = temp1.replace("###", "\n\###\n")
            final = temp2.replace("풀이", "\n풀이")

            task = final + user_text

            final_prompt = ChatPromptTemplate.from_messages([
                    ("system", "당신은 유능한 수학 선생님입니다. 예시(문제, 풀이)를 참고하고, 주어진 문제에 대하여 풀이 과정을 서술합니다. 반드시 생성한 풀이 과정만 출력하세요."),                  
                    ("user", "{query}")
                ]
            )

            chain = final_prompt | chat
            response = chain.invoke({"query":{task}})
            response = str(response)
            response = response.replace("content=", "")
            response = response.replace("\\n", "\n")
            response = response.strip('", {\', \'}')
            st.write(response)
         
            
    if st.button('Chroma DB에서 검색한 Few-Shot 보기 :floppy_disk:', key="문제 풀이 퓨샷"):
        with st.spinner('불러오는 중...'):
            # time.sleep(1)

            # 퓨샷 구성
            user_text = user_text_st

            few_shot_prompt = SetFewshot()
            search_data = few_shot_prompt.search_examples_with_user_input(user_text=user_text, data_to_search='search_fewshot', n_results=3)
            my_prompt = few_shot_prompt.fewshot(search_data)
            temp1 = few_shot_prompt.format_to_fewshot(my_prompt)
            # streamlit에서 마크다운 적용
            temp2 = temp1.replace("###", "\n\###\n")
            final = temp2.replace("풀이", "\n풀이")
            instruction = "당신은 유능한 수학 선생님입니다. 예시(문제, 풀이)를 참고하고, 주어진 문제에 대하여 풀이 과정을 서술합니다. 반드시 생성한 풀이 과정만 출력하세요."

            task = final + user_text


            st.write(instruction)
            st.write('====================== 퓨샷 예시:\n')
            st.write(final)
            st.write('====================== 최종 프롬프트:\n')
            st.write(instruction)
            st.write(task)


    st.button("Reset", type="primary", key="수학 문제 풀이 리셋")


on_b = st.toggle('초등 수학 교육과정 :mag_right:')
if on_b:


    math_curriculum_info = st.text_input('초등학교 수학 교육과정에 대해 질문해 보세요! :nerd_face: :mag_right:')
    if st.button('궁금해요!'):

        
        # ===========================================================================
        # 초등학교 수학 교과 과정 RAG LangChain 모듈 이용

        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader("elementary_math_edu.pdf")
        docs = loader.load_and_split()


        from operator import itemgetter

        from langchain_community.vectorstores import FAISS
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnableLambda, RunnablePassthrough
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader("elementary_math_edu.pdf")
        pages = loader.load_and_split()

        # print(pages[0])

        # ===========================================================================

        with st.spinner('교과 과정 검색 중...'):

            question = math_curriculum_info
            # question = "초등학교 4학년 수학 교과 과정을 알려줘."

            vectorstore = FAISS.from_texts(
                [question], embedding=OpenAIEmbeddings()
            )
            retriever = vectorstore.as_retriever()

            template = """당신은 초등학교 수학 교과 과정 가이드입니다. 다음 내용을 기반으로 질문에 답변하세요. 만약에 질문에 대한 내용을 찾을 수 없으면, "해당 내용은 초등학교 수학 교육과정에서 찾을 수 없습니다."라고 답변하세요.:
            {context}
            ###
            질문: {question}
            답변:

            """
            prompt = ChatPromptTemplate.from_template(template)

            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | chat
                | StrOutputParser()
            )

            result = chain.invoke(question)
            st.write(result)
            
    st.button("Reset", type="primary", key="수학 교육과정 리셋")


on_c = st.toggle('문제 생성 :spiral_note_pad:')
if on_c:

    st.text('아래에 문제를 입력하면, 비슷한 문제를 만들어 드릴게요.')
    st.divider()
    user_text_st = st.text_area("수학 문제를 입력하세요.", key="수학 문제 생성 리셋")

    if st.button('제출'):
        with st.spinner('문제 생성 중...'):
            
            # 퓨샷 구성
            user_text = user_text_st

            few_shot_prompt = SetFewshot()
            search_data = few_shot_prompt.search_examples_with_user_input(user_text=user_text, data_to_search='search_fewshot', n_results=3)
            my_prompt = few_shot_prompt.fewshot_with_difficulty(search_data)
            final = few_shot_prompt.format_to_fewshot(my_prompt)

            task = final + user_text

            final_prompt = ChatPromptTemplate.from_messages([
                    ("system", "당신은 초등학교 수학 문제 출제자입니다. 주어진 문제 4개와 비슷한 유형의 문제 1개를 새롭게 생성하는 작업을 합니다. 새롭게 생성한 문제의 출력 형식에는 '문제, 풀이, 난이도'가 포함되어야 합니다."),
                    ("user", "{query}")
                ]
            )

            chain = final_prompt | chat
            response = chain.invoke({"query":{task}})
            response = str(response)
            response = response.replace("content=", "")
            response = response.replace("\\n", "\n")
            response = response.strip('", {\', \'}')
            response = re.sub(r'\\+$', '', response)
            st.write(response)

    if st.button('Chroma DB에서 검색한 Few-Shot 보기 :floppy_disk:', key="문제 생성 퓨샷"):
        with st.spinner('불러오는 중...'):

            # 퓨샷 구성
            user_text = user_text_st

            few_shot_prompt = SetFewshot()
            search_data = few_shot_prompt.search_examples_with_user_input(user_text=user_text, data_to_search='search_fewshot', n_results=3)
            my_prompt = few_shot_prompt.fewshot(search_data)
            temp1 = few_shot_prompt.format_to_fewshot(my_prompt)
            # streamlit에서 마크다운 적용
            temp2 = temp1.replace("###", "\n\###\n")
            temp3 = temp2.replace("풀이", "\n풀이")
            final = temp3.replace("난이도", "\난이도")
            instruction = "당신은 초등학교 수학 문제 출제자입니다. 주어진 문제 4개와 비슷한 유형의 문제 1개를 새롭게 생성하는 작업을 합니다. 새롭게 생성한 문제의 출력 형식에는 '문제, 풀이, 난이도'가 포함되어야 합니다."

            task = final + user_text

            st.write('====================== 퓨샷 예시:\n')
            st.write(final)
            st.write('====================== 최종 프롬프트:\n')
            st.write(instruction)
            st.write(task)


    st.button("Reset", type="primary", key="수학 문제 생성")

