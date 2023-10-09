from selenium import webdriver
from selenium.webdriver.common.keys import Keys # 키보드 역할
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from multiprocessing import Pool
import pandas as pd
import os            # 이미지 저장을 위한 폴더 생성 관리
import time
import urllib.request # 이미지 다운로드를 위한 라이브러리

# 키워드 가져오기
# keys = pd.read_csv("./keyword.txt", encoding="utf-8", names=['keyword'])
"""
      keyword
0     dekopon
1     grapefruit
2     kanpei
3     orange
4     cheonhyehyang
"""
keys = {0: "dekopon", 1: "grapefruit", 2: "grapefruit", 3: "kanpei", 4: "orange", 5: "cheonhyehyang"}

# keyword = [keyword.append(keys['keyword'][x]) for x in range(len(keys))]
keyword = [keys[x] for x in range(len(keys))]

def create_folder(dir):
    # 이미지 저장할 폴더 구성
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print("Error creating folder", + dir)


def image_download(keyword):
    # image download 함수
    create_folder("./" + keyword + "/")

    # chromdriver 가져오기
    options = Options()
    # options.add_argument('--headless') # 백그라운드 실행
    options.add_experimental_option("detach", True) # 세션 종류 후에도 브라우저 창 유지하기
      
      service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(3) # 페이지 로딩이 완료될 때까지 기다리는 코드

    print("keyword: " + keyword)
      
    # 사이트 접속하기  
    driver.get('https://www.google.co.kr/imghp?hl=ko')
    keywords = driver.find_element_by_xpath(
        '/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input')
    keywords.send_keys(keyword)
    driver.find_element_by_xpath(
        '/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/button').send_keys(Keys.ENTER)

    print(keyword+' 스크롤 중........')
    elem = driver.find_element_by_tag_name("body")
    for i in range(60):
        elem.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.1)

    try:
        driver.find_element_by_xpath(
            '//*[@id="islmp"]/div/div/div/div[2]/div[1]/div[2]/div[2]/input').click()
        for i in range(60):
            elem.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.1)
    except:
        pass

    images = driver.find_elements_by_css_selector("img.rg_i.Q4LuWd")
    print(keyword+' 찾은 이미지 개수:', len(images))

    links = [] # 다운로드할 이미지 링크 수집
    for i in range(1, len(images)):
        try:
            print('//*[@id="islrg"]/div[1]/div['+str(i)+']/a[1]/div[1]/img')

            driver.find_element_by_xpath(
                '//*[@id="islrg"]/div[1]/div['+str(i)+']/a[1]/div[1]/img').click()
            links.append(driver.find_element_by_xpath(
                '//*[@id="Sva75c"]/div[2]/div/div[2]/div[2]/div[2]/c-wiz/div[2]/div[1]/div[1]/div[2]/div/a/img').get_attribute('src'))
            # driver.find_element_by_xpath('//*[@id="Sva75c"]/div[2]/div/div[2]/div[2]/div[2]/c-wiz/div[2]/div[1]/div[1]/div[2]/div/a').click()
            print(keyword+' 링크 수집 중........ number :'+str(i)+'/'+str(len(images)))
        except:
            continue

    forbidden = 0 # 이미지 다운로드 시 발생하는 예외 계산
    for k, i in enumerate(links):
        try:
            url = i
            start = time.time()
            urllib.request.urlretrieve(
                url, "./"+keyword+"/"+str(k-forbidden)+".jpg")
            print(str(k+1)+'/'+str(len(links))+' '+keyword +
                  ' 다운로드 중........ Download time : '+str(time.time() - start)[:5]+' 초')
        except:
            forbidden += 1
            continue
    print(keyword+' ---다운로드 완료---')

    driver.close()


# =============================================================================
# 실행
# =============================================================================
if __name__ == '__main__':
    pool = Pool(processes=5)  # 5개의 프로세스 사용
    pool.map(image_download, keyword)
