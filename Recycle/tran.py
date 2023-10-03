import sys
from os import rename, listdir

PATH = "./data/*"

filelist = listdir(PATH)

for name in filelist:
    if name.find('.') < 0:
        continue
    replaced = name.replace("jpg","png")
    rename(PATH+'\\'+name, PATH+'\\'+replaced)
    print(name,' -> ',replaced)

print('변환 완료')