from langdetect import detect

print(detect("这是一段中文"))  # zh-cn
print(detect("This is English"))  # en
