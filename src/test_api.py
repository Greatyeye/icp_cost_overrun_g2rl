import requests
import json

# =================配置区域=================
# 1. 请把你的 API Key 粘贴在引号中间
API_KEY = "AIzaSyCXGo9ImNjWGzAQWKqtY7RrSYzocn1wZNw"

# 2. 确保地址完整
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

data = {
    "model": "gemini-2.0-flash",
    "messages": [{"role": "user", "content": "Hello Gemini!"}]
}

print("🚀 正在发送请求...")
try:
    # 打印一下即将访问的地址，用于最后检查
    print(f"目标地址: {BASE_URL}")

    response = requests.post(BASE_URL, headers=headers, json=data, timeout=30)

    print(f"状态码: {response.status_code}")
    if response.status_code == 200:
        print("🎉 成功连接！回复内容：")
        print(response.json()['choices'][0]['message']['content'])
    else:
        print("❌ 请求被拒绝，服务器返回：")
        print(response.text)

except Exception as e:
    print(f"❌ 网络错误: {e}")