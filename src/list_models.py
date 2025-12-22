import requests
import urllib3
import json

# 忽略代理产生的 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================= 必填 =================
# 1. 填入你的 Key (请确保是新生成的)
API_KEY = "AIzaSyCXGo9ImNjWGzAQWKqtY7RrSYzocn1wZNw"

# 2. 代理地址 (保持不变)
PROXY_URL = "http://127.0.0.1:7890"
# ========================================

# 注意：这里我们访问的是 /models 接口，不是 /chat/completions
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/models"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}

PROXIES = {
    "http": PROXY_URL,
    "https": PROXY_URL
}

print(f"🔌 正在连接 Google 查询可用模型列表...")

try:
    response = requests.get(
        BASE_URL,
        headers=HEADERS,
        proxies=PROXIES,
        verify=False,
        timeout=15
    )

    print(f"状态码: {response.status_code}")

    if response.status_code == 200:
        models = response.json()['data']
        print("\n🎉 成功！你的账号可以使用以下模型：")
        print("=" * 40)
        found_any = False
        for m in models:
            # 过滤出 chat 模型
            if "gemini" in m['id']:
                print(f"✅ ID: {m['id']}")
                found_any = True
        print("=" * 40)

        if not found_any:
            print("虽然连接成功，但返回列表中没有包含 'gemini' 的模型。")
            print("完整返回内容:", response.json())
        else:
            print("💡 请将上面 ✅ 的 ID (去掉 'models/' 前缀) 填入你的 config.yaml")

    else:
        print("❌ 查询失败，服务器返回：")
        print(response.text)

except Exception as e:
    print(f"❌ 网络错误: {e}")