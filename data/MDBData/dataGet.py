import requests
import os
import re
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- 配置区 ---
save_folder = "downloaded_pdfs"
batch_size = 20  # 每次运行想新下载多少个？
# ----------------

# 1. 准备文件夹
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 2. 计算本地已有多少文件 (这就是我们的偏移量)
existing_files = [f for f in os.listdir(save_folder) if f.endswith('.pdf')]
current_count = len(existing_files)
print(f"📂 本地已有文件: {current_count} 个")
print(f"🚀 准备从第 {current_count + 1} 个开始下载新的一批...")

# 3. 设置网络连接
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# 4. 请求 API (带上 os 参数)
url = "https://search.worldbank.org/api/v2/wds"
params = {
    "format": "json",
    "fl": "display_title,docdt,pdfurl",
    "docty": "Project Appraisal Document",
    "rows": str(batch_size),  # 这次取多少个
    "os": str(current_count),  # <--- 关键：Offset，跳过本地已有的数量
    "qterm": "*",
    "sort": "docdt desc"  # 按时间倒序，保证顺序稳定
}

try:
    response = session.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()

    documents = data.get('documents', {})

    # API 有个坑：有时返回是列表，有时是字典，做个兼容处理
    if isinstance(documents, list):
        # 这种情况通常很少见，但为了代码健壮性
        doc_iter = documents
    else:
        doc_iter = documents.values()

    print(f"🔍 API 返回了 {len(documents)} 条新记录，开始处理...\n")

    new_download_count = 0
    for doc_info in doc_iter:
        pdf_link = doc_info.get('pdfurl')
        title = doc_info.get('display_title', 'untitled')

        if pdf_link:
            try:
                # 文件名处理
                safe_title = re.sub(r'[\\/*?:"<>|]', "", title)[:100].strip()
                filename = f"{save_folder}/{safe_title}.pdf"

                # 双重保险：虽然我们翻页了，但还是检查一下是否存在
                if os.path.exists(filename):
                    print(f"⏩ (偶发重复) 跳过: {safe_title}")
                    continue

                print(f"⬇️ 下载中: {safe_title}...")

                pdf_response = session.get(pdf_link, headers=headers, stream=True, timeout=60)
                with open(filename, 'wb') as f:
                    for chunk in pdf_response.iter_content(chunk_size=8192):
                        f.write(chunk)

                print(f"✅ 保存成功")
                new_download_count += 1
                time.sleep(2)  # 礼貌延时

            except Exception as e:
                print(f"❌ 出错: {e}")
                time.sleep(1)
        else:
            print(f"⚠️ 无 PDF 链接: {title}")

    print(f"\n🎉 本次运行结束！新下载了 {new_download_count} 个文件。")
    print(f"现在本地总共有 {len(os.listdir(save_folder))} 个文件。")

except Exception as e:
    print(f"发生错误: {e}")