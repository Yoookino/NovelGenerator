import os
import json
import requests
import numpy as np
import urllib3
import time
import sys
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 从环境变量读取配置 ---
load_dotenv()
# 代理设置
proxy_url = os.getenv("HTTP_PROXY", "http://127.0.0.1:7890")
os.environ['HTTP_PROXY'] = proxy_url
os.environ['HTTPS_PROXY'] = proxy_url

# API 密钥
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")

# 模型与运行配置
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-4-1-fast-reasoning")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "grok")

TOTAL_TOKENS_USED = 0 

BASE_DIR = "data"
CHAPTER_DIR = f"{BASE_DIR}/chapters"
SUMMARY_DIR = f"{BASE_DIR}/summaries"
PROMPT_DIR = f"{BASE_DIR}/prompts"
OUTLINE_DIR = f"{BASE_DIR}/outlines"
SETTING_DIR = f"{BASE_DIR}/settings" 
VECTOR_FILE = f"{BASE_DIR}/vector_store.npy"

for d in [CHAPTER_DIR, SUMMARY_DIR, PROMPT_DIR, OUTLINE_DIR, SETTING_DIR]:
    os.makedirs(d, exist_ok=True)

# ================= 工具函数 =================

def read_file_safe(filepath, default=""):
    if not os.path.exists(filepath): return default
    for enc in ['utf-8', 'gbk']:
        try:
            with open(filepath, "r", encoding=enc) as f: return f.read()
        except: continue
    return default

def embed_text(text):
    url = "https://api.siliconflow.cn/v1/embeddings"
    headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": EMBED_MODEL, "input": [text]}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        return np.array(r.json()["data"][0]["embedding"])
    except:
        return np.zeros(1024)

class VectorStore:
    def __init__(self):
        self.data = []
        if os.path.exists(VECTOR_FILE):
            try: self.data = list(np.load(VECTOR_FILE, allow_pickle=True))
            except: self.data = []

    def save(self): np.save(VECTOR_FILE, self.data)
    def add(self, text):
        self.data.append({"text": text, "vector": embed_text(text)})
        self.save()

    def search(self, query, top_k=5):
        if not self.data: return []
        qv = embed_text(query)
        scored = []
        for item in self.data:
            v = item["vector"]
            score = np.dot(qv, v) / (np.linalg.norm(qv) * np.linalg.norm(v) + 1e-9)
            scored.append((score, item["text"]))
        scored.sort(reverse=True)
        return [t for _, t in scored[:top_k]]

# ================= LLM 调用 (流式输出版本) =================

def call_llm(system_prompt, user_prompt):
    global LLM_PROVIDER, TOTAL_TOKENS_USED
    
    if LLM_PROVIDER == "deepseek":
        api_key, model, url = DEEPSEEK_API_KEY, DEEPSEEK_MODEL, "https://api.deepseek.com/v1/chat/completions"
    else:
        api_key, model, url = GROK_API_KEY, GROK_MODEL, "https://api.x.ai/v1/chat/completions"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,
        "stream": True  # 开启流式传输
    }

    try:
        print(f"正在调用 {LLM_PROVIDER} ({model}) [流式输出]...")
        r = requests.post(url, headers=headers, json=payload, timeout=120, verify=False, stream=True)
        r.raise_for_status()
        
        full_content = []
        print("-" * 30)
        
        for line in r.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    data_body = line_str[6:]
                    if data_body == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(data_body)
                        # 处理内容增量
                        delta = chunk['choices'][0].get('delta', {})
                        if 'content' in delta:
                            content_piece = delta['content']
                            print(content_piece, end="", flush=True)
                            full_content.append(content_piece)
                        
                        # 处理流中的 usage (部分供应商支持)
                        if 'usage' in chunk and chunk['usage'] is not None:
                            TOTAL_TOKENS_USED += chunk['usage'].get('total_tokens', 0)
                    except Exception:
                        continue
        
        print("\n" + "-" * 30)
        
        final_text = "".join(full_content)
        # 如果流中没有统计信息，根据字符粗略估算 (中文约 0.7 token/char, 英文更少)
        if not final_text:
            print("!!! 模型未返回任何内容")
            
        return final_text
    except Exception as e:
        print(f"\nLLM Error: {e}")
        return ""

# ================= 业务功能 =================

def polish_chapter(chapter_no):
    content = read_file_safe(f"{CHAPTER_DIR}/chapter_{chapter_no:03d}.txt")
    if not content: return "找不到该章节。"
    print(f"正在润色第 {chapter_no} 章...")
    system_prompt = "你是一位金牌小说编辑，负责提升文笔。请在保持原意前提下，优化修辞，增强氛围感。"
    polished_text = call_llm(system_prompt, f"请润色以下内容：\n\n{content}")
    if polished_text:
        out_path = f"{CHAPTER_DIR}/chapter_{chapter_no:03d}_polished.txt"
        with open(out_path, "w", encoding="utf-8") as f: f.write(polished_text)
        return f"润色完成: {out_path}"
    return "润色失败。"

def write_chapter(chapter_no, store):
    novel_outline = read_file_safe(f"{OUTLINE_DIR}/novel_outline.txt", "暂无总纲")
    arc_outline = read_file_safe(f"{OUTLINE_DIR}/arc_outline.txt", "暂无卷纲")
    style_prompt = read_file_safe(f"{PROMPT_DIR}/style_prompt.txt", "常规网文风格")
    chars_setting = read_file_safe(f"{SETTING_DIR}/characters.txt", "暂无角色设定")

    summaries = store.search(f"第{chapter_no-1}章 情报总结", 5) if chapter_no > 1 else []

    user_prompt = f"""
【总大纲】{novel_outline}
【卷大纲】{arc_outline}
【角色设定】{chars_setting}
【风格要求】{style_prompt}
【重要前情提要】
{chr(10).join(summaries) if summaries else "（第一章，无前文）"}

【任务】写作第 {chapter_no} 章正文。请尽可能详细展开，不要限制篇幅。
"""
    system_prompt = "你是小说续写引擎，严格执行大纲，禁止偏离。注重细节描写和氛围铺垫。"
    text = call_llm(system_prompt, user_prompt)
    
    if text:
        out_path = f"{CHAPTER_DIR}/chapter_{chapter_no:03d}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n   [文件写入] 第 {chapter_no} 章已保存。")
        return text
    return ""

def summarize_chapter(chapter_no, text, store):
    if not text: return
    system_prompt = "你是一位资深编辑，擅长提取高密度剧情情报。"
    prompt = f"""请为第{chapter_no}章提取“高密度摘要”。必须包含：
1. 【核心剧情】：两句话概括实质进度。
2. 【关键冲突】：本章最高潮的对立点。
3. 【伏笔/信息】：新出现的物品、暗示或细节。
4. 【状态变更】：角色心态、关系或位置的变化。

不要废话，只要干货。
【正文】：
{text}"""
    
    summary = call_llm(system_prompt, prompt)
    if summary:
        with open(f"{SUMMARY_DIR}/chapter_{chapter_no:03d}_summary.txt", "w", encoding="utf-8") as f:
            f.write(summary)
        store.add(f"[第{chapter_no}章情报] {summary}")
        print(f"   [记忆] 情报已归档。")

# ================= CLI =================
def main():
    global LLM_PROVIDER, TOTAL_TOKENS_USED
    store = VectorStore()
    
    print(f"====================================")
    print(f"系统就绪 | 当前模型: {LLM_PROVIDER}")
    print(f"输入 /help 查看指令列表")
    print(f"====================================")
    
    while True:
        try:
            cmd_input = input(">>> ").strip()
            if not cmd_input: continue
            cmd = cmd_input.lower()

            if cmd == "/help":
                print("""
    指令列表:
    /new <n>      - 生成第n章
    /next         - 自动续写下一章
    /auto <n>     - 连续自动生成n章 (带流式预览)
    /polish <n>   - 润色第n章
    /show <n>     - 显示第n章
    /stat         - 查看字数与Token统计
    /switch       - 切换模型 (Grok/DeepSeek)
    /exit         - 退出
                """)

            elif cmd.startswith("/new "):
                try:
                    no = int(cmd_input.split()[-1])
                    text = write_chapter(no, store)
                    if text:
                        summarize_chapter(no, text, store)
                        print(f"   [成功] 第 {no} 章处理完毕。")
                except ValueError:
                    print("!!! 输入错误，请输入数字。")

            elif cmd == "/next":
                chapter_indices = []
                for f in os.listdir(CHAPTER_DIR):
                    if f.startswith("chapter_") and f.endswith(".txt") and "_polished" not in f:
                        try:
                            index = int(f.split('_')[1].split('.')[0])
                            chapter_indices.append(index)
                        except: continue
                next_no = max(chapter_indices) + 1 if chapter_indices else 1
                
                print(f"准备开始第 {next_no} 章...")
                text = write_chapter(next_no, store)
                if text:
                    summarize_chapter(next_no, text, store)
                    print(f"   [成功] 第 {next_no} 章生成完毕。")

            elif cmd.startswith("/auto "):
                try:
                    count = int(cmd_input.split()[-1])
                except:
                    print("!!! 用法: /auto 3")
                    continue

                for i in range(count):
                    chapter_indices = [int(f.split('_')[1].split('.')[0]) for f in os.listdir(CHAPTER_DIR) 
                                       if f.startswith("chapter_") and f.endswith(".txt") and "_polished" not in f]
                    next_no = max(chapter_indices) + 1 if chapter_indices else 1
                    
                    print(f"\n[任务 {i+1}/{count}] 第 {next_no} 章")
                    text = ""
                    for attempt in range(2):
                        text = write_chapter(next_no, store)
                        if text: break
                        time.sleep(5)
                    
                    if text:
                        summarize_chapter(next_no, text, store)
                        if i < count - 1:
                            time.sleep(3)
                    else:
                        break

            elif cmd.startswith("/polish "):
                try:
                    no = int(cmd_input.split()[-1])
                    print(polish_chapter(no))
                except: pass

            elif cmd.startswith("/show "):
                try:
                    no = int(cmd_input.split()[-1])
                    path = os.path.join(CHAPTER_DIR, f"chapter_{no:03d}.txt")
                    if os.path.exists(path):
                        print(f"\n{read_file_safe(path)}\n")
                except: pass

            elif cmd == "/stat":
                files = [f for f in os.listdir(CHAPTER_DIR) if f.endswith(".txt") and "_polished" not in f]
                total_chars = sum([len(read_file_safe(os.path.join(CHAPTER_DIR, f))) for f in files])
                print(f"完成: {len(files)} 章 | 累计: {total_chars} 字 | 本次Token: {TOTAL_TOKENS_USED}")

            elif cmd == "/switch":
                LLM_PROVIDER = "deepseek" if LLM_PROVIDER == "grok" else "grok"
                print(f"已切换至: {LLM_PROVIDER}")

            elif cmd == "/exit":
                break

        except Exception as e:
            print(f"错误: {e}")

if __name__ == "__main__":
    main()