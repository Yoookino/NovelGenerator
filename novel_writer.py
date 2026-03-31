import os
import json
import re
import time
import requests
import numpy as np
import urllib3
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()
# Keep proxy wiring centralized so every downstream HTTP call inherits it.
proxy_url = os.getenv("HTTP_PROXY", "http://127.0.0.1:7890")
os.environ["HTTP_PROXY"] = proxy_url
os.environ["HTTPS_PROXY"] = proxy_url

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")

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
PLAN_DIR = f"{BASE_DIR}/plans"
QA_DIR = f"{BASE_DIR}/qa_reports"
MEMORY_DIR = f"{BASE_DIR}/memory"
VECTOR_FILE = f"{BASE_DIR}/vector_store.npy"
BLUEPRINT_FILE = f"{OUTLINE_DIR}/novel_blueprint.json"
STYLE_PROFILE_FILE = f"{PROMPT_DIR}/style_profile.json"
CHARACTER_STATE_FILE = f"{SETTING_DIR}/character_state.json"
PLOT_MEMORY_FILE = f"{MEMORY_DIR}/plot_memory.json"
SETTING_MEMORY_FILE = f"{MEMORY_DIR}/setting_memory.json"
FORESHADOW_MEMORY_FILE = f"{MEMORY_DIR}/foreshadow_memory.json"

for directory in [
    CHAPTER_DIR,
    SUMMARY_DIR,
    PROMPT_DIR,
    OUTLINE_DIR,
    SETTING_DIR,
    PLAN_DIR,
    QA_DIR,
    MEMORY_DIR,
]:
    os.makedirs(directory, exist_ok=True)

DEFAULT_STYLE_PROFILE = {
    "narrative_style": "偏沉浸式网文叙事",
    "pace": "中速推进，章节内有明显起伏",
    "description_density": "中高",
    "dialogue_ratio": "中",
    "emotion_intensity": "中高",
    "suspense_strength": "强",
    "romance_temperature": "克制暧昧",
    "action_density": "中",
    "sentence_style": "长短句交替",
    "ending_hook_strength": "强",
}


def read_file_safe(filepath, default=""):
    if not os.path.exists(filepath):
        return default
    for encoding in ["utf-8", "gbk"]:
        try:
            with open(filepath, "r", encoding=encoding) as file:
                return file.read()
        except Exception:
            continue
    return default


def write_text(filepath, content):
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(content)


def read_json_safe(filepath, default=None):
    if default is None:
        default = {}
    if not os.path.exists(filepath):
        return default
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return default


def write_json(filepath, payload):
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def ensure_list(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def extract_json_block(text):
    if not text:
        return None
    # Models may wrap JSON in fenced blocks or add extra narration around it.
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    raw = re.search(r"(\{.*\})", text, re.S)
    if raw:
        return raw.group(1)
    return None


def serialize_compact(payload):
    return json.dumps(payload, ensure_ascii=False, indent=2)


def current_llm_config():
    if LLM_PROVIDER == "deepseek":
        return DEEPSEEK_API_KEY, DEEPSEEK_MODEL, "https://api.deepseek.com/v1/chat/completions"
    return GROK_API_KEY, GROK_MODEL, "https://api.x.ai/v1/chat/completions"


def embed_text(text):
    url = "https://api.siliconflow.cn/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": EMBED_MODEL, "input": [text]}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        return np.array(response.json()["data"][0]["embedding"])
    except Exception:
        return np.zeros(1024)


class VectorStore:
    def __init__(self):
        self.data = []
        if os.path.exists(VECTOR_FILE):
            try:
                self.data = list(np.load(VECTOR_FILE, allow_pickle=True))
            except Exception:
                self.data = []
        self.data = [self._normalize_item(item) for item in self.data]

    def _normalize_item(self, item):
        # Older vector stores may contain plain strings; upgrade them on load.
        if isinstance(item, dict) and "text" in item and "vector" in item:
            normalized = dict(item)
            normalized["category"] = normalized.get("category", "general")
            normalized["chapter"] = normalized.get("chapter")
            return normalized
        return {"text": str(item), "vector": embed_text(str(item)), "category": "general", "chapter": None}

    def save(self):
        np.save(VECTOR_FILE, self.data)

    def add(self, text, category="general", chapter=None):
        self.data.append(
            {
                "text": text,
                "vector": embed_text(text),
                "category": category,
                "chapter": chapter,
            }
        )
        self.save()

    def search(self, query, top_k=5, category=None):
        if not self.data:
            return []
        query_vector = embed_text(query)
        scored = []
        for item in self.data:
            if category and item.get("category") != category:
                continue
            vector = item["vector"]
            # Cosine similarity is sufficient for this lightweight local memory store.
            score = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector) + 1e-9)
            scored.append((score, item["text"]))
        scored.sort(reverse=True)
        return [text for _, text in scored[:top_k]]


def call_llm(system_prompt, user_prompt, temperature=0.7):
    global TOTAL_TOKENS_USED
    api_key, model, url = current_llm_config()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "stream": True,
    }
    try:
        print(f"正在调用 {LLM_PROVIDER} ({model}) [流式输出]...")
        response = requests.post(url, headers=headers, json=payload, timeout=180, verify=False, stream=True)
        response.raise_for_status()
        content_parts = []
        print("-" * 30)
        for line in response.iter_lines():
            if not line:
                continue
            line_text = line.decode("utf-8")
            if not line_text.startswith("data: "):
                continue
            data_body = line_text[6:]
            if data_body == "[DONE]":
                break
            try:
                # Streaming providers send incremental deltas instead of one full message.
                chunk = json.loads(data_body)
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    piece = delta["content"]
                    print(piece, end="", flush=True)
                    content_parts.append(piece)
                if "usage" in chunk and chunk["usage"] is not None:
                    TOTAL_TOKENS_USED += chunk["usage"].get("total_tokens", 0)
            except Exception:
                continue
        print("\n" + "-" * 30)
        final_text = "".join(content_parts)
        if not final_text:
            print("!!! 模型未返回任何内容")
        return final_text
    except Exception as error:
        print(f"\nLLM Error: {error}")
        return ""


def call_llm_json(system_prompt, user_prompt, fallback=None, temperature=0.6):
    if fallback is None:
        fallback = {}
    # JSON mode is best-effort: callers still receive a usable fallback shape on parse failure.
    response = call_llm(
        system_prompt + "\n请只返回 JSON，不要输出解释，不要使用 markdown。",
        user_prompt,
        temperature=temperature,
    )
    json_block = extract_json_block(response)
    if not json_block:
        return fallback
    try:
        return json.loads(json_block)
    except Exception:
        return fallback


def get_existing_chapter_numbers():
    numbers = []
    for filename in os.listdir(CHAPTER_DIR):
        if filename.startswith("chapter_") and filename.endswith(".txt") and "_polished" not in filename:
            try:
                numbers.append(int(filename.split("_")[1].split(".")[0]))
            except Exception:
                continue
    return sorted(numbers)


def get_next_chapter_no():
    numbers = get_existing_chapter_numbers()
    return numbers[-1] + 1 if numbers else 1


def load_style_profile():
    profile = dict(DEFAULT_STYLE_PROFILE)
    profile.update(read_json_safe(STYLE_PROFILE_FILE, {}))
    return profile


def style_profile_text(profile):
    return "\n".join(f"- {key}: {value}" for key, value in profile.items())


def load_blueprint():
    return read_json_safe(BLUEPRINT_FILE, {})


def load_character_state():
    return read_json_safe(CHARACTER_STATE_FILE, {"characters": []})


def save_character_state(payload):
    write_json(CHARACTER_STATE_FILE, payload)


def append_json_list(filepath, items):
    current = read_json_safe(filepath, [])
    if not isinstance(current, list):
        current = []
    current.extend(items)
    write_json(filepath, current)


def blueprint_to_texts(blueprint):
    # Convert structured blueprint JSON into the plain-text files used by later steps.
    title = blueprint.get("title", "未命名小说")
    premise = blueprint.get("premise", "")
    genre = blueprint.get("genre", "")
    selling_points = ensure_list(blueprint.get("selling_points"))
    world_rules = ensure_list(blueprint.get("world_rules"))
    volume_plan = blueprint.get("volume_plan", [])
    character_cards = blueprint.get("character_cards", [])
    style_profile = blueprint.get("style_profile", DEFAULT_STYLE_PROFILE)

    novel_outline_lines = [
        f"书名：{title}",
        f"题材：{genre}",
        f"核心 premise：{premise}",
        "",
        "卖点：",
        *[f"- {item}" for item in selling_points],
        "",
        "世界规则：",
        *[f"- {item}" for item in world_rules],
        "",
        "卷结构：",
    ]
    for volume in volume_plan:
        novel_outline_lines.extend(
            [
                f"- {volume.get('volume_name', '未命名卷')}（{volume.get('chapter_range', '章节待定')}）",
                f"  目标：{volume.get('goal', '待补充')}",
                f"  冲突：{volume.get('conflict', '待补充')}",
                f"  高潮：{volume.get('climax', '待补充')}",
                f"  结果：{volume.get('result', '待补充')}",
            ]
        )
    novel_outline = "\n".join(novel_outline_lines).strip() + "\n"

    arc_lines = ["当前卷规划："]
    for volume in volume_plan[:3]:
        arc_lines.extend(
            [
                f"- {volume.get('volume_name', '未命名卷')}（{volume.get('chapter_range', '章节待定')}）",
                f"  主要事件：{volume.get('goal', '待补充')}",
                f"  冲突升级：{volume.get('conflict', '待补充')}",
                f"  结尾钩子：{volume.get('hook', '待补充')}",
            ]
        )
    arc_outline = "\n".join(arc_lines).strip() + "\n"

    char_lines = []
    characters_dump = []
    for index, card in enumerate(character_cards):
        if index > 0:
            char_lines.append("")
        char_lines.append(f"{card.get('name', '未知角色')}：")
        char_lines.extend(
            [
                f"- 定位：{card.get('role', '待补充')}",
                f"- 外显性格：{card.get('personality', '待补充')}",
                f"- 核心动机：{card.get('motivation', '待补充')}",
                f"- 当前关系：{card.get('relationship', '待补充')}",
                f"- 角色弧线：{card.get('arc', '待补充')}",
                f"- 隐藏秘密：{card.get('secret', '待补充')}",
            ]
        )
        characters_dump.append(
            {
                "name": card.get("name", "未知角色"),
                "role": card.get("role", ""),
                "motivation": card.get("motivation", ""),
                "relationship": card.get("relationship", ""),
                "emotion": "待展开",
                "stance": "未变化",
                "location": "待定",
                "secret_progress": "未揭露",
                "recent_change": "无",
            }
        )
    characters_text = "\n".join(char_lines).strip() + "\n" if character_cards else "暂无角色设定\n"

    style_lines = ["写作风格配置："]
    for key, value in style_profile.items():
        style_lines.append(f"- {key}: {value}")
    style_text = "\n".join(style_lines).strip() + "\n"

    return {
        "novel_outline": novel_outline,
        "arc_outline": arc_outline,
        "characters_text": characters_text,
        "style_text": style_text,
        "character_state": {"characters": characters_dump},
        "style_profile": style_profile,
        "plot_memory": [{"chapter": 0, "type": "global_story_frame", "content": premise}],
        "setting_memory": [{"name": "世界规则", "facts": world_rules}],
        "foreshadow_memory": [{"status": "planned", "content": item} for item in selling_points[:5]],
    }


def generate_novel_framework(idea):
    # Build the initial story package that every later command depends on.
    print("正在生成小说框架...")
    fallback = {
        "title": "未命名小说",
        "premise": idea.strip() or "一部待完善的长篇小说",
        "genre": "待定",
        "selling_points": ["强冲突", "持续钩子", "角色成长"],
        "world_rules": ["世界规则待补充"],
        "style_profile": dict(DEFAULT_STYLE_PROFILE),
        "character_cards": [],
        "volume_plan": [],
    }
    system_prompt = "你是长篇网文总策划，负责把一个创意扩展成可直接开写的完整框架。"
    user_prompt = f"""
用户创意：
{idea}

请生成一份完整的小说蓝图 JSON，字段必须包含：
title, premise, genre, selling_points, world_rules, style_profile, character_cards, volume_plan

约束：
1. style_profile 必须包含 narrative_style, pace, description_density, dialogue_ratio, emotion_intensity, suspense_strength, romance_temperature, action_density, sentence_style, ending_hook_strength
2. character_cards 至少 4 个角色，每个角色包含 name, role, personality, motivation, relationship, arc, secret
3. volume_plan 至少 3 卷，每卷包含 volume_name, chapter_range, goal, conflict, climax, result, hook
4. 结果要适合直接进入章节创作流程
"""
    blueprint = call_llm_json(system_prompt, user_prompt, fallback=fallback)
    for key, value in fallback.items():
        blueprint.setdefault(key, value)
    texts = blueprint_to_texts(blueprint)
    # Persist both machine-readable and human-readable artifacts for the rest of the workflow.
    write_json(BLUEPRINT_FILE, blueprint)
    write_text(f"{OUTLINE_DIR}/novel_outline.txt", texts["novel_outline"])
    write_text(f"{OUTLINE_DIR}/arc_outline.txt", texts["arc_outline"])
    write_text(f"{SETTING_DIR}/characters.txt", texts["characters_text"])
    write_text(f"{PROMPT_DIR}/style_prompt.txt", texts["style_text"])
    write_json(STYLE_PROFILE_FILE, texts["style_profile"])
    save_character_state(texts["character_state"])
    write_json(PLOT_MEMORY_FILE, texts["plot_memory"])
    write_json(SETTING_MEMORY_FILE, texts["setting_memory"])
    write_json(FORESHADOW_MEMORY_FILE, texts["foreshadow_memory"])
    print("   [框架] 小说蓝图已生成并落盘。")
    return blueprint


def build_memory_context(store, chapter_no):
    # Pull a compact cross-section of plot, character, setting, and foreshadow memory.
    plot_memories = store.search(f"第{chapter_no}章需要衔接的剧情节点", top_k=5, category="plot")
    char_memories = store.search(f"第{chapter_no}章角色状态变化", top_k=5, category="character")
    setting_memories = store.search(f"第{chapter_no}章世界设定规则", top_k=3, category="setting")
    foreshadow_memories = store.search(f"第{chapter_no}章需要回收的伏笔", top_k=5, category="foreshadow")
    parts = [
        "【剧情记忆】",
        *(plot_memories or ["暂无"]),
        "",
        "【角色记忆】",
        *(char_memories or ["暂无"]),
        "",
        "【设定记忆】",
        *(setting_memories or ["暂无"]),
        "",
        "【伏笔记忆】",
        *(foreshadow_memories or ["暂无"]),
    ]
    return "\n".join(parts)


def generate_chapter_plan(chapter_no, store):
    # Plans act as the contract between outline context and the segmented drafting step.
    novel_outline = read_file_safe(f"{OUTLINE_DIR}/novel_outline.txt", "暂无总纲")
    arc_outline = read_file_safe(f"{OUTLINE_DIR}/arc_outline.txt", "暂无卷纲")
    characters_text = read_file_safe(f"{SETTING_DIR}/characters.txt", "暂无角色设定")
    character_state = load_character_state()
    style_profile = load_style_profile()
    memory_context = build_memory_context(store, chapter_no)
    fallback = {
        "chapter_goal": f"推进第 {chapter_no} 章主线",
        "main_conflict": "制造冲突并推动人物变化",
        "twist": "在结尾抛出新的不确定性",
        "hook": "制造下一章阅读动力",
        "must_use_memories": [],
        "character_focus": [],
        "sections": [
            {"name": "开场", "purpose": "接上前文并快速进入情境", "beats": "", "caution": ""},
            {"name": "推进", "purpose": "扩大目标和阻碍", "beats": "", "caution": ""},
            {"name": "高潮", "purpose": "爆发本章主要冲突", "beats": "", "caution": ""},
            {"name": "收尾", "purpose": "给出变化并留下钩子", "beats": "", "caution": ""},
        ],
    }
    system_prompt = "你是网文章节规划编辑，擅长在正式写作前为单章建立清晰骨架。"
    user_prompt = f"""
请为第 {chapter_no} 章生成写作计划 JSON。

【总纲】
{novel_outline}

【卷纲】
{arc_outline}

【角色设定】
{characters_text}

【人物状态】
{serialize_compact(character_state)}

【文风配置】
{style_profile_text(style_profile)}

【记忆检索】
{memory_context}

字段必须包含：
chapter_goal, main_conflict, twist, hook, must_use_memories, character_focus, sections

sections 必须是 4 段，每段包含 name, purpose, beats, caution。
"""
    plan = call_llm_json(system_prompt, user_prompt, fallback=fallback)
    for key, value in fallback.items():
        plan.setdefault(key, value)
    # Save the plan twice: JSON for tooling and text for quick manual inspection.
    write_json(f"{PLAN_DIR}/chapter_{chapter_no:03d}_plan.json", plan)

    lines = [
        f"第 {chapter_no} 章写作计划",
        f"- 章节目标：{plan.get('chapter_goal', '')}",
        f"- 核心冲突：{plan.get('main_conflict', '')}",
        f"- 关键反转：{plan.get('twist', '')}",
        f"- 章节钩子：{plan.get('hook', '')}",
        "- 角色焦点：",
        *[f"  - {item}" for item in ensure_list(plan.get("character_focus"))],
        "- 段落规划：",
    ]
    for section in plan.get("sections", []):
        lines.extend(
            [
                f"  - {section.get('name', '段落')}: {section.get('purpose', '')}",
                f"    节拍：{section.get('beats', '')}",
                f"    注意：{section.get('caution', '')}",
            ]
        )
    write_text(f"{PLAN_DIR}/chapter_{chapter_no:03d}_plan.txt", "\n".join(lines).strip() + "\n")
    print(f"   [计划] 第 {chapter_no} 章计划已生成。")
    return plan


def write_segmented_chapter(chapter_no, plan, store):
    # Draft each planned section separately to keep long chapters more controllable.
    novel_outline = read_file_safe(f"{OUTLINE_DIR}/novel_outline.txt", "暂无总纲")
    arc_outline = read_file_safe(f"{OUTLINE_DIR}/arc_outline.txt", "暂无卷纲")
    characters_text = read_file_safe(f"{SETTING_DIR}/characters.txt", "暂无角色设定")
    style_profile = load_style_profile()
    character_state = load_character_state()
    memory_context = build_memory_context(store, chapter_no)

    full_text = []
    for index, section in enumerate(plan.get("sections", []), start=1):
        existing_text = "\n\n".join(full_text).strip() or "（暂无已写内容）"
        system_prompt = "你是小说分段写作引擎。请只写当前段落的正文，不要写标题，不要解释。"
        user_prompt = f"""
当前正在写第 {chapter_no} 章的第 {index} 段。

【总纲】
{novel_outline}

【卷纲】
{arc_outline}

【角色设定】
{characters_text}

【人物状态】
{serialize_compact(character_state)}

【文风配置】
{style_profile_text(style_profile)}

【记忆检索】
{memory_context}

【整章计划】
{serialize_compact(plan)}

【已完成正文】
{existing_text}

【当前段落要求】
名称：{section.get("name", f"第{index}段")}
目标：{section.get("purpose", "")}
节拍：{section.get("beats", "")}
注意事项：{section.get("caution", "")}

请写出这一段正文，并自然衔接上一段。
"""
        segment_text = call_llm(system_prompt, user_prompt, temperature=0.75).strip()
        if segment_text:
            full_text.append(segment_text)

    chapter_text = "\n\n".join(full_text).strip()
    if chapter_text:
        write_text(f"{CHAPTER_DIR}/chapter_{chapter_no:03d}.txt", chapter_text)
        print(f"   [文件写入] 第 {chapter_no} 章已保存。")
    return chapter_text


def run_quality_check(chapter_no, chapter_text, plan):
    # QA runs as a separate pass so weak chapters can be revised without changing the planner.
    fallback = {
        "score": 75,
        "issues": ["未发现结构化结果，建议人工复核"],
        "strengths": ["完成了章节生成"],
        "revision_advice": ["增强结尾钩子和人物状态变化"],
        "pass": True,
    }
    system_prompt = "你是网文章节质检编辑，专门检查长篇连载章节的稳定性和追更感。"
    user_prompt = f"""
请检查第 {chapter_no} 章，并输出 JSON。

【章节计划】
{serialize_compact(plan)}

【正文】
{chapter_text}

检查维度：
1. 有没有跑偏或空转
2. 人设和关系是否一致
3. 节奏是否拖沓
4. 伏笔或设定有没有矛盾
5. 结尾是否有钩子

字段必须包含：
score, issues, strengths, revision_advice, pass
"""
    report = call_llm_json(system_prompt, user_prompt, fallback=fallback)
    for key, value in fallback.items():
        report.setdefault(key, value)
    # Keep QA artifacts on disk so generation decisions stay inspectable.
    write_json(f"{QA_DIR}/chapter_{chapter_no:03d}_qa.json", report)

    lines = [
        f"第 {chapter_no} 章质检报告",
        f"- 分数：{report.get('score')}",
        f"- 是否通过：{report.get('pass')}",
        "- 优点：",
        *[f"  - {item}" for item in ensure_list(report.get("strengths"))],
        "- 问题：",
        *[f"  - {item}" for item in ensure_list(report.get("issues"))],
        "- 修改建议：",
        *[f"  - {item}" for item in ensure_list(report.get("revision_advice"))],
    ]
    write_text(f"{QA_DIR}/chapter_{chapter_no:03d}_qa.txt", "\n".join(lines).strip() + "\n")
    return report


def revise_chapter_if_needed(chapter_no, chapter_text, plan, report):
    # Only spend another LLM call when the QA pass says the chapter is not good enough yet.
    issues = ensure_list(report.get("issues"))
    advice = ensure_list(report.get("revision_advice"))
    score = report.get("score", 0)
    if report.get("pass") and score >= 80:
        return chapter_text
    system_prompt = "你是小说精修编辑，请根据质检报告做最小必要修改，保留已有亮点。"
    user_prompt = f"""
请根据以下质检反馈修改第 {chapter_no} 章正文。

【章节计划】
{serialize_compact(plan)}

【问题列表】
{chr(10).join(f"- {item}" for item in issues) if issues else "暂无"}

【修改建议】
{chr(10).join(f"- {item}" for item in advice) if advice else "暂无"}

【原正文】
{chapter_text}

请直接输出修订后的完整正文。
"""
    revised = call_llm(system_prompt, user_prompt, temperature=0.65).strip()
    if revised:
        write_text(f"{CHAPTER_DIR}/chapter_{chapter_no:03d}.txt", revised)
        print(f"   [修订] 第 {chapter_no} 章已根据质检报告修订。")
        return revised
    return chapter_text


def summarize_chapter_structured(chapter_no, text):
    fallback = {
        "core_plot": "本章完成了一次主线推进。",
        "key_conflict": "本章存在显性冲突。",
        "foreshadowing": [],
        "state_changes": [],
        "plot_memories": [],
        "character_updates": [],
        "setting_updates": [],
    }
    character_state = load_character_state()
    system_prompt = "你是连载小说情报编辑，负责把正文拆成可追踪的结构化记忆。"
    user_prompt = f"""
请基于第 {chapter_no} 章正文输出 JSON。

【现有人物状态】
{serialize_compact(character_state)}

【正文】
{text}

字段必须包含：
core_plot, key_conflict, foreshadowing, state_changes, plot_memories, character_updates, setting_updates
"""
    structured = call_llm_json(system_prompt, user_prompt, fallback=fallback)
    for key, value in fallback.items():
        structured.setdefault(key, value)
    return structured


def update_character_state(structured):
    # Merge partial chapter updates into the long-lived character state snapshot.
    current = load_character_state()
    current_map = {item.get("name"): item for item in current.get("characters", []) if item.get("name")}
    for update in structured.get("character_updates", []):
        name = update.get("name")
        if not name:
            continue
        current_map.setdefault(name, {"name": name})
        # Each chapter may emit only changed fields, so merge instead of replace.
        current_map[name].update(update)
    updated = {"characters": list(current_map.values())}
    save_character_state(updated)
    return updated


def persist_structured_memory(chapter_no, structured, store):
    # Mirror structured memory into both JSON files and the vector store for later retrieval.
    summary_lines = [
        f"【核心剧情】{structured.get('core_plot', '')}",
        f"【关键冲突】{structured.get('key_conflict', '')}",
        "【伏笔/信息】",
        *[f"- {item}" for item in ensure_list(structured.get("foreshadowing"))],
        "【状态变更】",
        *[f"- {item}" for item in ensure_list(structured.get("state_changes"))],
    ]
    write_text(f"{SUMMARY_DIR}/chapter_{chapter_no:03d}_summary.txt", "\n".join(summary_lines).strip() + "\n")

    plot_entries = ensure_list(structured.get("plot_memories"))
    char_updates = structured.get("character_updates", [])
    setting_updates = ensure_list(structured.get("setting_updates"))
    foreshadowing = ensure_list(structured.get("foreshadowing"))

    for item in plot_entries:
        store.add(f"[第{chapter_no}章剧情] {item}", category="plot", chapter=chapter_no)
    for item in foreshadowing:
        store.add(f"[第{chapter_no}章伏笔] {item}", category="foreshadow", chapter=chapter_no)
    for item in setting_updates:
        store.add(f"[第{chapter_no}章设定] {item}", category="setting", chapter=chapter_no)
    for item in char_updates:
        name = item.get("name", "未知角色")
        snippet = f"{name} | 情绪:{item.get('emotion', '')} | 立场:{item.get('stance', '')} | 位置:{item.get('location', '')} | 变化:{item.get('recent_change', '')}"
        store.add(f"[第{chapter_no}章人物] {snippet}", category="character", chapter=chapter_no)

    append_json_list(PLOT_MEMORY_FILE, [{"chapter": chapter_no, "content": item} for item in plot_entries])
    append_json_list(SETTING_MEMORY_FILE, [{"chapter": chapter_no, "content": item} for item in setting_updates])
    append_json_list(FORESHADOW_MEMORY_FILE, [{"chapter": chapter_no, "status": "open", "content": item} for item in foreshadowing])
    update_character_state(structured)
    print("   [记忆] 结构化剧情、人物、设定与伏笔已归档。")


def summarize_and_archive(chapter_no, text, store):
    if not text:
        return
    # Archive only the final post-QA chapter text.
    structured = summarize_chapter_structured(chapter_no, text)
    persist_structured_memory(chapter_no, structured, store)


def write_chapter(chapter_no, store):
    # Main chapter pipeline: plan -> draft -> QA -> optional revision.
    plan = generate_chapter_plan(chapter_no, store)
    chapter_text = write_segmented_chapter(chapter_no, plan, store)
    if not chapter_text:
        return ""
    report = run_quality_check(chapter_no, chapter_text, plan)
    chapter_text = revise_chapter_if_needed(chapter_no, chapter_text, plan, report)
    return chapter_text


def polish_chapter(chapter_no):
    content = read_file_safe(f"{CHAPTER_DIR}/chapter_{chapter_no:03d}.txt")
    if not content:
        return "找不到该章节。"
    style_profile = load_style_profile()
    print(f"正在润色第 {chapter_no} 章...")
    system_prompt = "你是一位金牌小说编辑，负责提升文笔和阅读吸引力。"
    user_prompt = f"""
请润色以下正文，保留剧情信息不变，但优化节奏、语感和画面感。

【文风配置】
{style_profile_text(style_profile)}

【正文】
{content}
"""
    polished_text = call_llm(system_prompt, user_prompt, temperature=0.65)
    if polished_text:
        out_path = f"{CHAPTER_DIR}/chapter_{chapter_no:03d}_polished.txt"
        write_text(out_path, polished_text)
        return f"润色完成: {out_path}"
    return "润色失败。"


def show_help():
    print(
        """
================ 框架搭建 ================
/outline <idea>         - 一键生成小说蓝图、大纲、角色设定、文风配置
/blueprint              - 查看当前小说蓝图
/state                  - 查看当前人物状态卡
/style                  - 查看当前文风配置

================ 章节创作 ================
/plan <n>               - 只生成第 n 章写作计划
/new <n>                - 生成第 n 章
                          流程: 计划 -> 分段写作 -> 质检 -> 修订 -> 记忆归档
/next                   - 自动续写下一章
/auto <n>               - 连续自动生成 n 章
/polish <n>             - 润色第 n 章

================ 内容查看 ================
/show <n>               - 查看第 n 章正文
/showplan <n>           - 查看第 n 章计划
/showqa <n>             - 查看第 n 章质检报告
/stat                   - 查看已完成章节、累计字数、Token 统计

================ 系统命令 ================
/help                   - 显示本帮助菜单
/switch                 - 切换模型: Grok / DeepSeek
/exit                   - 退出程序
        """
    )


def print_file_if_exists(filepath, empty_text):
    if os.path.exists(filepath):
        print(f"\n{read_file_safe(filepath)}\n")
    else:
        print(empty_text)


def main():
    # CLI shell around the story pipeline; commands mostly map to one workflow step.
    global LLM_PROVIDER
    store = VectorStore()
    print("====================================")
    print(f"系统就绪 | 当前模型: {LLM_PROVIDER}")
    print("输入 /help 查看指令列表")
    print("====================================")

    while True:
        try:
            cmd_input = input(">>> ").strip()
            if not cmd_input:
                continue
            cmd = cmd_input.lower()

            if cmd == "/help":
                show_help()

            elif cmd.startswith("/outline "):
                idea = cmd_input[len("/outline ") :].strip()
                if not idea:
                    print("!!! 请输入一句话创意，例如：/outline 现代都市背景下的长篇复仇成长文")
                    continue
                blueprint = generate_novel_framework(idea)
                print(f"   [完成] 已生成小说框架：{blueprint.get('title', '未命名小说')}")

            elif cmd.startswith("/plan "):
                try:
                    chapter_no = int(cmd_input.split()[-1])
                    generate_chapter_plan(chapter_no, store)
                except ValueError:
                    print("!!! 输入错误，请输入数字。")

            elif cmd.startswith("/new "):
                try:
                    chapter_no = int(cmd_input.split()[-1])
                    text = write_chapter(chapter_no, store)
                    if text:
                        summarize_and_archive(chapter_no, text, store)
                        print(f"   [成功] 第 {chapter_no} 章处理完毕。")
                except ValueError:
                    print("!!! 输入错误，请输入数字。")

            elif cmd == "/next":
                chapter_no = get_next_chapter_no()
                print(f"准备开始第 {chapter_no} 章...")
                text = write_chapter(chapter_no, store)
                if text:
                    summarize_and_archive(chapter_no, text, store)
                    print(f"   [成功] 第 {chapter_no} 章生成完毕。")

            elif cmd.startswith("/auto "):
                try:
                    count = int(cmd_input.split()[-1])
                except ValueError:
                    print("!!! 用法: /auto 3")
                    continue
                for index in range(count):
                    chapter_no = get_next_chapter_no()
                    print(f"\n[任务 {index + 1}/{count}] 第 {chapter_no} 章")
                    text = ""
                    for _ in range(2):
                        text = write_chapter(chapter_no, store)
                        if text:
                            break
                        time.sleep(5)
                    if not text:
                        break
                    summarize_and_archive(chapter_no, text, store)
                    if index < count - 1:
                        time.sleep(3)

            elif cmd.startswith("/polish "):
                try:
                    chapter_no = int(cmd_input.split()[-1])
                    print(polish_chapter(chapter_no))
                except ValueError:
                    print("!!! 输入错误，请输入数字。")

            elif cmd.startswith("/showplan "):
                try:
                    chapter_no = int(cmd_input.split()[-1])
                    print_file_if_exists(f"{PLAN_DIR}/chapter_{chapter_no:03d}_plan.txt", "还没有这个章节的计划文件。")
                except ValueError:
                    print("!!! 输入错误，请输入数字。")

            elif cmd.startswith("/showqa "):
                try:
                    chapter_no = int(cmd_input.split()[-1])
                    print_file_if_exists(f"{QA_DIR}/chapter_{chapter_no:03d}_qa.txt", "还没有这个章节的质检报告。")
                except ValueError:
                    print("!!! 输入错误，请输入数字。")

            elif cmd.startswith("/show "):
                try:
                    chapter_no = int(cmd_input.split()[-1])
                    print_file_if_exists(f"{CHAPTER_DIR}/chapter_{chapter_no:03d}.txt", "找不到该章节。")
                except ValueError:
                    print("!!! 输入错误，请输入数字。")

            elif cmd == "/blueprint":
                blueprint = load_blueprint()
                if blueprint:
                    print(f"\n{serialize_compact(blueprint)}\n")
                else:
                    print("当前还没有小说蓝图，请先使用 /outline <idea>。")

            elif cmd == "/state":
                state = load_character_state()
                if state.get("characters"):
                    print(f"\n{serialize_compact(state)}\n")
                else:
                    print("当前还没有人物状态，请先生成框架或章节。")

            elif cmd == "/style":
                print(f"\n{serialize_compact(load_style_profile())}\n")

            elif cmd == "/stat":
                files = [name for name in os.listdir(CHAPTER_DIR) if name.endswith(".txt") and "_polished" not in name]
                total_chars = sum(len(read_file_safe(os.path.join(CHAPTER_DIR, name))) for name in files)
                print(f"完成: {len(files)} 章 | 累计: {total_chars} 字 | 本次Token: {TOTAL_TOKENS_USED}")

            elif cmd == "/switch":
                LLM_PROVIDER = "deepseek" if LLM_PROVIDER == "grok" else "grok"
                print(f"已切换至: {LLM_PROVIDER}")

            elif cmd == "/exit":
                break

            else:
                print("未知命令，输入 /help 查看可用指令。")

        except Exception as error:
            print(f"错误: {error}")


if __name__ == "__main__":
    main()
