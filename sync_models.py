#!/usr/bin/env python3
"""
ModelSearch — HuggingFace Model Sync + Embedding Pipeline

Usage:
  # Full sync (rebuild everything)
  uv run python sync_models.py --full --limit 5000
  
  # Incremental sync (last 24h changes only)
  uv run python sync_models.py --incremental
  
  # Generate embeddings only (models.json already exists)
  uv run python sync_models.py --embed-only
"""

import json, os, sys, time, argparse
from datetime import datetime, timedelta
import requests
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

HF_API = "https://huggingface.co/api"
MODELS_FILE = "models.json"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Pipeline tags → Chinese
PIPELINE_ZH = {
    "text-generation": "文字生成",
    "text2text-generation": "文字到文字生成",
    "text-classification": "文字分類",
    "token-classification": "標記分類",
    "question-answering": "問答系統",
    "translation": "翻譯",
    "summarization": "摘要生成",
    "feature-extraction": "特徵提取",
    "fill-mask": "填空預測",
    "zero-shot-classification": "零樣本分類",
    "text-to-image": "文字生成圖片",
    "image-to-text": "圖片轉文字",
    "image-classification": "圖片分類",
    "object-detection": "物件偵測",
    "image-segmentation": "圖片分割",
    "automatic-speech-recognition": "語音辨識",
    "text-to-speech": "文字轉語音",
    "audio-classification": "音訊分類",
    "text-to-video": "文字生成影片",
    "video-classification": "影片分類",
    "image-to-video": "圖片轉影片",
    "visual-question-answering": "視覺問答",
    "document-question-answering": "文件問答",
    "image-to-image": "圖片生成圖片",
    "unconditional-image-generation": "無條件圖片生成",
    "text-to-3d": "文字生成3D",
    "image-to-3d": "圖片生成3D",
    "image-feature-extraction": "圖片特徵提取",
    "video-text-to-text": "影片描述文字生成",
    "table-question-answering": "表格問答",
    "table-structure-recognition": "表格結構辨識",
    "keypoint-detection": "關鍵點偵測",
    "mask-generation": "遮罩生成",
    "depth-estimation": "深度估計",
    "reinforcement-learning": "強化學習",
    "robotics": "機器人控制",
    "sentence-similarity": "語句相似度",
}

# Keyword enrichment per pipeline tag
PIPELINE_KW = {
    "text-generation": "LLM 大語言模型 聊天 對話 寫作 code 翻譯 摘要",
    "text2text-generation": "文字轉換 翻譯 摘要 改寫 問答",
    "text-classification": "情感分析 垃圾郵件 意圖分類 情緒辨識",
    "token-classification": "NER 命名實體辨識 詞性標註 關鍵字提取",
    "question-answering": "QA 問答 閱讀理解 知識庫",
    "translation": "翻譯 中英 英中 多語言 在地化",
    "summarization": "摘要 濃縮 長文縮短 會議摘要",
    "feature-extraction": "embedding 嵌入 向量 相似度 RAG 檢索",
    "fill-mask": "完形填空 文法檢查 錯字修正",
    "zero-shot-classification": "零樣本 話題分類 意圖辨識 不需要訓練",
    "text-to-image": "繪圖 AI繪圖 圖片生成 stable-diffusion 海報 設計",
    "image-to-text": "圖片描述 caption 看圖說故事 輔助視覺",
    "image-classification": "圖片分類 辨識 照片分類",
    "object-detection": "物件偵測 yolo 人臉辨識 安全監控",
    "image-segmentation": "圖片分割 去背 語意分割 醫療影像",
    "automatic-speech-recognition": "語音辨識 ASR 語音轉文字 whisper 逐字稿 會議記錄",
    "text-to-speech": "TTS 文字轉語音 語音合成 朗讀 有聲書",
    "audio-classification": "聲音分類 音樂辨識 環境音 鳥類辨識",
    "text-to-video": "文字生成影片 AI影片 動畫生成",
    "video-classification": "影片分類 動作辨識 監控影片",
    "visual-question-answering": "VQA 視覺問答 多模態 圖片問答",
    "document-question-answering": "文件問答 合約分析 發票辨識 表單理解",
    "image-to-image": "圖片編輯 風格轉換 修圖 inpainting",
    "unconditional-image-generation": "隨機生成圖片 藝術創作",
    "text-to-3d": "3D生成 3D模型 建模 AI建模",
    "image-to-3d": "照片轉3D 3D重建",
    "image-feature-extraction": "圖片向量 CLIP 圖片搜尋 相似圖片",
    "sentence-similarity": "語意相似度 同義句 搜尋排序 去重",
    "depth-estimation": "深度估計 3D深度 自動駕駛 距離",
    "reinforcement-learning": "RL 強化學習 遊戲AI 決策",
    "robotics": "機器人 機械手臂 自動化控制",
    "mask-generation": "SAM 圖片分割 背景去除 物件遮罩",
}

# Map HF license strings to short codes
LICENSE_SHORT = {
    "mit": "MIT",
    "apache-2.0": "Apache-2.0",
    "openrail": "OpenRAIL",
    "bigscience-openrail-m": "BigScience-OpenRAIL",
    "creativeml-openrail-m": "CreativeML-OpenRAIL",
    "cc-by-4.0": "CC-BY-4.0",
    "cc-by-nc-4.0": "CC-BY-NC-4.0",
    "cc-by-sa-4.0": "CC-BY-SA-4.0",
    "cc-by-nc-sa-4.0": "CC-BY-NC-SA-4.0",
    "llama2": "Llama2",
    "llama3": "Llama3",
    "llama3.1": "Llama3.1",
    "gemma": "Gemma",
    "other": "Other",
}

def fetch_models(sort="lastModified", limit=5000, since=None):
    """Fetch model list from HF API."""
    url = f"{HF_API}/models?sort={sort}&direction=-1&limit={limit}&full=true"
    if since:
        url += f"&search=lastModified:>{since}"
    
    print(f"Fetching: {url}")
    models = []
    while url:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        models.extend(data)
        # Check for pagination
        url = None
        link = resp.headers.get("Link", "")
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part.split(";")[0].strip(" <>")
                print(f"  Next page: {url}")
                break
        if len(models) >= limit:
            break
    return models[:limit]

def fetch_model_detail(model_id):
    """Fetch detailed info for a single model."""
    try:
        resp = requests.get(f"{HF_API}/models/{model_id}", timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

def extract_model_info(item, detail=None):
    """Extract structured info from HF API model item."""
    model_id = item.get("id", item.get("modelId", ""))
    if not model_id:
        return None
    
    # Pipeline tag
    pipeline = item.get("pipeline_tag", "") or ""
    pipeline_zh = PIPELINE_ZH.get(pipeline, pipeline)
    
    # Downloads & likes
    downloads = item.get("downloads", 0) or 0
    likes = item.get("likes", 0) or 0
    
    # Tags (HF community tags)
    tags = item.get("tags", []) or []
    
    # Size from siblings (approximate)
    size_str = ""
    siblings = item.get("siblings", []) or []
    total_bytes = sum(s.get("size", 0) or 0 for s in siblings)
    if total_bytes > 1e12:
        size_str = f"{total_bytes/1e12:.1f}T"
    elif total_bytes > 1e9:
        size_str = f"{total_bytes/1e9:.1f}G"
    elif total_bytes > 1e6:
        size_str = f"{total_bytes/1e6:.0f}M"
    
    # License
    license_info = item.get("license", []) or []
    if isinstance(license_info, str):
        license_info = [license_info]
    license_short = []
    for lic in license_info:
        if isinstance(lic, str):
            short = LICENSE_SHORT.get(lic.lower(), lic)
            license_short.append(short)
        elif isinstance(lic, dict):
            license_short.append(lic.get("license", "?"))
    
    # From detail endpoint
    library = ""
    safetensors = 0
    params_str = ""
    description = ""
    card_data = {}
    
    if detail:
        library = detail.get("library_name", "") or ""
        safetensors_info = detail.get("safetensors", {}) or {}
        safetensors = safetensors_info.get("total", 0) or 0 if safetensors_info else 0
        config = detail.get("config", {}) or {}
        card_data = detail.get("card_data", {}) or {}
        
        # Extract param count from config
        hidden_size = config.get("hidden_size", 0) or 0
        num_layers = config.get("num_hidden_layers", 0) or 0
        num_heads = config.get("num_attention_heads", 0) or 0
        vocab_size = config.get("vocab_size", 0) or 0
        
        if hidden_size and num_layers:
            # Rough estimate: 12 * n_layers * h_size^2 (transformer params)
            params = 12 * num_layers * hidden_size * hidden_size
            if params > 1e9:
                params_str = f"{params/1e9:.1f}B"
            elif params > 1e6:
                params_str = f"{params/1e6:.0f}M"
        
        # Description from card_data or model card
        description = card_data.get("language", "")
        if isinstance(description, list):
            description = ", ".join(description)
        # Try to get the actual description from tags or elsewhere
        if not description:
            description = item.get("description", "") or ""
    
    # Build keywords from pipeline
    keywords = PIPELINE_KW.get(pipeline, "")
    
    # Build search text for embedding
    search_text = f"{model_id} {pipeline_zh} {pipeline} {' '.join(tags)} {keywords} {library} {description}"
    
    return {
        "id": model_id,
        "d": downloads,
        "l": likes,
        "p": pipeline,
        "pz": pipeline_zh,
        "sz": size_str,
        "pm": params_str,
        "t": tags[:15],  # Cap tags
        "kw": keywords,
        "lib": library,
        "lic": " ".join(license_short[:3]) if license_short else "",
        "sf": safetensors,
        "desc": (description or "")[:200],
    }, search_text

def build_search_texts(models):
    """Build search text strings for each model."""
    texts = []
    for m in models:
        t = f"{m['id']} {m['pz']} {m['p']} {' '.join(m.get('t',[]))} {m.get('kw','')} {m.get('lib','')} {m.get('desc','')} {m.get('pm','')}"
        texts.append(t)
    return texts

def generate_embeddings(texts, batch_size=64):
    """Generate embeddings for search texts (float16 to save space)."""
    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    print(f"Generating embeddings for {len(texts)} models (batch={batch_size})...")
    
    import numpy as np
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        # Convert to float16 for storage efficiency
        emb_f16 = emb.astype(np.float16)
        embeddings.extend(emb_f16.tolist())
    
    return embeddings

def load_existing():
    """Load existing models.json if any."""
    if os.path.exists(MODELS_FILE):
        with open(MODELS_FILE) as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            # New format with embeddings
            if isinstance(data, dict) and "models" in data:
                return data
    return None

def main():
    parser = argparse.ArgumentParser(description="ModelSearch sync pipeline")
    parser.add_argument("--full", action="store_true", help="Full sync from scratch")
    parser.add_argument("--incremental", action="store_true", help="Incremental sync (24h)")
    parser.add_argument("--embed-only", action="store_true", help="Re-generate embeddings only")
    parser.add_argument("--limit", type=int, default=5000, help="Max models to fetch")
    parser.add_argument("--skip-detail", action="store_true", help="Skip per-model detail API calls")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding generation")
    args = parser.parse_args()
    
    existing = load_existing()
    existing_models = []
    existing_map = {}
    
    if isinstance(existing, dict):
        existing_models = existing.get("models", [])
        for m in existing_models:
            existing_map[m["id"]] = m
    elif isinstance(existing, list):
        existing_models = existing
        for m in existing_models:
            existing_map[m["id"]] = m
    
    if args.embed_only and existing_models:
        print(f"Embed-only mode: re-embedding {len(existing_models)} existing models")
        texts = build_search_texts(existing_models)
        embeddings = generate_embeddings(texts)
        output = {
            "version": "v0.3",
            "updated": datetime.utcnow().isoformat() + "Z",
            "count": len(existing_models),
            "models": existing_models,
            "embeddings": embeddings,
        }
        with open(MODELS_FILE, "w") as f:
            json.dump(output, f, ensure_ascii=False, separators=(",", ":"))
        size_mb = os.path.getsize(MODELS_FILE) / 1e6
        print(f"Saved {MODELS_FILE} ({size_mb:.1f}MB) with {len(existing_models)} models + embeddings")
        return
    
    if args.incremental:
        since = (datetime.utcnow() - timedelta(hours=24)).strftime("%Y-%m-%d")
        print(f"Incremental sync since: {since}")
        new_models = fetch_models(sort="lastModified", limit=args.limit, since=since)
    else:
        print(f"Full sync: fetching up to {args.limit} models")
        new_models = fetch_models(sort="downloads", limit=args.limit)
    
    print(f"Fetched {len(new_models)} models from API")
    
    # Process models
    processed = []
    new_count = 0
    
    for item in tqdm(new_models):
        model_id = item.get("id", item.get("modelId", ""))
        
        # Skip private/gated models
        if item.get("private", False) or item.get("gated", False):
            continue
        
        # Check if we need detail
        detail = None
        if not args.skip_detail:
            # Only fetch detail for new/updated models
            should_fetch = model_id not in existing_map
            if should_fetch:
                detail = fetch_model_detail(model_id)
        
        result = extract_model_info(item, detail)
        if result:
            model_info, _ = result
            if model_id not in existing_map:
                new_count += 1
            processed.append(model_info)
    
    print(f"Processed: {len(processed)} models ({new_count} new)")
    
    # Merge with existing
    all_models_map = existing_map.copy()
    for m in processed:
        all_models_map[m["id"]] = m
    
    # Sort by downloads
    all_models = sorted(all_models_map.values(), key=lambda x: x.get("d", 0), reverse=True)
    
    print(f"Total models: {len(all_models)}")
    
    # Generate embeddings
    embeddings = []
    if not args.skip_embed:
        texts = build_search_texts(all_models)
        embeddings = generate_embeddings(texts)
    
    # Output
    output = {
        "version": "v0.3",
        "updated": datetime.utcnow().isoformat() + "Z",
        "count": len(all_models),
        "models": all_models,
        "embeddings": embeddings,
    }
    
    with open(MODELS_FILE, "w") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"))
    
    size_mb = os.path.getsize(MODELS_FILE) / 1e6
    print(f"Saved {MODELS_FILE} ({size_mb:.1f}MB) with {len(all_models)} models + {len(embeddings)} embeddings")

if __name__ == "__main__":
    main()
