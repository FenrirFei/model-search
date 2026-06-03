# ModelSearch 🔍

AI 模型搜尋引擎 — 中英雙語，自然語言搜尋。從 HuggingFace 同步模型資料，Fuse.js 客戶端搜尋。

## 功能

- 🔍 **中英雙語搜尋** — 輸入「語音辨識」或「whisper」都能找到
- 🏷️ **類別篩選** — LLM / TTS / ASR / 多模態 / 圖片
- 📊 **177+ 模型** — 從 HF 熱門排行自動同步
- ⚡ **零後端** — 純靜態網站，Fuse.js 模糊搜尋
- 🌐 **中文化** — pipeline tag 全部翻譯為中文

## 技術棧

```
index.html      ← Google-like 搜尋介面
models.json     ← 模型資料庫（HF API 同步）
Fuse.js         ← 客戶端模糊搜尋引擎
Cloudflare Pages ← 部署（免費，無限流量）
```

## 部署

```bash
# 本地開發
python3 -m http.server 8890

# 部署到 Cloudflare Pages
# 直接連接 GitHub repo，自動部署
```

## 下一步

- [ ] 擴充模型資料庫到 5000+
- [ ] 加入 giscus 討論串（每個模型頁面可留言）
- [ ] 語意搜尋（transformers.js embedding）
- [ ] RSS feed（依類別訂閱新模型）
- [ ] GitHub Pages 自動部署

---

Made with ❤️ by [FenrirFei](https://github.com/FenrirFei)
