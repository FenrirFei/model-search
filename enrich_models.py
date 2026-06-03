#!/usr/bin/env python3
"""Fetch full model info from HF API list endpoint and merge into models.json."""
import json, requests, time, os
from tqdm import tqdm

HF_API = "https://huggingface.co/api"

with open("models.json") as f:
    data = json.load(f)

models = data.get("models", data)
embeddings = data.get("embeddings", [])

print(f"Loading {len(models)} models...")

# Build existing ID set
existing_ids = {m["id"] for m in models}

# Fetch all models from HF API with full=true, sorted by downloads
print("Fetching full model data from HF API...")
fetched = {}
url = f"{HF_API}/models?sort=downloads&direction=-1&limit=100&full=true"
pages = 0

while url and pages < 55:  # 55 pages × 100 = 5500 models
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            print(f"  Error: {resp.status_code}")
            break
        
        batch = resp.json()
        for m in batch:
            mid = m.get("id", "")
            if mid in existing_ids:
                fetched[mid] = m
        
        pages += 1
        
        # Pagination
        url = None
        link = resp.headers.get("Link", "")
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part.split(";")[0].strip(" <>")
                break
        
        if pages % 10 == 0:
            print(f"  Page {pages}: {len(fetched)} matched so far...")
        time.sleep(0.1)
        
    except Exception as e:
        print(f"  Exception: {e}")
        break

print(f"\nMatched {len(fetched)}/{len(models)} models from API")

# Merge fields
updated = 0
for m in models:
    api = fetched.get(m["id"])
    if not api:
        continue
    
    changed = False
    
    # License - from tags (HF stores licenses as tags: "license:apache-2.0")
    if not m.get("lic") or m.get("lic") == "":
        api_tags = api.get("tags", [])
        licenses = [t.replace("license:", "") for t in api_tags if t.startswith("license:")]
        if licenses:
            m["lic"] = " ".join(licenses[:3])
            changed = True
    
    # Library
    if not m.get("lib") and api.get("library_name"):
        m["lib"] = api["library_name"]
        changed = True
    
    # Safetensors
    if not m.get("sf") and api.get("safetensors"):
        sf = api["safetensors"]
        if isinstance(sf, dict):
            m["sf"] = sf.get("total", 0) or 0
        changed = True
    
    # Size from siblings
    if not m.get("sz") and api.get("siblings"):
        total = sum(s.get("size", 0) or 0 for s in api["siblings"])
        if total > 1e12:
            m["sz"] = f"{total/1e12:.1f}T"
        elif total > 1e9:
            m["sz"] = f"{total/1e9:.1f}G"
        elif total > 1e6:
            m["sz"] = f"{total/1e6:.0f}M"
        if m["sz"]:
            changed = True
    
    # Description
    if not m.get("desc") and api.get("description"):
        m["desc"] = str(api["description"])[:200]
        changed = True
    
    if changed:
        updated += 1

print(f"Updated {updated} models with new fields")

# Stats
lic_counts = {}
for m in models:
    lic = m.get("lic", "") or "未知"
    lic_counts[lic] = lic_counts.get(lic, 0) + 1

print(f"\nLicense distribution (top 10):")
for lic, count in sorted(lic_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"  {lic}: {count}")

lib_counts = {}
for m in models:
    lib = m.get("lib", "") or "未知"
    lib_counts[lib] = lib_counts.get(lib, 0) + 1

print(f"\nLibrary distribution (top 10):")
for lib, count in sorted(lib_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"  {lib}: {count}")

# Save
output = {
    "version": "v0.3",
    "updated": data.get("updated", "2026-06-03T00:00:00Z"),
    "count": len(models),
    "models": models,
    "embeddings": embeddings,
}

with open("models.json", "w") as f:
    json.dump(output, f, ensure_ascii=False, separators=(",", ":"))

size_mb = os.path.getsize("models.json") / 1e6
print(f"\nSaved models.json ({size_mb:.1f}MB)")
