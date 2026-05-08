import os
import json
import time
import random
import datetime
import requests
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from transformers import DistilBertModel, DistilBertTokenizer
from PIL import Image
import io
import numpy as np
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
load_dotenv()

TARGET_PER_CATEGORY = 10  # Number of videos to fetch per category
DELAY_BETWEEN_CALLS = 3   # Seconds to wait between API calls to prevent rate-limits

TARGET_MEAN = 7.670568273844549
TARGET_STD  = 1.6172140104057109
MODEL_PATH  = "v9_abl_flat.pth"
LOG_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions_log.json")

CATEGORIES = {
    "Gaming":         "20",
    "Music":          "10",
    "Sports":         "17",
    "Education":      "27",
    "Science & Tech": "28",
    "Travel":         "19",
    "Comedy":         "23",
}

CATEGORY_QUERIES = {
    "20":  ["gaming", "gameplay", "lets play", "playthrough", "game"],
    "10":  ["music video", "official audio", "new song", "official video"],
    "17":  ["highlights", "sports", "match", "game day", "high school basketball", "amateur football", "local tournament", "sunday league", "golf vlog"],
    "27":  ["tutorial", "explained", "learn", "how to"],
    "28":  ["tech review", "unboxing", "technology", "review"],
    "19":  ["travel vlog", "travel", "exploring"],
    "23":  ["funny", "comedy", "prank", "try not to laugh"],
}

API_KEYS = [k for k in [
    os.getenv("YT_API_KEY_1"), os.getenv("YT_API_KEY_2"),
    os.getenv("YT_API_KEY_3"), os.getenv("YT_API_KEY_4"),
    os.getenv("YT_API_KEY_5"), os.getenv("YT_API_KEY_6"),
] if k and "your_" not in str(k)]

# ─────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────
class FlatConcatV9(nn.Module):
    def __init__(self):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for p in self.distilbert.parameters(): p.requires_grad = False
        for p in self.distilbert.transformer.layer[-1].parameters(): p.requires_grad = True
        for p in self.distilbert.transformer.layer[-2].parameters(): p.requires_grad = True

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for p in resnet.parameters(): p.requires_grad = False
        for p in resnet.layer4.parameters(): p.requires_grad = True

        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        self.text_proj = nn.Linear(768, 256)
        self.img_proj  = nn.Linear(2048, 256)

        self.fusion = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.reg_head = nn.Linear(64, 1)

    def forward(self, input_ids, attention_mask, image):
        hidden = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)[0]
        mask_exp = attention_mask.unsqueeze(-1).float()
        text_feat = ((hidden * mask_exp).sum(1) / mask_exp.sum(1))
        text_proj = self.text_proj(text_feat)
        img_feat = self.resnet_features(image).view(image.size(0), -1)
        img_proj = self.img_proj(img_feat)
        fused = self.fusion(torch.cat([text_proj, img_proj], dim=1))
        return self.reg_head(fused).squeeze(1)

val_tfm = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
class KeyRotator:
    def __init__(self, keys): self.keys=keys; self.idx=0; self.exhausted=set()
    def get(self): return self.keys[self.idx] if self.keys else None
    def rotate(self):
        self.exhausted.add(self.idx)
        for i in range(len(self.keys)):
            if i not in self.exhausted: self.idx=i; return True
        return False
    def call(self, url, params):
        if not self.keys: return None,"No API keys configured"
        for _ in range(len(self.keys)):
            params['key'] = self.get()
            try:
                r = requests.get(url,params=params,timeout=10)
                if r.status_code==200: return r.json(),None
                if r.status_code in [403,429]:
                    if not self.rotate(): return None,"All API keys exhausted"
                else: return None,f"API error {r.status_code}"
            except Exception as e: return None,str(e)
        return None,"All keys failed"

def accuracy_info(pred, actual):
    if actual==0: return "N/A", "—"
    r = pred/actual
    ratio_str = f"{r:.2f}×"
    if 0.5<=r<=2.0:   return "✅ ON TARGET", ratio_str
    if 0.25<=r<=4.0:  return "⚠️ CLOSE", ratio_str
    return "❌ OFF", ratio_str

# ─────────────────────────────────────────────
# MAIN AUTOMATION SCRIPT
# ─────────────────────────────────────────────
def main():
    print(f"🚀 Starting Auto-Fetcher...")
    print(f"🎯 Target: {TARGET_PER_CATEGORY} videos per selected category.")
    
    # 1. Load Model
    print("Loading model and tokenizer...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = FlatConcatV9().to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
        model.eval()
        print("✅ Model loaded successfully.")
    else:
        print(f"❌ ERROR: Model {MODEL_PATH} not found.")
        return

    # 2. Interactive Menu
    print("\n========================================")
    print("📋 SELECT A CATEGORY TO FETCH")
    print("========================================")
    category_list = list(CATEGORIES.keys())
    for i, cat in enumerate(category_list, 1):
        print(f"  [{i}] {cat}")
    print(f"  [0] ALL Categories")
    
    try:
        choice = int(input("\nEnter your choice (0-7): "))
    except ValueError:
        print("Invalid input. Please run again and enter a number.")
        return

    if choice == 0:
        selected_categories = CATEGORIES
    elif 1 <= choice <= len(category_list):
        selected_label = category_list[choice - 1]
        selected_categories = {selected_label: CATEGORIES[selected_label]}
    else:
        print("Invalid choice. Please run again.")
        return

    rotator = KeyRotator(API_KEYS)
    
    # 3. Load Existing Log to prevent duplicates across runs
    existing_log = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                existing_log = json.load(f)
        except Exception: pass
    
    seen_video_ids = {e.get("video_id") for e in existing_log if e.get("video_id")}
    
    # 4. Iterate through selected categories
    for cat_label, cat_id in selected_categories.items():
        print(f"\n========================================")
        print(f"📂 FETCHING: {cat_label}")
        print(f"========================================")
        
        success_count = 0
        attempts = 0
        max_attempts = 20 # Hard limit of 20 attempts
        
        while success_count < TARGET_PER_CATEGORY and attempts < max_attempts:
            attempts += 1
            time.sleep(DELAY_BETWEEN_CALLS) # Safeguard against API rate-limiting
            
            query = random.choice(CATEGORY_QUERIES.get(cat_id, ["video"]))
            search_params = {
                "part": "snippet", "type": "video", "q": query, "order": "date",
                "publishedAfter": (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=14)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                "maxResults": 15, "videoDuration": "medium", "regionCode": "US", 
                "relevanceLanguage": "en", "videoCategoryId": cat_id
            }

            data, err = rotator.call("https://www.googleapis.com/youtube/v3/search", search_params)
            if err or not data or "items" not in data:
                print(f"  ⚠️ Search failed ({err}) - Retrying...")
                continue
                
            items = data.get("items", [])
            random.shuffle(items)
            
            video_processed_in_batch = False
            for item in items:
                vid_id = item["id"].get("videoId")
                channel_id = item["snippet"].get("channelId")
                
                # Skip invalid or already seen videos
                if not vid_id or not channel_id or vid_id in seen_video_ids: continue
                
                # Check channel subs (5k to 100k)
                ch_data, err = rotator.call("https://www.googleapis.com/youtube/v3/channels", {"part":"statistics","id":channel_id})
                if err or not ch_data or not ch_data.get("items"): continue
                subs = int(ch_data["items"][0]["statistics"].get("subscriberCount",0))
                if not (5_000 <= subs <= 100_000): continue

                # Check video stats
                vid_data, err = rotator.call("https://www.googleapis.com/youtube/v3/videos", {"part":"snippet,statistics","id":vid_id})
                if err or not vid_data or not vid_data.get("items"): continue

                vitem = vid_data["items"][0]
                views = int(vitem["statistics"].get("viewCount",0))
                title = vitem["snippet"].get("title","")
                
                if views == 0 or "#shorts" in title.lower() or "#short" in title.lower(): continue
                if sum(1 for c in title if ord(c) > 127) > len(title) * 0.2: continue # Remove heavy non-ascii

                pub_dt = datetime.datetime.strptime(vitem["snippet"].get("publishedAt",""), '%Y-%m-%dT%H:%M:%SZ')
                days_live = max(1, (datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) - pub_dt).days)
                actual_vpd = views / days_live

                # Download Thumbnail
                thumbs = vitem["snippet"].get("thumbnails",{})
                thumb_url = (thumbs.get("maxres") or thumbs.get("high") or thumbs.get("medium",{})).get("url","")
                if not thumb_url: continue
                
                try:
                    r = requests.get(thumb_url, timeout=10)
                    pil_img = Image.open(io.BytesIO(r.content)).convert('RGB')
                except: continue

                # --- PREDICTION ---
                img_t = val_tfm(pil_img).unsqueeze(0).to(device)
                enc = tokenizer(title.strip(), add_special_tokens=True, max_length=64, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
                
                with torch.no_grad():
                    pred_norm = model(enc['input_ids'].to(device), enc['attention_mask'].to(device), img_t).item()
                
                pred_vpd = float(np.expm1(pred_norm * TARGET_STD + TARGET_MEAN))
                
                # Log Data
                label, ratio = accuracy_info(pred_vpd, actual_vpd)
                
                new_entry = {
                    "timestamp":   datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "mode":        "surprise",
                    "title":       title,
                    "pred_vpd":    round(pred_vpd, 2),
                    "actual_vpd":  round(actual_vpd, 2),
                    "video_id":    vid_id,
                    "channel":     vitem["snippet"].get("channelTitle",""),
                    "subs":        subs,
                    "accuracy":    label,
                    "ratio":       ratio,
                    "yt_url":      f"https://youtube.com/watch?v={vid_id}",
                    "category":    cat_label,
                }
                
                existing_log.append(new_entry)
                
                # Atomic save
                tmp = LOG_FILE + ".tmp"
                with open(tmp, 'w') as f: json.dump(existing_log, f, indent=2)
                os.replace(tmp, LOG_FILE)
                
                seen_video_ids.add(vid_id)
                success_count += 1
                video_processed_in_batch = True
                
                print(f"  [{success_count}/{TARGET_PER_CATEGORY}] ✅ {title[:40]}... | Pred: {int(pred_vpd)} | Actual: {int(actual_vpd)}")
                break # Move back to outer search loop to avoid overloading one specific keyword/date search
            
            if not video_processed_in_batch:
                print(f"  [Attempt {attempts}/20] No suitable videos found in this search batch.")

        if success_count < TARGET_PER_CATEGORY:
            print(f"  ⚠️ Reached max attempts (20). Found {success_count} out of {TARGET_PER_CATEGORY} videos.")

    print(f"\n🎉 Automation Complete! Check your Streamlit app's Accuracy page to see the updated dataset.")

if __name__ == "__main__":
    main()