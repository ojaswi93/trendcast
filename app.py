import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from PIL import Image
from transformers import DistilBertModel, DistilBertTokenizer
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import requests
import random
import json
import os
import io
import datetime
import pandas as pd
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

# Import our separate page UI functions
from predictor_page import render_predictor
from accuracy_page import render_accuracy_tracker

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TrendCast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

TARGET_MEAN = 7.670568273844549
TARGET_STD  = 1.6172140104057109
IG_STEPS    = 30
MODEL_PATH  = "v9_abl_flat.pth"
LOG_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "predictions_log.json")

API_KEYS = [k for k in [
    os.getenv("YT_API_KEY_1"), os.getenv("YT_API_KEY_2"),
    os.getenv("YT_API_KEY_3"), os.getenv("YT_API_KEY_4"),
    os.getenv("YT_API_KEY_5"), os.getenv("YT_API_KEY_6"),
] if k and "your_" not in str(k)]

# "Any" and "Food" removed as requested
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
    "17":  ["highlights", "sports", "match", "game day"],
    "27":  ["tutorial", "explained", "learn", "how to"],
    "28":  ["tech review", "unboxing", "technology", "review"],
    "19":  ["travel vlog", "travel", "exploring"],
    "23":  ["funny", "comedy", "prank", "try not to laugh"],
}

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass # Fails silently if style.css isn't created yet

load_css("style.css")

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class FlatConcatV9(nn.Module):

    def __init__(self):
        super().__init__()

        # ── TEXT BACKBONE ─────────────────────
        self.distilbert = DistilBertModel.from_pretrained(
            'distilbert-base-uncased'
        )

        for p in self.distilbert.parameters():
            p.requires_grad = False

        for p in self.distilbert.transformer.layer[-1].parameters():
            p.requires_grad = True

        for p in self.distilbert.transformer.layer[-2].parameters():
            p.requires_grad = True

        # ── IMAGE BACKBONE ────────────────────
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )

        for p in resnet.parameters():
            p.requires_grad = False

        for p in resnet.layer4.parameters():
            p.requires_grad = True

        self.resnet_features = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        # ── PROJECTIONS ───────────────────────
        self.text_proj = nn.Linear(768, 256)
        self.img_proj  = nn.Linear(2048, 256)

        # ── FUSION ────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.reg_head = nn.Linear(64, 1)

    def forward(self, input_ids, attention_mask, image):
        hidden = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[0]
        mask_exp = attention_mask.unsqueeze(-1).float()
        text_feat = ((hidden * mask_exp).sum(1) / mask_exp.sum(1))
        text_proj = self.text_proj(text_feat)
        
        img_feat = self.resnet_features(image).view(image.size(0), -1)
        img_proj = self.img_proj(img_feat)
        
        fused = self.fusion(torch.cat([text_proj, img_proj], dim=1))
        return self.reg_head(fused).squeeze(1)

@st.cache_resource
def load_model_instance():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = FlatConcatV9().to(device)
    
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, tokenizer, device

val_tfm = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ─────────────────────────────────────────────
# GRAD-CAM & INTEGRATED GRADIENTS (Unchanged helpers)
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model=model; self.gradients=None; self.activations=None
        tl = model.resnet_features[-2]
        tl.register_forward_hook(lambda m,i,o: setattr(self,'activations',o.detach()))
        tl.register_full_backward_hook(lambda m,gi,go: setattr(self,'gradients',go[0].detach()))

    def generate(self, ids, mask, image):
        self.model.zero_grad()
        ig = image.requires_grad_(True)
        pred = self.model(ids, mask, ig); pred.sum().backward()
        w   = self.gradients.mean(dim=[2,3],keepdim=True)
        cam = F.relu((w*self.activations).sum(dim=1,keepdim=True))
        cam = F.interpolate(cam,size=(224,224),mode='bilinear',align_corners=False)
        cam = cam.squeeze().cpu().numpy(); cam -= cam.min()
        if cam.max()>0: cam /= cam.max()
        return cam, float(np.expm1(pred.item()*TARGET_STD+TARGET_MEAN))

def make_heatmap_fig(pil_img, cam):
    w,h  = pil_img.size
    cr   = np.array(Image.fromarray(np.uint8(cam*255)).resize((w,h),Image.BILINEAR))/255.0
    hm   = Image.fromarray(np.uint8(cm.get_cmap('jet')(cr)[:,:,:3]*255))
    bl   = Image.blend(pil_img.convert('RGB'),hm,0.5)
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    fig.patch.set_facecolor('#111111')
    for ax in axes: ax.set_facecolor('#111111'); ax.axis('off')
    axes[0].imshow(pil_img); axes[0].set_title('Original',color='#888',fontsize=10,pad=8)
    axes[1].imshow(bl);      axes[1].set_title('Grad-CAM  (red = model focus)',color='#FFD600',fontsize=10,pad=8)
    plt.tight_layout(pad=1.5); return fig

def integrated_gradients(model, tokenizer, ids, mask, image, device):
    model.eval()
    embed_layer = model.distilbert.embeddings.word_embeddings
    actual_embeds = embed_layer(ids).detach()
    base_embeds = embed_layer(torch.zeros_like(ids)).detach()
    grad_sum = torch.zeros_like(actual_embeds)
    alphas = torch.linspace(0, 1, IG_STEPS).to(device)

    for alpha in alphas:
        interp = (base_embeds + alpha * (actual_embeds - base_embeds)).requires_grad_(True)
        db_out = model.distilbert(attention_mask=mask, inputs_embeds=interp)
        hidden = db_out.last_hidden_state
        mask_exp = mask.unsqueeze(-1).float()
        text_feat = ((hidden * mask_exp).sum(1) / mask_exp.sum(1))
        text_proj = model.text_proj(text_feat)
        img_feat = model.resnet_features(image).view(image.size(0), -1).detach()
        img_proj = model.img_proj(img_feat)
        fused = model.fusion(torch.cat([text_proj, img_proj], dim=1))
        pred = model.reg_head(fused).squeeze(1)
        grad = torch.autograd.grad(pred.sum(), interp)[0]
        grad_sum += grad.detach()

    avg_grads = grad_sum / IG_STEPS
    attr = ((avg_grads * (actual_embeds - base_embeds)).sum(dim=-1).squeeze(0).cpu().numpy())
    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze(0).cpu().numpy())
    return attr, tokens

def make_attr_fig(attr, tokens):
    valid = [(t.replace('##',''), a) for t, a in zip(tokens, attr) if t not in ['[PAD]','[CLS]','[SEP]','<pad>']]
    if not valid: return None
    labels = [t for t, _ in valid]
    vals   = np.array([a for _, a in valid])
    norm   = Normalize(vmin=vals.min(), vmax=vals.max())
    colors = [plt.cm.RdYlGn(norm(v)) for v in vals]
    fig_h = max(3.5, len(labels) * 0.38)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#111111')
    ax.barh(range(len(labels)), vals, color=colors, edgecolor='#222', linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9, color='#CCCCCC')
    ax.axvline(0, color='#444', linewidth=1, linestyle='--')
    ax.set_xlabel('Attribution Score', fontsize=9, color='#888')
    ax.tick_params(colors='#666', labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor('#333')
    sm = ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm); sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cb.ax.yaxis.set_tick_params(color='#666'); cb.outline.set_edgecolor('#333')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#888', fontsize=8)
    plt.tight_layout(); return fig

def top_tokens(attr, tokens, n=3):
    valid = []
    for t, a in zip(tokens, attr):
        if t in ['[PAD]', '[CLS]', '[SEP]', '<pad>']: continue
        clean = t.replace('##', '')
        if len(clean) < 3: continue
        valid.append((clean, a))
    pos_sorted = sorted(valid, key=lambda x: x[1], reverse=True)
    neg_sorted = sorted(valid, key=lambda x: x[1])
    pos = [t for t, a in pos_sorted if a > 0][:n]
    neg = [t for t, a in neg_sorted if a < 0][:n]
    return pos, neg

def generate_ai_suggestions(title, pred_vpd, pos_words, neg_words, cam, attr, tokens, actual_vpd=None):
    title_sug = []
    thumb_sug = []
    perf_sug = []
    
    title_lower = title.lower()
    valid_attr = [abs(a) for t, a in zip(tokens, attr) if t not in ['[PAD]', '[CLS]', '[SEP]', '<pad>']]
    
    # ── TITLE INSIGHTS ─────────────────────────────────────────
    if len(valid_attr) > 0:
        top_strength = np.mean(sorted(valid_attr, reverse=True)[:5])
        attr_std = np.std(valid_attr)
        strength_ratio = top_strength / (attr_std + 1e-6)
        if strength_ratio < 1.2:
            title_sug.append(("bad", 'Title lacks strong high-impact keywords.'))
            title_sug.append(("warn", 'Consider adding stronger emotional, challenge, curiosity, or outcome-focused wording.'))
        elif strength_ratio < 2.0:
            title_sug.append(("warn", 'Title signal strength appears moderate.'))
            title_sug.append(("warn", 'More emotionally weighted keywords may improve click potential.'))
        else:
            title_sug.append(("good", 'Title contains strong attention-driving language patterns.'))

    if len(pos_words) > 0: title_sug.append(("good", f'Strong keywords detected: {", ".join(pos_words)}.'))
    if len(neg_words) > 0: title_sug.append(("bad", f'Words reducing prediction score: {", ".join(neg_words)}.'))

    wc = len(title.split())
    if wc < 4: title_sug.append(("warn", 'Title may be too short. Add more context or curiosity.'))
    elif wc > 14: title_sug.append(("warn", 'Title may be too long. Shorter titles are often more clickable.'))

    # ── THUMBNAIL INSIGHTS ─────────────────────────────────────
    center_focus = np.mean(cam[80:144, 80:144])
    left_focus = np.mean(cam[:, :70])
    right_focus = np.mean(cam[:, 154:])
    overall_focus = np.mean(cam)
    focus_std = max(np.std(cam), 1e-6)

    if center_focus < 0.20:
        thumb_sug.append(("bad", 'Thumbnail lacks a strong central focal point.'))
        thumb_sug.append(("warn", 'Consider larger faces, objects, or bold text near the center.'))
    if abs(left_focus - right_focus) > 0.15: thumb_sug.append(("warn", 'Visual attention appears uneven across the thumbnail.'))
    if overall_focus < (focus_std * 0.85): thumb_sug.append(("bad", 'Thumbnail may need stronger contrast or clearer subject separation.'))

    # ── PERFORMANCE INSIGHTS ───────────────────────────────────
    if actual_vpd is None:
        if pred_vpd < 10_000: perf_sug.append(("bad", 'Current packaging shows weak click-driving potential.'))
        elif pred_vpd < 100_000: perf_sug.append(("warn", 'Packaging has moderate potential but could be optimized further.'))
        else: perf_sug.append(("good", 'Packaging signals appear strong overall.'))
    else:
        ratio = pred_vpd / max(actual_vpd, 1)
        if ratio < 0.5:
            perf_sug.append(("good", 'Video significantly outperformed model expectations.'))
            perf_sug.append(("warn", 'Possible external virality factors: trends, audience momentum, or creator influence.'))
        elif ratio > 2.0:
            perf_sug.append(("bad", 'Packaging appeared strong but audience response was weaker than expected.'))
            perf_sug.append(("warn", 'Possible mismatch between click appeal and viewer retention.'))
        else:
            perf_sug.append(("good", 'Prediction aligned closely with actual audience response.'))

    # Deduplicate helper
    def dedup(sugs):
        unique, seen = [], set()
        for level, text in sugs:
            if text not in seen:
                unique.append((level, text))
                seen.add(text)
        return unique

    return {
        "title": dedup(title_sug),
        "thumbnail": dedup(thumb_sug),
        "performance": dedup(perf_sug)
    }

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_tier(v):
    if v>=500_000: return "VIRAL","tier-viral"
    if v>=100_000: return "STRONG","tier-strong"
    if v>=10_000:  return "MODERATE","tier-moderate"
    return "LOW","tier-low"

def get_score_cls(v):
    if v>=500_000: return "score-high"
    if v>=100_000: return "score-mid"
    return "score-low"

def fmt(v):
    if v>=1_000_000: return f"{v/1_000_000:.1f}M"
    if v>=1_000:     return f"{v/1_000:.1f}K"
    return str(int(v))

def accuracy_info(pred, actual):
    if actual==0: return "N/A","acc-ok","#FFD600","—"
    r = pred/actual
    ratio_str = f"{r:.2f}×"
    if 0.5<=r<=2.0:   return "✅ ON TARGET","acc-good","#4ADE80", ratio_str
    if 0.25<=r<=4.0:  return "⚠️ CLOSE",   "acc-ok",  "#FFD600", ratio_str
    return "❌ OFF",  "acc-bad","#FF3B3B", ratio_str

def load_log():
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
    except Exception: pass
    return []

def append_prediction(entry):
    log = load_log()
    log.append(entry)
    tmp = LOG_FILE + ".tmp"
    try:
        with open(tmp, 'w') as f: json.dump(log, f, indent=2)
        os.replace(tmp, LOG_FILE)
    except Exception as e:
        st.warning(f"Could not save prediction log: {e}")

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

@st.cache_resource
def get_rotator(): return KeyRotator(API_KEYS)

def fetch_surprise_video(category_id=None):
    rotator    = get_rotator()
    base       = "https://www.googleapis.com/youtube/v3"
    query_list = CATEGORY_QUERIES.get(category_id, ["video"])
    query      = random.choice(query_list)

    search_params = {
        "part":              "snippet",
        "type":              "video",
        "q":                 query,
        "order":             "date",
        "publishedAfter":    (datetime.datetime.utcnow() - datetime.timedelta(days=14)).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "maxResults":        50,
        "videoDuration":     "medium",
        "regionCode":        "US",
        "relevanceLanguage": "en",
    }
    if category_id: search_params["videoCategoryId"] = category_id

    data, err = rotator.call(f"{base}/search", search_params)
    if err or not data or "items" not in data: return None, f"Search failed: {err}"

    items = data.get("items",[])
    if not items: return None,"No videos found — try another category"
    random.shuffle(items)

    for item in items:
        vid_id     = item["id"].get("videoId")
        channel_id = item["snippet"].get("channelId")
        if not vid_id or not channel_id: continue

        ch_data,err = rotator.call(f"{base}/channels", {"part":"statistics","id":channel_id})
        if err or not ch_data or not ch_data.get("items"): continue
        subs = int(ch_data["items"][0]["statistics"].get("subscriberCount",0))
        if not (5_000<=subs<=100_000): continue

        vid_data,err = rotator.call(f"{base}/videos", {"part":"snippet,statistics","id":vid_id})
        if err or not vid_data or not vid_data.get("items"): continue

        vitem   = vid_data["items"][0]
        stats   = vitem["statistics"]
        snippet = vitem["snippet"]
        views   = int(stats.get("viewCount",0))

        title_text = snippet.get("title","")
        title_lower = title_text.lower()
        if "#shorts" in title_lower or "#short" in title_lower: continue
        if views==0: continue

        non_ascii = sum(1 for c in title_text if ord(c) > 127)
        if non_ascii > len(title_text) * 0.2: continue

        published  = snippet.get("publishedAt","")
        pub_dt     = datetime.datetime.strptime(published,'%Y-%m-%dT%H:%M:%SZ')
        days_live  = max(1,(datetime.datetime.utcnow()-pub_dt).days)
        actual_vpd = views/days_live

        thumbs    = snippet.get("thumbnails",{})
        thumb_url = (thumbs.get("maxres") or thumbs.get("high") or thumbs.get("medium",{})).get("url","")
        if not thumb_url: continue

        return {
            "video_id":         vid_id,
            "title":            snippet.get("title",""),
            "thumbnail_url":    thumb_url,
            "channel_title":    snippet.get("channelTitle",""),
            "subscriber_count": subs,
            "view_count":       views,
            "published_at":     published,
            "days_live":        days_live,
            "actual_vpd":       actual_vpd,
            "yt_url":           f"https://youtube.com/watch?v={vid_id}",
        }, None

    return None,"Could not find a suitable video — try a different category or press again"

def download_thumbnail(url):
    try:
        r = requests.get(url,timeout=10)
        return Image.open(io.BytesIO(r.content)).convert('RGB')
    except: return None

def run_prediction(model, tokenizer, device, title, pil_img):
    img_t = val_tfm(pil_img).unsqueeze(0).to(device)
    enc   = tokenizer(title.strip(), add_special_tokens=True, max_length=64,
                      padding='max_length', truncation=True,
                      return_attention_mask=True, return_tensors='pt')
    ids  = enc['input_ids'].to(device)
    mask = enc['attention_mask'].to(device)
    with torch.no_grad(): pred_norm = model(ids, mask, img_t).item()
    return ids, mask, img_t, float(np.expm1(pred_norm*TARGET_STD+TARGET_MEAN))

def render_prediction_metrics(pred_vpd, actual_vpd=None, video_info=None):
    tier, tier_cls = get_tier(pred_vpd)
    sc             = get_score_cls(pred_vpd)

    st.markdown(f"""
    <div class="score-block">
        <div class="score-label">Predicted Views / Day</div>
        <div class="score-number {sc}">{fmt(pred_vpd)}</div>
        <div class="score-unit">views per day at publish</div>
        <div><span class="tier-badge {tier_cls}">{tier}</span></div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)

    if actual_vpd is not None:
        label, acc_cls, color, ratio = accuracy_info(pred_vpd, actual_vpd)
        st.markdown(f"""
        <div class="acc-highlight {acc_cls}">
            <div class="acc-highlight-label">Accuracy vs Actual</div>
            <div class="acc-highlight-value">{label}</div>
        </div>""", unsafe_allow_html=True)

        cp, ca, cr_col = st.columns(3)
        with cp:
            st.markdown(f"""<div class="metric-stack">
                <div class="metric-stack-label">Predicted vpd</div>
                <div class="metric-stack-value">{fmt(pred_vpd)}</div>
            </div>""", unsafe_allow_html=True)
        with ca:
            st.markdown(f"""<div class="metric-stack">
                <div class="metric-stack-label">Actual vpd</div>
                <div class="metric-stack-value" style="color:{color};">{fmt(actual_vpd)}</div>
            </div>""", unsafe_allow_html=True)
        with cr_col:
            st.markdown(f"""<div class="metric-stack">
                <div class="metric-stack-label">Pred / Actual ratio</div>
                <div class="metric-stack-value" style="color:{color};">{ratio}</div>
            </div>""", unsafe_allow_html=True)
    else:
        c7, c30 = st.columns(2)
        with c7:
            st.markdown(f"""<div class="metric-stack">
                <div class="metric-stack-label">Est. views (7 days)</div>
                <div class="metric-stack-value">{fmt(pred_vpd*7)}</div>
            </div>""", unsafe_allow_html=True)
        with c30:
            st.markdown(f"""<div class="metric-stack">
                <div class="metric-stack-label">Est. views (30 days)</div>
                <div class="metric-stack-value">{fmt(pred_vpd*30)}</div>
            </div>""", unsafe_allow_html=True)

    if video_info:
        st.markdown(f"""
        <div style="text-align: center; margin-top: 10px;">
            <a href="{video_info['yt_url']}" target="_blank" class="yt-btn">
                ▶ WATCH ON YOUTUBE
            </a>
        </div>""", unsafe_allow_html=True)

def render_explainability(model, tokenizer, device, gradcam_obj, ids, mask, img_t, pil_img, pred_vpd, title, actual_vpd=None):
    st.markdown('<br><br><div class="section-header">EXPLAINABILITY & AI INSIGHTS</div>', unsafe_allow_html=True)
    
    # ── Charts ──
    exp_c1, exp_c2 = st.columns(2, gap="large")
    
    with exp_c1:
        st.markdown('<div class="exp-header img-header">📸 VISUAL FOCUS (Grad-CAM)</div>', unsafe_allow_html=True)
        with st.spinner('Generating Grad-CAM...'):
            cam, _ = gradcam_obj.generate(ids, mask, img_t)
            fig_c = make_heatmap_fig(pil_img, cam)
            st.pyplot(fig_c, use_container_width=True)
            plt.close(fig_c)
        st.markdown('<p style="font-size:11px;color:#888;text-align:center;">Red/warm regions = visual elements the model found important.</p>', unsafe_allow_html=True)
        
    with exp_c2:
        st.markdown('<div class="exp-header txt-header">📝 TITLE SIGNAL (Integrated Gradients)</div>', unsafe_allow_html=True)
        with st.spinner('Running Integrated Gradients (~5s)...'):
            attr, tokens = integrated_gradients(model, tokenizer, ids, mask, img_t, device)
            pos_w, neg_w = top_tokens(attr, tokens)
            chips = "".join([f'<span class="insight-chip chip-pos">↑ {w}</span>' for w in pos_w] +
                            [f'<span class="insight-chip chip-neg">↓ {w}</span>' for w in neg_w])
            st.markdown(f'<div style="text-align:center;margin-bottom:10px;">{chips}</div>', unsafe_allow_html=True)
            
            fig_ig = make_attr_fig(attr, tokens)
            if fig_ig:
                st.pyplot(fig_ig, use_container_width=True)
                plt.close(fig_ig)
        st.markdown('<p style="font-size:11px;color:#888;text-align:center;">Green bars = positive impact · Red bars = negative impact</p>', unsafe_allow_html=True)

    # ── Suggestions ──
    st.markdown('<br><div class="section-header" style="margin-top:16px;">AI OPTIMIZATION SUGGESTIONS</div>', unsafe_allow_html=True)
    grouped_suggestions = generate_ai_suggestions(title, pred_vpd, pos_w, neg_w, cam, attr, tokens, actual_vpd)
    
    sug_col1, sug_col2, sug_col3 = st.columns(3, gap="large")
    
    def render_suggestion_group(col, header_title, header_class, items):
        with col:
            st.markdown(f'<div class="exp-header {header_class}">{header_title}</div>', unsafe_allow_html=True)
            if not items:
                st.markdown('<div class="suggestion-card" style="color:#666; justify-content:center;">No specific insights.</div>', unsafe_allow_html=True)
            for level, text in items:
                css_class = {"good": "suggestion-good", "warn": "suggestion-warn", "bad": "suggestion-bad"}.get(level, "suggestion-warn")
                st.markdown(f'<div class="suggestion-card {css_class}">{text}</div>', unsafe_allow_html=True)

    render_suggestion_group(sug_col1, "📝 TITLE", "txt-header", grouped_suggestions['title'])
    render_suggestion_group(sug_col2, "📸 THUMBNAIL", "img-header", grouped_suggestions['thumbnail'])
    render_suggestion_group(sug_col3, "📊 PERFORMANCE", "perf-header", grouped_suggestions['performance'])
    
# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Bebas Neue',sans-serif;font-size:36px;
                letter-spacing:3px;color:white;margin-bottom:4px;">TRENDCAST</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                color:#FFD600;letter-spacing:2px;margin-bottom:32px;">
        V9 · MIT-WPU 2026</div>
    """, unsafe_allow_html=True)

    page = st.radio("nav", ["Predictor", "Accuracy Tracker"], label_visibility="hidden")

    st.markdown('<br>', unsafe_allow_html=True)
    ok = len(API_KEYS) > 0
    st.markdown(f"""<div style="font-family:'JetBrains Mono',monospace;font-size:9px;
        color:{'#4ADE80' if ok else '#FF3B3B'};letter-spacing:1px;">
        API KEYS: {len(API_KEYS)} loaded</div>""", unsafe_allow_html=True)

    # Filtering "Any" and "Food"
    log_count = len([e for e in load_log() if e.get("mode")=="surprise" and e.get("actual_vpd") and e.get("category") not in ["Any", "Food"]])
    st.markdown(f"""<div style="font-family:'JetBrains Mono',monospace;font-size:9px;
        color:#444;letter-spacing:1px;margin-top:8px;">
        LOGGED PREDICTIONS: {log_count}</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────

if page == "Predictor":
    with st.spinner('Loading TrendCast V9 Flat Concat...'):
        try:
            model, tokenizer, device = load_model_instance()
            gradcam_obj = GradCAM(model)
        except FileNotFoundError:
            st.error(f'Model not found: `{MODEL_PATH}` — place it in the same folder as app.py')
            st.stop()

    render_predictor(
            model=model, 
            tokenizer=tokenizer, 
            device=device, 
            gradcam_obj=gradcam_obj, 
            fetch_surprise_video=fetch_surprise_video, 
            run_prediction=run_prediction, 
            render_prediction_metrics=render_prediction_metrics, 
            render_explainability=render_explainability, 
            append_prediction=append_prediction, 
            download_thumbnail=download_thumbnail, 
            fmt=fmt, 
            CATEGORIES=CATEGORIES, 
            accuracy_info=accuracy_info
        )
elif page == "Accuracy Tracker":
    render_accuracy_tracker(load_log=load_log, fmt=fmt)


st.markdown("<br><br>", unsafe_allow_html=True)

footer_html = """
<div style="border-top:1px solid #1a1a1a; padding-top:16px; display:flex; justify-content:space-between;">
    <span style="font-family:'JetBrains Mono', monospace; font-size:9px; color:#2a2a2a; letter-spacing:2px;">
        TRENDCAST V9 · MIT-WPU CAPSTONE 2026 · GROUP 31
    </span>
    <span style="font-family:'JetBrains Mono', monospace; font-size:9px; color:#2a2a2a; letter-spacing:1px;">
        Flat Concat Fusion · DistilBERT + ResNet50
    </span>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)