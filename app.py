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

CATEGORIES = {
    "Any":            None,
    "Gaming":         "20",
    "Music":          "10",
    "Food":           "26",
    "Sports":         "17",
    "Education":      "27",
    "Science & Tech": "28",
    "Travel":         "19",
    "Comedy":         "23",
}

CATEGORY_QUERIES = {
    "20":  ["gaming", "gameplay", "lets play", "playthrough", "game"],
    "10":  ["music video", "official audio", "new song", "official video"],
    "26":  ["recipe", "cooking", "food", "how to cook"],
    "17":  ["highlights", "sports", "match", "game day"],
    "27":  ["tutorial", "explained", "learn", "how to"],
    "28":  ["tech review", "unboxing", "technology", "review"],
    "19":  ["travel vlog", "travel", "exploring"],
    "23":  ["funny", "comedy", "prank", "try not to laugh"],
    None:  ["vlog", "video", "new", "watch", "today", "challenge"],
}

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');
:root {
    --red:#FF3B3B; --yellow:#FFD600; --dark:#0A0A0A;
    --card:#111111; --border:#222222; --text:#E8E8E8; --muted:#666666;
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:var(--dark);color:var(--text);}
.stApp{background:var(--dark);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:2rem 3rem;max-width:1400px;}
[data-testid="stSidebar"]{background:#0D0D0D!important;border-right:1px solid #1a1a1a!important;}
[data-testid="stSidebar"] *{color:var(--text)!important;}

.hero-title{font-family:'Bebas Neue',sans-serif;font-size:72px;letter-spacing:4px;color:white;line-height:1;margin:0;}
.hero-tag{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--yellow);background:rgba(255,214,0,0.1);border:1px solid rgba(255,214,0,0.3);padding:4px 10px;border-radius:2px;letter-spacing:2px;}
.hero-sub{font-size:14px;color:var(--muted);margin-bottom:32px;}
.divider{height:1px;background:linear-gradient(90deg,var(--red),var(--yellow),transparent);margin:0 0 32px 0;}

.score-block{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:32px;text-align:center;position:relative;overflow:hidden;}
.score-block::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--red),var(--yellow));}
.score-label{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;}
.score-number{font-family:'Bebas Neue',sans-serif;font-size:64px;line-height:1;margin-bottom:8px;}
.score-unit{font-size:13px;color:var(--muted);font-weight:300;}
.score-high{color:#4ADE80;} .score-mid{color:var(--yellow);} .score-low{color:var(--red);}
.tier-badge{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:2px;padding:6px 16px;border-radius:2px;margin-top:16px;font-weight:600;}
.tier-viral{background:rgba(74,222,128,0.15);color:#4ADE80;border:1px solid rgba(74,222,128,0.4);}
.tier-strong{background:rgba(255,214,0,0.15);color:#FFD600;border:1px solid rgba(255,214,0,0.4);}
.tier-moderate{background:rgba(251,146,60,0.15);color:#FB923C;border:1px solid rgba(251,146,60,0.4);}
.tier-low{background:rgba(255,59,59,0.15);color:#FF3B3B;border:1px solid rgba(255,59,59,0.4);}

.section-header{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--yellow);letter-spacing:3px;text-transform:uppercase;margin-bottom:16px;display:flex;align-items:center;gap:10px;}
.section-header::after{content:'';flex:1;height:1px;background:var(--border);}

.metric-stack{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px 20px;margin-bottom:10px;}
.metric-stack-label{font-size:11px;color:var(--muted);margin-bottom:4px;}
.metric-stack-value{font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:600;color:white;}

/* Accuracy highlight card — big and obvious */
.acc-highlight{border-radius:8px;padding:18px 24px;margin-bottom:10px;text-align:center;}
.acc-highlight-label{font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;}
.acc-highlight-value{font-family:'Bebas Neue',sans-serif;font-size:36px;letter-spacing:2px;}
.acc-good{background:rgba(74,222,128,0.15);border:1px solid rgba(74,222,128,0.5);}
.acc-ok{background:rgba(255,214,0,0.15);border:1px solid rgba(255,214,0,0.5);}
.acc-bad{background:rgba(255,59,59,0.15);border:1px solid rgba(255,59,59,0.5);}
.acc-good .acc-highlight-label{color:#4ADE80;} .acc-good .acc-highlight-value{color:#4ADE80;}
.acc-ok .acc-highlight-label{color:#FFD600;}   .acc-ok .acc-highlight-value{color:#FFD600;}
.acc-bad .acc-highlight-label{color:#FF3B3B;}  .acc-bad .acc-highlight-value{color:#FF3B3B;}

.insight-chip{font-family:'JetBrains Mono',monospace;font-size:11px;padding:6px 14px;border-radius:2px;letter-spacing:1px;display:inline-block;margin:3px;}
.chip-pos{background:rgba(74,222,128,0.1);color:#4ADE80;border:1px solid rgba(74,222,128,0.3);}
.chip-neg{background:rgba(255,59,59,0.1);color:#FF6B6B;border:1px solid rgba(255,59,59,0.3);}

.video-meta{margin-top:10px;font-family:'JetBrains Mono',monospace;font-size:10px;color:#555;letter-spacing:1px;line-height:1.8;}
.video-title{font-size:14px;font-weight:600;color:white;margin-bottom:6px;line-height:1.4;}

.stat-card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:24px;text-align:center;}
.stat-number{font-family:'Bebas Neue',sans-serif;font-size:52px;line-height:1;color:white;}
.stat-label{font-size:11px;color:var(--muted);margin-top:6px;letter-spacing:1px;}

.pred-row{padding:14px 20px;border-radius:6px;margin-bottom:8px;}
.pred-row-good{background:rgba(74,222,128,0.07);border-left:3px solid #4ADE80;}
.pred-row-ok{background:rgba(255,214,0,0.07);border-left:3px solid #FFD600;}
.pred-row-bad{background:rgba(255,59,59,0.07);border-left:3px solid #FF3B3B;}

.stButton>button{width:100%;background:linear-gradient(135deg,var(--red),#FF6B35)!important;color:white!important;border:none!important;border-radius:4px!important;padding:14px 32px!important;font-family:'Bebas Neue',sans-serif!important;font-size:20px!important;letter-spacing:3px!important;}
.stButton>button:hover{opacity:0.85!important;}
.stTextInput>div>div>input{background:#1A1A1A!important;border:1px solid var(--border)!important;border-radius:4px!important;color:white!important;font-size:15px!important;padding:12px 16px!important;}
.stTextInput>div>div>input:focus{border-color:var(--yellow)!important;box-shadow:0 0 0 1px var(--yellow)!important;}

.suggestion-card{
    background:#111111;
    border:1px solid #222222;
    border-left:4px solid #FFD600;
    border-radius:8px;
    padding:14px 18px;
    margin-bottom:10px;
    font-size:13px;
    color:#DDDDDD;
    line-height:1.6;
}

.suggestion-good{
    border-left:4px solid #4ADE80;
}

.suggestion-warn{
    border-left:4px solid #FFD600;
}

.suggestion-bad{
    border-left:4px solid #FF3B3B;
}
</style>
""", unsafe_allow_html=True)

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

        # ── TEXT FEATURES ────────────────────
        hidden = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[0]

        mask_exp = attention_mask.unsqueeze(-1).float()

        text_feat = (
            (hidden * mask_exp).sum(1)
            / mask_exp.sum(1)
        )

        text_proj = self.text_proj(text_feat)

        # ── IMAGE FEATURES ───────────────────
        img_feat = self.resnet_features(image).view(
            image.size(0),
            -1
        )

        img_proj = self.img_proj(img_feat)

        # ── FLAT CONCAT FUSION ───────────────
        fused = self.fusion(
            torch.cat([text_proj, img_proj], dim=1)
        )

        return self.reg_head(fused).squeeze(1)

@st.cache_resource
def load_model():

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    tokenizer = DistilBertTokenizer.from_pretrained(
        'distilbert-base-uncased'
    )

    model = FlatConcatV9().to(device)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        state_dict,
        strict=False
    )

    model.eval()

    return model, tokenizer, device

val_tfm = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model=model; self.gradients=None; self.activations=None
        tl = model.resnet_features[-2]
        tl.register_forward_hook(
            lambda m,i,o: setattr(self,'activations',o.detach()))
        tl.register_full_backward_hook(
            lambda m,gi,go: setattr(self,'gradients',go[0].detach()))

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

# ─────────────────────────────────────────────
# INTEGRATED GRADIENTS
# ─────────────────────────────────────────────
def integrated_gradients(model, tokenizer, ids, mask, image, device):

    model.eval()

    embed_layer = model.distilbert.embeddings.word_embeddings

    actual_embeds = embed_layer(ids).detach()

    base_embeds = embed_layer(
        torch.zeros_like(ids)
    ).detach()

    grad_sum = torch.zeros_like(actual_embeds)

    alphas = torch.linspace(
        0,
        1,
        IG_STEPS
    ).to(device)

    for alpha in alphas:

        interp = (
            base_embeds
            + alpha * (actual_embeds - base_embeds)
        ).requires_grad_(True)

        db_out = model.distilbert(
            attention_mask=mask,
            inputs_embeds=interp
        )

        hidden = db_out.last_hidden_state

        mask_exp = mask.unsqueeze(-1).float()

        text_feat = (
            (hidden * mask_exp).sum(1)
            / mask_exp.sum(1)
        )

        text_proj = model.text_proj(text_feat)

        img_feat = model.resnet_features(image).view(
            image.size(0),
            -1
        ).detach()

        img_proj = model.img_proj(img_feat)

        fused = model.fusion(
            torch.cat(
                [text_proj, img_proj],
                dim=1
            )
        )

        pred = model.reg_head(fused).squeeze(1)

        grad = torch.autograd.grad(
            pred.sum(),
            interp
        )[0]

        grad_sum += grad.detach()

    avg_grads = grad_sum / IG_STEPS

    attr = (
        (avg_grads * (actual_embeds - base_embeds))
        .sum(dim=-1)
        .squeeze(0)
        .cpu()
        .numpy()
    )

    tokens = tokenizer.convert_ids_to_tokens(
        ids.squeeze(0).cpu().numpy()
    )

    return attr, tokens

def make_attr_fig(attr, tokens):

    valid = [
        (t.replace('##',''), a)
        for t, a in zip(tokens, attr)
        if t not in ['[PAD]','[CLS]','[SEP]','<pad>']
    ]

    if not valid:
        return None

    labels = [t for t, _ in valid]
    vals   = np.array([a for _, a in valid])

    norm   = Normalize(
        vmin=vals.min(),
        vmax=vals.max()
    )

    colors = [
        plt.cm.RdYlGn(norm(v))
        for v in vals
    ]

    fig_h = max(3.5, len(labels) * 0.38)

    fig, ax = plt.subplots(
        figsize=(8, fig_h)
    )

    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#111111')

    ax.barh(
        range(len(labels)),
        vals,
        color=colors,
        edgecolor='#222',
        linewidth=0.5
    )

    ax.set_yticks(range(len(labels)))

    ax.set_yticklabels(
        labels,
        fontsize=9,
        color='#CCCCCC'
    )

    ax.axvline(
        0,
        color='#444',
        linewidth=1,
        linestyle='--'
    )

    ax.set_xlabel(
        'Attribution Score',
        fontsize=9,
        color='#888'
    )

    ax.tick_params(
        colors='#666',
        labelsize=8
    )

    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

    sm = ScalarMappable(
        cmap=plt.cm.RdYlGn,
        norm=norm
    )

    sm.set_array([])

    cb = plt.colorbar(
        sm,
        ax=ax,
        fraction=0.03,
        pad=0.04
    )

    cb.ax.yaxis.set_tick_params(color='#666')

    cb.outline.set_edgecolor('#333')

    plt.setp(
        cb.ax.yaxis.get_ticklabels(),
        color='#888',
        fontsize=8
    )

    plt.tight_layout()

    return fig

def top_tokens(attr, tokens, n=3):

    valid = []

    for t, a in zip(tokens, attr):

        if t in ['[PAD]', '[CLS]', '[SEP]', '<pad>']:
            continue

        clean = t.replace('##', '')

        if len(clean) < 3:
            continue

        valid.append((clean, a))

    pos_sorted = sorted(valid, key=lambda x: x[1], reverse=True)
    neg_sorted = sorted(valid, key=lambda x: x[1])

    pos = [t for t, a in pos_sorted if a > 0][:n]
    neg = [t for t, a in neg_sorted if a < 0][:n]

    return pos, neg

# ─────────────────────────────────────────────
# AI OPTIMIZATION + ANALYSIS ENGINE
# ─────────────────────────────────────────────
def generate_ai_suggestions(
    title,
    pred_vpd,
    pos_words,
    neg_words,
    cam,
    attr,
    tokens,
    actual_vpd=None
):

    suggestions = []

    title_lower = title.lower()

    # ─────────────────────────────────────────
    # MODEL-DRIVEN TITLE STRENGTH ANALYSIS
    # ─────────────────────────────────────────

    valid_attr = [
        abs(a)
        for t, a in zip(tokens, attr)
        if t not in ['[PAD]', '[CLS]', '[SEP]', '<pad>']
    ]

    if len(valid_attr) > 0:

        top_strength = np.mean(
            sorted(valid_attr, reverse=True)[:5]
        )
        attr_std = np.std(valid_attr)
        strength_ratio = top_strength / (attr_std + 1e-6)
        # weak title signal
        if strength_ratio< 1.2:

            suggestions.append((
                "bad",
                'Title lacks strong high-impact keywords.'
            ))

            suggestions.append((
                "warn",
                'Consider adding stronger emotional, challenge, curiosity, or outcome-focused wording.'
            ))

        # medium strength
        elif strength_ratio < 2.0:

            suggestions.append((
                "warn",
                'Title signal strength appears moderate.'
            ))

            suggestions.append((
                "warn",
                'More emotionally weighted keywords may improve click potential.'
            ))

        # strong title
        else:

            suggestions.append((
                "good",
                'Title contains strong attention-driving language patterns.'
            ))

    # ─────────────────────────────────────────
    # POSITIVE / NEGATIVE TOKENS
    # ─────────────────────────────────────────

    if len(pos_words) > 0:

        suggestions.append((
            "good",
            f'Strong keywords detected: {", ".join(pos_words)}.'
        ))

    if len(neg_words) > 0:

        suggestions.append((
            "bad",
            f'Words reducing prediction score: {", ".join(neg_words)}.'
        ))

    # ─────────────────────────────────────────
    # TITLE LENGTH ANALYSIS
    # ─────────────────────────────────────────

    wc = len(title.split())

    if wc < 4:

        suggestions.append((
            "warn",
            'Title may be too short. Add more context or curiosity.'
        ))

    elif wc > 14:

        suggestions.append((
            "warn",
            'Title may be too long. Shorter titles are often more clickable.'
        ))

    # ─────────────────────────────────────────
    # THUMBNAIL ANALYSIS
    # ─────────────────────────────────────────

    center_focus = np.mean(cam[80:144, 80:144])

    left_focus = np.mean(cam[:, :70])

    right_focus = np.mean(cam[:, 154:])

    overall_focus = np.mean(cam)
    focus_std = max(np.std(cam), 1e-6)

    if center_focus < 0.20:

        suggestions.append((
            "bad",
            'Thumbnail lacks a strong central focal point.'
        ))

        suggestions.append((
            "warn",
            'Consider larger faces, objects, or bold text near the center.'
        ))

    if abs(left_focus - right_focus) > 0.15:

        suggestions.append((
            "warn",
            'Visual attention appears uneven across the thumbnail.'
        ))

    if overall_focus < (focus_std * 0.85):

        suggestions.append((
            "bad",
            'Thumbnail may need stronger contrast or clearer subject separation.'
        ))

    # ─────────────────────────────────────────
    # PRE-PUBLICATION ANALYSIS
    # ─────────────────────────────────────────

    if actual_vpd is None:

        if pred_vpd < 10_000:

            suggestions.append((
                "bad",
                'Current packaging shows weak click-driving potential.'
            ))

        elif pred_vpd < 100_000:

            suggestions.append((
                "warn",
                'Packaging has moderate potential but could be optimized further.'
            ))

        else:

            suggestions.append((
                "good",
                'Packaging signals appear strong overall.'
            ))

    # ─────────────────────────────────────────
    # POST-PUBLICATION ANALYSIS
    # ─────────────────────────────────────────

    else:

        ratio = pred_vpd / max(actual_vpd, 1)

        if ratio < 0.5:

            suggestions.append((
                "good",
                'Video significantly outperformed model expectations.'
            ))

            suggestions.append((
                "warn",
                'Possible external virality factors: trends, audience momentum, or creator influence.'
            ))

        elif ratio > 2.0:

            suggestions.append((
                "bad",
                'Packaging appeared strong but audience response was weaker than expected.'
            ))

            suggestions.append((
                "warn",
                'Possible mismatch between click appeal and viewer retention.'
            ))

        else:

            suggestions.append((
                "good",
                'Prediction aligned closely with actual audience response.'
            ))

    # ─────────────────────────────────────────
    # REMOVE DUPLICATES
    # ─────────────────────────────────────────

    unique = []

    seen = set()

    for level, text in suggestions:

        if text not in seen:

            unique.append((level, text))

            seen.add(text)

    return unique[:8]
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
    """Returns (label, css_class, color, ratio_str)"""
    if actual==0: return "N/A","acc-ok","#FFD600","—"
    r = pred/actual
    ratio_str = f"{r:.2f}×"
    if 0.5<=r<=2.0:   return "✅ ON TARGET","acc-good","#4ADE80", ratio_str
    if 0.25<=r<=4.0:  return "⚠️ CLOSE",   "acc-ok",  "#FFD600", ratio_str
    return "❌ OFF",  "acc-bad","#FF3B3B", ratio_str

# ─────────────────────────────────────────────
# PREDICTION LOG — robust file-based
# ─────────────────────────────────────────────
def load_log():
    """Always reads fresh from disk."""
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
    except Exception:
        pass
    return []

def append_prediction(entry):
    """Reads current log, appends, writes back atomically."""
    log = load_log()
    log.append(entry)
    tmp = LOG_FILE + ".tmp"
    try:
        with open(tmp, 'w') as f:
            json.dump(log, f, indent=2)
        os.replace(tmp, LOG_FILE)   # atomic on all platforms
    except Exception as e:
        st.warning(f"Could not save prediction log: {e}")

# ─────────────────────────────────────────────
# YOUTUBE API — KEY ROTATOR
# ─────────────────────────────────────────────
class KeyRotator:
    def __init__(self, keys):
        self.keys=keys; self.idx=0; self.exhausted=set()

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
                else:
                    return None,f"API error {r.status_code}"
            except Exception as e:
                return None,str(e)
        return None,"All keys failed"

@st.cache_resource
def get_rotator(): return KeyRotator(API_KEYS)

# ─────────────────────────────────────────────
# YOUTUBE FETCH
# ─────────────────────────────────────────────
def fetch_surprise_video(category_id=None):
    rotator    = get_rotator()
    base       = "https://www.googleapis.com/youtube/v3"
    query_list = CATEGORY_QUERIES.get(category_id, CATEGORY_QUERIES[None])
    query      = random.choice(query_list)

    search_params = {
        "part":              "snippet",
        "type":              "video",
        "q":                 query,
        "order":             "date",
        "publishedAfter":    (datetime.datetime.utcnow() -
                              datetime.timedelta(days=14)).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "maxResults":        50,
        "videoDuration":     "medium",
        "regionCode":        "US",
        "relevanceLanguage": "en",
    }
    if category_id:
        search_params["videoCategoryId"] = category_id

    data, err = rotator.call(f"{base}/search", search_params)
    if err or not data or "items" not in data:
        return None, f"Search failed: {err}"

    items = data.get("items",[])
    if not items: return None,"No videos found — try another category"
    random.shuffle(items)

    for item in items:
        vid_id     = item["id"].get("videoId")
        channel_id = item["snippet"].get("channelId")
        if not vid_id or not channel_id: continue

        ch_data,err = rotator.call(f"{base}/channels",
                                   {"part":"statistics","id":channel_id})
        if err or not ch_data or not ch_data.get("items"): continue
        subs = int(ch_data["items"][0]["statistics"].get("subscriberCount",0))
        if not (5_000<=subs<=100_000): continue

        vid_data,err = rotator.call(f"{base}/videos",
                                    {"part":"snippet,statistics","id":vid_id})
        if err or not vid_data or not vid_data.get("items"): continue

        vitem   = vid_data["items"][0]
        stats   = vitem["statistics"]
        snippet = vitem["snippet"]
        views   = int(stats.get("viewCount",0))

        title_text = snippet.get("title","")
        title_lower = title_text.lower()
        if "#shorts" in title_lower or "#short" in title_lower: continue
        if views==0: continue

        # Skip non-English titles — detect by checking for non-ASCII characters
        # that indicate Indian/Asian scripts (Telugu, Hindi, Tamil, etc.)
        non_ascii = sum(1 for c in title_text if ord(c) > 127)
        if non_ascii > len(title_text) * 0.2:  # >20% non-ASCII = non-English
            continue

        published  = snippet.get("publishedAt","")
        pub_dt     = datetime.datetime.strptime(published,'%Y-%m-%dT%H:%M:%SZ')
        days_live  = max(1,(datetime.datetime.utcnow()-pub_dt).days)
        actual_vpd = views/days_live

        thumbs    = snippet.get("thumbnails",{})
        thumb_url = (thumbs.get("maxres") or thumbs.get("high") or
                     thumbs.get("medium",{})).get("url","")
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

# ─────────────────────────────────────────────
# SHARED PREDICT + RENDER
# ─────────────────────────────────────────────
def run_prediction(model, tokenizer, device, title, pil_img):
    img_t = val_tfm(pil_img).unsqueeze(0).to(device)
    enc   = tokenizer(title.strip(), add_special_tokens=True, max_length=64,
                      padding='max_length', truncation=True,
                      return_attention_mask=True, return_tensors='pt')
    ids  = enc['input_ids'].to(device)
    mask = enc['attention_mask'].to(device)
    with torch.no_grad():
        pred_norm = model(ids, mask, img_t).item()
    return ids, mask, img_t, float(np.expm1(pred_norm*TARGET_STD+TARGET_MEAN))

def render_results(model, tokenizer, device, gradcam_obj,
                   ids, mask, img_t, pil_img, pred_vpd,
                   title,
                   actual_vpd=None, video_info=None):

    tier, tier_cls = get_tier(pred_vpd)
    sc             = get_score_cls(pred_vpd)

    # ── Score card ──────────────────────────────────────────
    st.markdown(f"""
    <div class="score-block">
        <div class="score-label">Predicted Views / Day</div>
        <div class="score-number {sc}">{fmt(pred_vpd)}</div>
        <div class="score-unit">views per day at publish</div>
        <div><span class="tier-badge {tier_cls}">{tier}</span></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Accuracy block (Surprise Me) ────────────────────────
    if actual_vpd is not None:
        label, acc_cls, color, ratio = accuracy_info(pred_vpd, actual_vpd)

        # Big highlighted accuracy card
        st.markdown(f"""
        <div class="acc-highlight {acc_cls}">
            <div class="acc-highlight-label">Accuracy vs Actual</div>
            <div class="acc-highlight-value">{label}</div>
        </div>""", unsafe_allow_html=True)

        # Three metric cards side by side
        cp, ca, cr_col = st.columns(3)
        with cp:
            st.markdown(f"""<div class="metric-stack">
                <div class="metric-stack-label">Predicted vpd</div>
                <div class="metric-stack-value">{fmt(pred_vpd)}</div>
            </div>""", unsafe_allow_html=True)
        with ca:
            st.markdown(f"""<div class="metric-stack">
                <div class="metric-stack-label">Actual vpd</div>
                <div class="metric-stack-value" style="color:{color};">
                    {fmt(actual_vpd)}</div>
            </div>""", unsafe_allow_html=True)
        with cr_col:
            st.markdown(f"""<div class="metric-stack">
                <div class="metric-stack-label">Pred / Actual ratio</div>
                <div class="metric-stack-value" style="color:{color};">
                    {ratio}</div>
            </div>""", unsafe_allow_html=True)

    # ── Manual mode estimates ────────────────────────────────
    else:
        c7, c30 = st.columns(2)
        with c7:
            st.markdown(f"""<div class="metric-stack">
                <div class="metric-stack-label">Est. views in 7 days</div>
                <div class="metric-stack-value">{fmt(pred_vpd*7)}</div>
            </div>""", unsafe_allow_html=True)
        with c30:
            st.markdown(f"""<div class="metric-stack">
                <div class="metric-stack-label">Est. views in 30 days</div>
                <div class="metric-stack-value">{fmt(pred_vpd*30)}</div>
            </div>""", unsafe_allow_html=True)

    if video_info:
        st.markdown(f"""
        <a href="{video_info['yt_url']}" target="_blank"
           style="font-family:'JetBrains Mono',monospace;font-size:11px;
                  color:#FFD600;text-decoration:none;letter-spacing:1px;">
            ▶ WATCH ON YOUTUBE →</a>""", unsafe_allow_html=True)

    # ── Explainability ───────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">EXPLAINABILITY</div>',
                unsafe_allow_html=True)
    cam, _ = gradcam_obj.generate(ids, mask, img_t)
    tab1,tab2 = st.tabs(['📸 Thumbnail · Grad-CAM',
                          '📝 Title · Integrated Gradients'])
    with tab1:
        with st.spinner('Generating Grad-CAM...'):
            
            fig_c = make_heatmap_fig(pil_img,cam)
            st.pyplot(fig_c,use_container_width=True); plt.close(fig_c)
        st.caption('Red/warm = regions the model weighted heavily.')
    with tab2:
        with st.spinner('Running Integrated Gradients (~5s)...'):
            attr, tokens = integrated_gradients(
                model,
                tokenizer,
                ids,
                mask,
                img_t,
                device
            )

            pos_w, neg_w = top_tokens(attr, tokens)

            ai_suggestions = generate_ai_suggestions(
                title=title,
                pred_vpd=pred_vpd,
                pos_words=pos_w,
                neg_words=neg_w,
                cam=cam,
                attr=attr,
                tokens=tokens,
                actual_vpd=actual_vpd
            )

            chips = "".join(
                [f'<span class="insight-chip chip-pos">↑ {w}</span>' for w in pos_w] +
                [f'<span class="insight-chip chip-neg">↓ {w}</span>' for w in neg_w]
            )

            st.markdown(chips, unsafe_allow_html=True)

            st.markdown('<br>', unsafe_allow_html=True)

            st.markdown(
                '<div class="section-header">AI OPTIMIZATION SUGGESTIONS</div>',
                unsafe_allow_html=True
            )

            for level, text in ai_suggestions:
                css_class = {
                    "good": "suggestion-good",
                    "warn": "suggestion-warn",
                    "bad": "suggestion-bad"
                }.get(level, "suggestion-warn")

                st.markdown(
                    f'''
                    <div class="suggestion-card {css_class}">
                        {text}
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

            fig_ig = make_attr_fig(attr, tokens)

            if fig_ig:
                st.pyplot(fig_ig, use_container_width=True)
                plt.close(fig_ig)

        st.caption('Green = pushed score up · Red = pushed score down')
   
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

    page = st.radio("nav",["Predictor","Accuracy Tracker"],
                    label_visibility="hidden")

    st.markdown('<br>',unsafe_allow_html=True)
    ok = len(API_KEYS)>0
    st.markdown(f"""<div style="font-family:'JetBrains Mono',monospace;font-size:9px;
        color:{'#4ADE80' if ok else '#FF3B3B'};letter-spacing:1px;">
        API KEYS: {len(API_KEYS)} loaded</div>""", unsafe_allow_html=True)

    # Show log count live in sidebar
    log_count = len([e for e in load_log()
                     if e.get("mode")=="surprise" and e.get("actual_vpd")])
    st.markdown(f"""<div style="font-family:'JetBrains Mono',monospace;font-size:9px;
        color:#444;letter-spacing:1px;margin-top:8px;">
        LOGGED PREDICTIONS: {log_count}</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 1 — PREDICTOR
# ─────────────────────────────────────────────
if page == "Predictor":

    st.markdown("""
    <div style="display:flex;align-items:baseline;gap:16px;margin-bottom:4px;">
        <p class="hero-title">TRENDCAST</p>
        <span class="hero-tag">V9 · Flat Concat</span>
    </div>
    <p class="hero-sub">
        Predict video virality before you publish — title + thumbnail only.<br>
        DistilBERT + ResNet-50 Flat Concat Fusion · R²=0.401
    </p>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    with st.spinner('Loading TrendCast V9 Flat Concat...'):
        try:
            model, tokenizer, device = load_model()
            gradcam_obj = GradCAM(model)
        except FileNotFoundError:
            st.error(f'Model not found: `{MODEL_PATH}` — place it in the same folder as app.py')
            st.stop()

    mode_tab1, mode_tab2 = st.tabs(["✏️  Manual Input","🎲  Surprise Me"])

    # ── MANUAL ──────────────────────────────────────────────
    with mode_tab1:
        st.markdown('<br>',unsafe_allow_html=True)
        col_in,col_out = st.columns([1,1],gap='large')
        with col_in:
            st.markdown('<div class="section-header">INPUT</div>',unsafe_allow_html=True)
            title    = st.text_input("t",label_visibility="hidden",
                                     placeholder="e.g. I survived 100 days in Antarctica...")
            uploaded = st.file_uploader("u",label_visibility="hidden",
                                        type=['jpg','jpeg','png'])
            if uploaded:
                pil_manual = Image.open(uploaded).convert('RGB')
                st.image(pil_manual,use_container_width=True)
            st.markdown('<br>',unsafe_allow_html=True)
            run_manual = st.button("PREDICT VIRALITY",key="manual_btn")

        with col_out:
            st.markdown('<div class="section-header">PREDICTION</div>',unsafe_allow_html=True)
            if run_manual:
                if not title.strip(): st.warning("Please enter a title.")
                elif not uploaded:    st.warning("Please upload a thumbnail.")
                else:
                    with st.spinner("Analysing..."):
                        ids,mask,img_t,pred_vpd = run_prediction(
                            model,tokenizer,device,title,pil_manual)
                    render_results(
                        model,
                        tokenizer,
                        device,
                        gradcam_obj,
                        ids,
                        mask,
                        img_t,
                        pil_manual,
                        pred_vpd,
                        title=title
                    )
                    append_prediction({
                        "timestamp":  datetime.datetime.utcnow().isoformat(),
                        "mode":       "manual",
                        "title":      title.strip(),
                        "pred_vpd":   pred_vpd,
                        "actual_vpd": None,
                        "video_id":   None,
                        "channel":    None,
                        "subs":       None,
                    })
            else:
                st.markdown("""
                <div style="border:1px dashed #222;border-radius:8px;padding:60px 40px;text-align:center;">
                    <div style="font-family:'Bebas Neue',sans-serif;font-size:48px;
                                letter-spacing:4px;color:#222;margin-bottom:12px;">AWAITING INPUT</div>
                    <div style="font-size:13px;color:#444;">Enter a title and upload a thumbnail</div>
                </div>""", unsafe_allow_html=True)

    # ── SURPRISE ME ─────────────────────────────────────────
    with mode_tab2:
        st.markdown('<br>',unsafe_allow_html=True)
        if not API_KEYS:
            st.error("No YouTube API keys found — add YT_API_KEY_1 … YT_API_KEY_6 to your .env file.")
        else:
            col_cat,col_btn = st.columns([2,1],gap='medium')
            with col_cat:
                cat_label = st.selectbox("Category",list(CATEGORIES.keys()),key="cat")
            with col_btn:
                st.markdown('<br>',unsafe_allow_html=True)
                surprise_btn = st.button("SURPRISE ME",key="surprise_btn")

            if surprise_btn:
                cat_id = CATEGORIES[cat_label]
                with st.spinner("Fetching a random small-channel video (5K–100K subs, US, English)..."):
                    video_info,err = fetch_surprise_video(cat_id)

                if err:
                    st.error(f"Could not fetch video: {err}")
                else:
                    pil_s = download_thumbnail(video_info["thumbnail_url"])
                    if not pil_s:
                        st.error("Could not download thumbnail — try again.")
                    else:
                        st.markdown("---")
                        c1,c2 = st.columns([1,1],gap='large')
                        with c1:
                            st.markdown('<div class="section-header">FETCHED VIDEO</div>',
                                        unsafe_allow_html=True)
                            st.image(pil_s,use_container_width=True)
                            st.markdown(f"""
                            <div class="video-title">{video_info['title']}</div>
                            <div class="video-meta">
                                {video_info['channel_title']} &nbsp;·&nbsp;
                                {fmt(video_info['subscriber_count'])} subs &nbsp;·&nbsp;
                                {video_info['days_live']}d old &nbsp;·&nbsp;
                                {fmt(video_info['view_count'])} views
                            </div>""", unsafe_allow_html=True)

                        with c2:
                            st.markdown('<div class="section-header">PREDICTION</div>',
                                        unsafe_allow_html=True)
                            with st.spinner("Predicting..."):
                                ids,mask,img_t,pred_vpd = run_prediction(
                                    model,tokenizer,device,video_info['title'],pil_s)

                            # Log BEFORE render so disk write always completes
                            label,_,_,ratio = accuracy_info(pred_vpd,video_info['actual_vpd'])
                            append_prediction({
                                "timestamp":   datetime.datetime.utcnow().isoformat(),
                                "mode":        "surprise",
                                "title":       video_info['title'],
                                "pred_vpd":    round(pred_vpd,2),
                                "actual_vpd":  round(video_info['actual_vpd'],2),
                                "video_id":    video_info['video_id'],
                                "channel":     video_info['channel_title'],
                                "subs":        video_info['subscriber_count'],
                                "accuracy":    label,
                                "ratio":       ratio,
                                "yt_url":      video_info['yt_url'],
                                "category":    cat_label,
                            })

                            render_results(
                                model,
                                tokenizer,
                                device,
                                gradcam_obj,
                                ids,
                                mask,
                                img_t,
                                pil_s,
                                pred_vpd,
                                title=video_info['title'],
                                actual_vpd=video_info['actual_vpd'],
                                video_info=video_info
                            )

# ─────────────────────────────────────────────
# PAGE 2 — ACCURACY TRACKER
# ─────────────────────────────────────────────
elif page == "Accuracy Tracker":

    st.markdown("""
    <div style="display:flex;align-items:baseline;gap:16px;margin-bottom:4px;">
        <p class="hero-title">ACCURACY</p>
        <span class="hero-tag">Prediction Log</span>
    </div>
    <p class="hero-sub">
        Every Surprise Me prediction logged against actual views/day at fetch time.<br>
        On-target = predicted within 2× of actual vpd. &nbsp;·&nbsp;
        Close = within 4×.
    </p>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    # Always read fresh from disk
    surprise_log = [e for e in load_log()
                    if e.get("mode")=="surprise" and e.get("actual_vpd")]

    col_refresh, _ = st.columns([1,5])
    with col_refresh:
        if st.button("🔄 Refresh", key="refresh_btn"):
            st.rerun()

    if not surprise_log:
        st.markdown("""
        <div style="border:1px dashed #222;border-radius:8px;padding:80px 40px;text-align:center;">
            <div style="font-family:'Bebas Neue',sans-serif;font-size:52px;
                        letter-spacing:4px;color:#222;margin-bottom:12px;">NO DATA YET</div>
            <div style="font-size:13px;color:#444;">
                Use the Surprise Me feature to generate verified predictions.</div>
        </div>""", unsafe_allow_html=True)
    else:
        total     = len(surprise_log)
        on_target = sum(1 for e in surprise_log
                        if 0.5<=e['pred_vpd']/max(e['actual_vpd'],1)<=2.0)
        close     = sum(1 for e in surprise_log
                        if 0.25<=e['pred_vpd']/max(e['actual_vpd'],1)<=4.0
                        and not 0.5<=e['pred_vpd']/max(e['actual_vpd'],1)<=2.0)
        errors    = [abs(e['pred_vpd']-e['actual_vpd'])/max(e['actual_vpd'],1)*100
                     for e in surprise_log]
        avg_err   = round(sum(errors)/len(errors))
        pct_t     = round(on_target/total*100)
        pct_c     = round((on_target+close)/total*100)

        # ── Stat cards ──────────────────────────────────────
        s1,s2,s3,s4 = st.columns(4)
        for col,num,label in [
            (s1, total,        "TOTAL PREDICTIONS"),
            (s2, f"{pct_t}%",  "ON TARGET  (within 2×)"),
            (s3, f"{pct_c}%",  "CLOSE  (within 4×)"),
            (s4, f"{avg_err}%","AVG % ERROR"),
        ]:
            with col:
                st.markdown(f"""<div class="stat-card">
                    <div class="stat-number">{num}</div>
                    <div class="stat-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<br>',unsafe_allow_html=True)

        # ── Bar chart — grouped by category ─────────────────
        # ── CATEGORY ACCURACY SUMMARY CHART ─────────────────
        st.markdown(
                '<div class="section-header">CATEGORY-WISE PREDICTION ACCURACY</div>',
                unsafe_allow_html=True
            )



        cats_present = sorted(set(e.get("category", "Any") for e in surprise_log))

        on_target_counts = []
        close_counts = []
        off_counts = []

        for cat in cats_present:

                cat_entries = [
                    e for e in surprise_log
                    if e.get("category", "Any") == cat
                ]

                on_t = 0
                close = 0
                off = 0

                for e in cat_entries:

                    r = e['pred_vpd'] / max(e['actual_vpd'], 1)

                    if 0.5 <= r <= 2.0:
                        on_t += 1

                    elif 0.25 <= r <= 4.0:
                        close += 1

                    else:
                        off += 1

                on_target_counts.append(on_t)
                close_counts.append(close)
                off_counts.append(off)

            # Plotting
        x = np.arange(len(cats_present))
        w = 0.25

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#111111')
        ax.set_facecolor('#111111')

        bars1 = ax.bar(
                x - w,
                on_target_counts,
                width=w,
                color='#4ADE80',
                label='On Target'
            )

        bars2 = ax.bar(
                x,
                close_counts,
                width=w,
                color='#FFD600',
                label='Close'
            )

        bars3 = ax.bar(
                x + w,
                off_counts,
                width=w,
                color='#FF3B3B',
                label='Off'
            )

            # Labels on bars
        for bars in [bars1, bars2, bars3]:
                for b in bars:
                    h = b.get_height()
                    ax.text(
                        b.get_x() + b.get_width()/2,
                        h + 0.05,
                        str(int(h)),
                        ha='center',
                        fontsize=8,
                        color='white'
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(
                cats_present,
                fontsize=10,
                color='#CCCCCC'
            )

        ax.set_ylabel(
                "Number of Predictions",
                fontsize=10,
                color='#AAAAAA'
            )

        ax.set_title(
                "Prediction Accuracy by Category",
                fontsize=14,
                color='#FFD600',
                pad=15
            )

        ax.tick_params(colors='#666')

        for sp in ax.spines.values():
                sp.set_edgecolor('#2a2a2a')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        legend = ax.legend(
                facecolor='#1a1a1a',
                edgecolor='#333',
                fontsize=9
            )

        for text in legend.get_texts():
                text.set_color('#CCCCCC')

        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown('<br>',unsafe_allow_html=True)

        # ── Prediction table ─────────────────────────────────
        st.markdown('<div class="section-header">ALL PREDICTIONS</div>',
                    unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:12px;color:#444;margin-bottom:16px;'>"
                    f"Showing {total} verified predictions</div>",
                    unsafe_allow_html=True)

        for e in reversed(surprise_log):
            r = e['pred_vpd']/max(e['actual_vpd'],1)
            if 0.5<=r<=2.0:    row_cls="pred-row-good"; badge="✅ On Target"; col="#4ADE80"
            elif 0.25<=r<=4.0: row_cls="pred-row-ok";   badge="⚠️ Close";    col="#FFD600"
            else:               row_cls="pred-row-bad";  badge="❌ Off";       col="#FF3B3B"
            ts  = e.get('timestamp','')[:10]
            url = e.get('yt_url','#')
            err_pct = round(abs(e['pred_vpd']-e['actual_vpd'])/max(e['actual_vpd'],1)*100)

            st.markdown(f"""
            <div class="pred-row {row_cls}">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div style="flex:1;margin-right:20px;">
                        <div style="font-size:13px;font-weight:500;
                                    color:white;margin-bottom:4px;">{e['title']}</div>
                        <div style="font-family:'JetBrains Mono',monospace;
                                    font-size:9px;color:#555;letter-spacing:1px;">
                            {e.get('channel','—')} ·
                            {fmt(e.get('subs') or 0)} subs · {ts}
                        </div>
                    </div>
                    <div style="text-align:right;min-width:280px;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:12px;margin-bottom:4px;">
                            Pred:&nbsp;<span style="color:white;">{fmt(e['pred_vpd'])}</span>&nbsp;vpd
                            &nbsp;·&nbsp;
                            Actual:&nbsp;<span style="color:{col};">{fmt(e['actual_vpd'])}</span>&nbsp;vpd
                        </div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:11px;">
                            <span style="color:{col};font-weight:600;">{badge}</span>
                            &nbsp;·&nbsp;
                            <span style="color:#555;">{err_pct}% error</span>
                            &nbsp;·&nbsp;
                            <a href="{url}" target="_blank"
                               style="color:#FFD600;text-decoration:none;">▶ Watch</a>
                        </div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Export ───────────────────────────────────────────
        st.markdown('<br>',unsafe_allow_html=True)
        df_export = pd.DataFrame(surprise_log)
        st.download_button("⬇ Export CSV",
                           data=df_export.to_csv(index=False),
                           file_name="trendcast_predictions.csv",
                           mime="text/csv")

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

