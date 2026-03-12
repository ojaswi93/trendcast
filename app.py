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
import io
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TrendCast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

TARGET_MEAN = 8.74300812380281
TARGET_STD  = 2.0089800699907783
IG_STEPS    = 30   # slightly reduced for faster demo (still accurate)
MODEL_PATH  = "trendcast_v6_best_model.pth"

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --red:    #FF3B3B;
    --yellow: #FFD600;
    --dark:   #0A0A0A;
    --card:   #111111;
    --border: #222222;
    --text:   #E8E8E8;
    --muted:  #666666;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--dark);
    color: var(--text);
}

.stApp { background-color: var(--dark); }

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1400px; }

/* Hero header */
.hero {
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin-bottom: 4px;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 72px;
    letter-spacing: 4px;
    color: white;
    line-height: 1;
    margin: 0;
}
.hero-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--yellow);
    background: rgba(255,214,0,0.1);
    border: 1px solid rgba(255,214,0,0.3);
    padding: 4px 10px;
    border-radius: 2px;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.hero-sub {
    font-size: 14px;
    color: var(--muted);
    margin-bottom: 40px;
    letter-spacing: 0.3px;
}

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, var(--red), var(--yellow), transparent);
    margin: 0 0 40px 0;
}

/* Input card */
.input-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 28px;
    margin-bottom: 24px;
}
.input-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--yellow);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

/* Score display */
.score-block {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 32px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.score-block::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--red), var(--yellow));
}
.score-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.score-number {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 64px;
    line-height: 1;
    margin-bottom: 8px;
}
.score-unit {
    font-size: 13px;
    color: var(--muted);
    font-weight: 300;
}
.score-high { color: #4ADE80; }
.score-mid  { color: var(--yellow); }
.score-low  { color: var(--red); }

/* Tier badge */
.tier-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    padding: 6px 16px;
    border-radius: 2px;
    margin-top: 16px;
    font-weight: 600;
}
.tier-viral    { background: rgba(74,222,128,0.15); color: #4ADE80; border: 1px solid rgba(74,222,128,0.4); }
.tier-strong   { background: rgba(255,214,0,0.15);  color: var(--yellow); border: 1px solid rgba(255,214,0,0.4); }
.tier-moderate { background: rgba(251,146,60,0.15); color: #FB923C; border: 1px solid rgba(251,146,60,0.4); }
.tier-low      { background: rgba(255,59,59,0.15);  color: var(--red); border: 1px solid rgba(255,59,59,0.4); }

/* Section headers */
.section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--yellow);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* Insight cards */
.insight-row {
    display: flex;
    gap: 12px;
    margin-bottom: 12px;
    flex-wrap: wrap;
}
.insight-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    padding: 6px 14px;
    border-radius: 2px;
    letter-spacing: 1px;
}
.chip-pos { background: rgba(74,222,128,0.1); color: #4ADE80; border: 1px solid rgba(74,222,128,0.3); }
.chip-neg { background: rgba(255,59,59,0.1);  color: #FF6B6B; border: 1px solid rgba(255,59,59,0.3); }
.chip-neu { background: rgba(255,255,255,0.05); color: #888; border: 1px solid #333; }

/* Stacked metric */
.metric-stack {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.metric-stack-label {
    font-size: 11px;
    color: var(--muted);
    margin-bottom: 4px;
}
.metric-stack-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px;
    font-weight: 600;
    color: white;
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, var(--red), #FF6B35) !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 14px 32px !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 20px !important;
    letter-spacing: 3px !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* File uploader */
.stFileUploader {
    background: var(--card) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
}

/* Text input */
.stTextInput > div > div > input {
    background: #1A1A1A !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: white !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    padding: 12px 16px !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--yellow) !important;
    box-shadow: 0 0 0 1px var(--yellow) !important;
}

/* Plot backgrounds */
.stImage { border-radius: 6px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL DEFINITION (must match training code)
# ─────────────────────────────────────────────
class TrendCastV6(nn.Module):
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

        self.fusion = nn.Sequential(
            nn.Linear(2816, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),  nn.ReLU(),
        )
        self.reg_head = nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask, image):
        text_feat = self.distilbert(
            input_ids=input_ids, attention_mask=attention_mask
        )[0][:, 0, :]
        img_feat  = self.resnet_features(image).view(image.size(0), -1)
        fused     = self.fusion(torch.cat([text_feat, img_feat], dim=1))
        return self.reg_head(fused).squeeze(1)


# ─────────────────────────────────────────────
# LOAD MODEL (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model     = TrendCastV6().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, tokenizer, device

val_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer     = model.resnet_features[-2]

        def fwd(module, inp, out): self.activations = out.detach()
        def bwd(module, gi, go):   self.gradients   = go[0].detach()

        target_layer.register_forward_hook(fwd)
        target_layer.register_full_backward_hook(bwd)

    def generate(self, input_ids, attention_mask, image):
        self.model.zero_grad()
        image_g = image.requires_grad_(True)
        pred    = self.model(input_ids, attention_mask, image_g)
        pred.sum().backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam     = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        cam    -= cam.min()
        if cam.max() > 0: cam /= cam.max()

        pred_vpd = float(np.expm1(pred.item() * TARGET_STD + TARGET_MEAN))
        return cam, pred_vpd


def make_heatmap_figure(pil_img, cam):
    """Returns a matplotlib figure with original + heatmap side by side."""
    w, h        = pil_img.size
    cam_resized = np.array(
        Image.fromarray(np.uint8(cam * 255)).resize((w, h), Image.BILINEAR)
    ) / 255.0
    colormap    = cm.get_cmap('jet')
    heatmap_rgb = colormap(cam_resized)[:, :, :3]
    heatmap_pil = Image.fromarray(np.uint8(heatmap_rgb * 255))
    blended     = Image.blend(pil_img.convert('RGB'), heatmap_pil, 0.5)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor('#111111')
    for ax in axes:
        ax.set_facecolor('#111111')
        ax.axis('off')

    axes[0].imshow(pil_img)
    axes[0].set_title('Original Thumbnail', color='#888', fontsize=10, pad=8)

    axes[1].imshow(blended)
    axes[1].set_title('Grad-CAM  (red = model focused here)', color='#FFD600',
                      fontsize=10, pad=8)

    plt.tight_layout(pad=1.5)
    return fig


# ─────────────────────────────────────────────
# INTEGRATED GRADIENTS
# ─────────────────────────────────────────────
def integrated_gradients_text(model, tokenizer, input_ids, attention_mask,
                               image, device, n_steps=IG_STEPS):
    model.eval()
    embed_layer   = model.distilbert.embeddings.word_embeddings
    actual_embeds = embed_layer(input_ids).detach()
    base_embeds   = embed_layer(torch.zeros_like(input_ids)).detach()
    alphas        = torch.linspace(0, 1, n_steps).to(device)
    grad_sum      = torch.zeros_like(actual_embeds)

    for alpha in alphas:
        interp  = (base_embeds + alpha * (actual_embeds - base_embeds)).requires_grad_(True)
        db_out  = model.distilbert(attention_mask=attention_mask, inputs_embeds=interp)
        cls_out = db_out.last_hidden_state[:, 0, :]
        img_f   = model.resnet_features(image).view(image.size(0), -1).detach()
        pred    = model.reg_head(model.fusion(torch.cat([cls_out, img_f], dim=1))).squeeze(1)
        grad    = torch.autograd.grad(pred.sum(), interp)[0]
        grad_sum += grad.detach()

    avg_grads    = grad_sum / n_steps
    attributions = ((avg_grads * (actual_embeds - base_embeds))
                    .sum(dim=-1).squeeze(0).cpu().numpy())
    tokens       = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().numpy())
    return attributions, tokens


def make_attribution_figure(attributions, tokens):
    """Returns a dark-themed matplotlib figure of token attributions."""
    valid = [(t.replace('##', ''), a) for t, a in zip(tokens, attributions)
             if t not in ['[PAD]', '[CLS]', '[SEP]', '<pad>']]
    if not valid:
        return None

    labels = [t for t, _ in valid]
    vals   = np.array([a for _, a in valid])
    norm   = Normalize(vmin=vals.min(), vmax=vals.max())
    colors = [plt.cm.RdYlGn(norm(v)) for v in vals]

    fig_h = max(3.5, len(labels) * 0.38)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#111111')

    bars = ax.barh(range(len(labels)), vals, color=colors,
                   edgecolor='#222', linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9, color='#CCCCCC')
    ax.axvline(0, color='#444', linewidth=1, linestyle='--')
    ax.set_xlabel('Attribution Score', fontsize=9, color='#888')
    ax.tick_params(colors='#666', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

    sm = ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cb.ax.yaxis.set_tick_params(color='#666')
    cb.outline.set_edgecolor('#333')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#888', fontsize=8)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# SCORING HELPERS
# ─────────────────────────────────────────────
def get_tier(vpd):
    if vpd >= 500_000:  return "VIRAL",    "tier-viral"
    if vpd >= 100_000:  return "STRONG",   "tier-strong"
    if vpd >= 10_000:   return "MODERATE", "tier-moderate"
    return "LOW",  "tier-low"

def get_score_class(vpd):
    if vpd >= 500_000: return "score-high"
    if vpd >= 100_000: return "score-mid"
    return "score-low"

def fmt_vpd(vpd):
    if vpd >= 1_000_000: return f"{vpd/1_000_000:.1f}M"
    if vpd >= 1_000:     return f"{vpd/1_000:.1f}K"
    return str(int(vpd))

def top_tokens(attributions, tokens, n=3):
    valid = [(t.replace('##',''), a) for t, a in zip(tokens, attributions)
             if t not in ['[PAD]','[CLS]','[SEP]','<pad>']]
    valid.sort(key=lambda x: x[1], reverse=True)
    pos = [t for t, a in valid if a > 0][:n]
    neg = [t for t, a in valid if a < 0][:n]
    return pos, neg


# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">TRENDCAST</p>
    <span class="hero-tag">V6 · Pre-Publication</span>
</div>
<p class="hero-sub">
    Predict video virality before you publish — from title + thumbnail alone.
    Powered by DistilBERT + ResNet-50 multimodal fusion.
</p>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────
with st.spinner('Loading TrendCast V6...'):
    try:
        model, tokenizer, device = load_model()
        gradcam = GradCAM(model)
        st.success(f'Model ready · {device}', icon='✅')
    except FileNotFoundError:
        st.error(f'Model file not found: `{MODEL_PATH}`  '
                 f'Place `trendcast_v6_best_model.pth` in the same folder as app.py')
        st.stop()

st.markdown('<br>', unsafe_allow_html=True)

# ── Input section ─────────────────────────────
col_in, col_out = st.columns([1, 1], gap='large')

with col_in:
    st.markdown('<div class="section-header">INPUT</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-label">Video Title</div>', unsafe_allow_html=True)
    title = st.text_input(
        label='title_hidden',
        label_visibility='hidden',
        placeholder='e.g. I survived 100 days in Antarctica...',
        key='title_input'
    )

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="input-label">Thumbnail Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        label='thumb_hidden',
        label_visibility='hidden',
        type=['jpg', 'jpeg', 'png'],
        key='thumb_input'
    )

    if uploaded:
        thumb_img = Image.open(uploaded).convert('RGB')
        st.image(thumb_img, use_container_width=True, caption='Uploaded thumbnail')

    st.markdown('<br>', unsafe_allow_html=True)
    run_btn = st.button('PREDICT VIRALITY', key='run')

# ── Results section ───────────────────────────
with col_out:
    st.markdown('<div class="section-header">PREDICTION</div>', unsafe_allow_html=True)

    if run_btn:
        if not title.strip():
            st.warning('Please enter a video title.')
        elif uploaded is None:
            st.warning('Please upload a thumbnail image.')
        else:
            with st.spinner('Analysing...'):
                # Preprocess
                img_tensor = val_tfm(thumb_img).unsqueeze(0).to(device)
                enc = tokenizer(
                    title.strip(), add_special_tokens=True, max_length=64,
                    padding='max_length', truncation=True,
                    return_attention_mask=True, return_tensors='pt'
                )
                ids  = enc['input_ids'].to(device)
                mask = enc['attention_mask'].to(device)

                # Predict
                with torch.no_grad():
                    pred_norm = model(ids, mask, img_tensor).item()
                pred_vpd  = float(np.expm1(pred_norm * TARGET_STD + TARGET_MEAN))
                tier, tier_cls  = get_tier(pred_vpd)
                score_cls       = get_score_class(pred_vpd)

            # ── Score card ────────────────────────────
            st.markdown(f"""
            <div class="score-block">
                <div class="score-label">Predicted Views / Day</div>
                <div class="score-number {score_cls}">{fmt_vpd(pred_vpd)}</div>
                <div class="score-unit">views per day at publish</div>
                <div><span class="tier-badge {tier_cls}">{tier}</span></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<br>', unsafe_allow_html=True)

            # ── Metrics row ───────────────────────────
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"""
                <div class="metric-stack">
                    <div class="metric-stack-label">Est. views in 7 days</div>
                    <div class="metric-stack-value">{fmt_vpd(pred_vpd * 7)}</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-stack">
                    <div class="metric-stack-label">Est. views in 30 days</div>
                    <div class="metric-stack-value">{fmt_vpd(pred_vpd * 30)}</div>
                </div>""", unsafe_allow_html=True)

            # ── Explainability ────────────────────────
            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">EXPLAINABILITY</div>',
                        unsafe_allow_html=True)

            tab1, tab2 = st.tabs(['📸  Thumbnail (Grad-CAM)', '📝  Title (Integrated Gradients)'])

            with tab1:
                with st.spinner('Generating Grad-CAM...'):
                    cam, _ = gradcam.generate(ids, mask, img_tensor)
                    fig_cam = make_heatmap_figure(thumb_img, cam)
                    st.pyplot(fig_cam, use_container_width=True)
                    plt.close(fig_cam)
                st.caption('Red/warm regions = areas the model weighted heavily for the virality score.')

            with tab2:
                with st.spinner('Running Integrated Gradients (~5s)...'):
                    attributions, tokens = integrated_gradients_text(
                        model, tokenizer, ids, mask, img_tensor, device
                    )
                    fig_ig = make_attribution_figure(attributions, tokens)
                    pos_words, neg_words = top_tokens(attributions, tokens)

                if fig_ig:
                    # Quick insight chips
                    chips_html = '<div class="insight-row">'
                    for w in pos_words:
                        chips_html += f'<span class="insight-chip chip-pos">↑ {w}</span>'
                    for w in neg_words:
                        chips_html += f'<span class="insight-chip chip-neg">↓ {w}</span>'
                    chips_html += '</div>'
                    st.markdown(chips_html, unsafe_allow_html=True)

                    st.pyplot(fig_ig, use_container_width=True)
                    plt.close(fig_ig)
                st.caption('Green = word pushed score up · Red = word pushed score down')

    else:
        st.markdown("""
        <div style="
            border: 1px dashed #222;
            border-radius: 8px;
            padding: 60px 40px;
            text-align: center;
            color: #444;
        ">
            <div style="font-family:'Bebas Neue',sans-serif;font-size:48px;letter-spacing:4px;
                        color:#222;margin-bottom:12px;">AWAITING INPUT</div>
            <div style="font-size:13px;">
                Enter a title and upload a thumbnail,<br>then click PREDICT VIRALITY
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────
st.markdown('<br><br>', unsafe_allow_html=True)
st.markdown("""
<div style="border-top:1px solid #1a1a1a;padding-top:20px;display:flex;
            justify-content:space-between;align-items:center;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#333;
                 letter-spacing:2px;">TRENDCAST V6 · MIT-WPU CAPSTONE 2026</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#333;
                 letter-spacing:1px;">DistilBERT + ResNet-50 · R²=0.42</span>
</div>
""", unsafe_allow_html=True)