import streamlit as st
import datetime
from PIL import Image

def render_predictor(model, tokenizer, device, gradcam_obj, fetch_surprise_video, run_prediction, render_prediction_metrics, render_explainability, append_prediction, download_thumbnail, fmt, CATEGORIES, accuracy_info):
    
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

    mode_tab1, mode_tab2 = st.tabs(["✏️  Manual Input", "🎲  Surprise Me"])

    # ── MANUAL ──────────────────────────────────────────────
    with mode_tab1:
        st.markdown('<br>', unsafe_allow_html=True)
        col_in, col_out = st.columns([1, 1], gap='large')
        
        validation_passed = False
        
        with col_in:
            st.markdown('<div class="section-header">INPUT</div>', unsafe_allow_html=True)
            title = st.text_input("t", label_visibility="hidden", placeholder="e.g. I survived 100 days in Antarctica...")
            uploaded = st.file_uploader("u", label_visibility="hidden", type=['jpg', 'jpeg', 'png'])
            
            if uploaded:
                pil_manual = Image.open(uploaded).convert('RGB')
                st.image(pil_manual, use_container_width=True)
                
            st.markdown('<br>', unsafe_allow_html=True)
            run_manual = st.button("PREDICT VIRALITY", key="manual_btn")

        if run_manual:
            if not title.strip(): 
                with col_out: st.warning("Please enter a title.")
            elif not uploaded:    
                with col_out: st.warning("Please upload a thumbnail.")
            else:
                validation_passed = True
        elif not run_manual:
            with col_out:
                st.markdown('<div class="section-header">PREDICTION</div>', unsafe_allow_html=True)
                st.markdown("""
                <div style="border:1px dashed #222;border-radius:8px;padding:60px 40px;text-align:center;height:100%;">
                    <div style="font-family:'Bebas Neue',sans-serif;font-size:48px;
                                letter-spacing:4px;color:#222;margin-bottom:12px;">AWAITING INPUT</div>
                    <div style="font-size:13px;color:#444;">Enter a title and upload a thumbnail</div>
                </div>""", unsafe_allow_html=True)

        if validation_passed:
            with st.spinner("Analysing..."):
                ids, mask, img_t, pred_vpd = run_prediction(model, tokenizer, device, title, pil_manual)
            
            # 1. Render scores in the right column
            with col_out:
                st.markdown('<div class="section-header">PREDICTION</div>', unsafe_allow_html=True)
                render_prediction_metrics(pred_vpd, actual_vpd=None, video_info=None)
                
            # 2. Render explainability FULL WIDTH at the bottom
            render_explainability(model, tokenizer, device, gradcam_obj, ids, mask, img_t, pil_manual, pred_vpd, title)
            
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

    # ── SURPRISE ME ─────────────────────────────────────────
    with mode_tab2:
        st.markdown('<br>', unsafe_allow_html=True)
        col_cat, col_btn = st.columns([2, 1], gap='medium')
        
        with col_cat:
            cat_label = st.selectbox("Category", list(CATEGORIES.keys()), key="cat")
        with col_btn:
            st.markdown('<br>', unsafe_allow_html=True)
            surprise_btn = st.button("SURPRISE ME", key="surprise_btn")

        if surprise_btn:
            cat_id = CATEGORIES[cat_label]
            with st.spinner("Fetching a random small-channel video (5K–100K subs, US, English)..."):
                video_info, err = fetch_surprise_video(cat_id)

            if err:
                st.error(f"Could not fetch video: {err}")
            else:
                pil_s = download_thumbnail(video_info["thumbnail_url"])
                if not pil_s:
                    st.error("Could not download thumbnail — try again.")
                else:
                    st.markdown("---")
                    c1, c2 = st.columns([1, 1], gap='large')
                    
                    with st.spinner("Predicting..."):
                        ids, mask, img_t, pred_vpd = run_prediction(model, tokenizer, device, video_info['title'], pil_s)

                    with c1:
                        st.markdown('<div class="section-header">FETCHED VIDEO</div>', unsafe_allow_html=True)
                        st.image(pil_s, use_container_width=True)
                        st.markdown(f"""
                        <div class="video-title">{video_info['title']}</div>
                        <div class="video-meta">
                            {video_info['channel_title']} &nbsp;·&nbsp;
                            {fmt(video_info['subscriber_count'])} subs &nbsp;·&nbsp;
                            {video_info['days_live']}d old &nbsp;·&nbsp;
                            {fmt(video_info['view_count'])} views
                        </div>""", unsafe_allow_html=True)

                    with c2:
                        st.markdown('<div class="section-header">PREDICTION</div>', unsafe_allow_html=True)
                        render_prediction_metrics(pred_vpd, actual_vpd=video_info['actual_vpd'], video_info=video_info)
                        
                    # Break out of columns for FULL WIDTH explainability
                    render_explainability(model, tokenizer, device, gradcam_obj, ids, mask, img_t, pil_s, pred_vpd, title=video_info['title'], actual_vpd=video_info['actual_vpd'])

                    label, _, _, ratio = accuracy_info(pred_vpd, video_info['actual_vpd'])
                    
                    append_prediction({
                        "timestamp":   datetime.datetime.utcnow().isoformat(),
                        "mode":        "surprise",
                        "title":       video_info['title'],
                        "pred_vpd":    round(pred_vpd, 2),
                        "actual_vpd":  round(video_info['actual_vpd'], 2),
                        "video_id":    video_info['video_id'],
                        "channel":     video_info['channel_title'],
                        "subs":        video_info['subscriber_count'],
                        "accuracy":    label,
                        "ratio":       ratio,
                        "yt_url":      video_info['yt_url'],
                        "category":    cat_label,
                    })