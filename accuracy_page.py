import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def render_accuracy_tracker(load_log, fmt):
    
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

    # Always read fresh from disk - filtered out "Any" and "Food" historical logs
    surprise_log = [e for e in load_log()
                    if e.get("mode")=="surprise" and e.get("actual_vpd") and e.get("category") not in ["Any", "Food"]]

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
        
        pct_t     = round(on_target/total*100)
        pct_c     = round((on_target+close)/total*100)

        # ── Stat cards (Avg Error Removed) ──────────────────
        s1,s2,s3 = st.columns(3)
        for col,num,label in [
            (s1, total,        "TOTAL PREDICTIONS"),
            (s2, f"{pct_t}%",  "ON TARGET  (within 2×)"),
            (s3, f"{pct_c}%",  "CLOSE  (within 4×)"),
        ]:
            with col:
                st.markdown(f"""<div class="stat-card">
                    <div class="stat-number">{num}</div>
                    <div class="stat-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<br>',unsafe_allow_html=True)

        # ── Bar chart — grouped by category ─────────────────
        st.markdown(
            '<div class="section-header">CATEGORY-WISE PREDICTION ACCURACY</div>',
            unsafe_allow_html=True
        )

        cats_present = sorted(set(e.get("category", "Unknown") for e in surprise_log))

        on_target_counts = []
        close_counts = []
        off_counts = []

        for cat in cats_present:
            cat_entries = [
                e for e in surprise_log
                if e.get("category", "Unknown") == cat
            ]

            on_t = 0
            close_val = 0
            off = 0

            for e in cat_entries:
                r = e['pred_vpd'] / max(e['actual_vpd'], 1)

                if 0.5 <= r <= 2.0:
                    on_t += 1
                elif 0.25 <= r <= 4.0:
                    close_val += 1
                else:
                    off += 1

            on_target_counts.append(on_t)
            close_counts.append(close_val)
            off_counts.append(off)

        # Plotting
        x = np.arange(len(cats_present))
        w = 0.25

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#111111')
        ax.set_facecolor('#111111')

        bars1 = ax.bar(x - w, on_target_counts, width=w, color='#4ADE80', label='On Target')
        bars2 = ax.bar(x, close_counts, width=w, color='#FFD600', label='Close')
        bars3 = ax.bar(x + w, off_counts, width=w, color='#FF3B3B', label='Off')

        # Labels on bars
        for bars in [bars1, bars2, bars3]:
            for b in bars:
                h = b.get_height()
                if h > 0:
                    ax.text(
                        b.get_x() + b.get_width()/2,
                        h + 0.05,
                        str(int(h)),
                        ha='center',
                        fontsize=8,
                        color='white'
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(cats_present, fontsize=10, color='#CCCCCC')
        ax.set_ylabel("Number of Predictions", fontsize=10, color='#AAAAAA')
        ax.set_title("Prediction Accuracy by Category", fontsize=14, color='#FFD600', pad=15)
        ax.tick_params(colors='#666')

        for sp in ax.spines.values():
            sp.set_edgecolor('#2a2a2a')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        legend = ax.legend(facecolor='#1a1a1a', edgecolor='#333', fontsize=9)
        for text in legend.get_texts():
            text.set_color('#CCCCCC')

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown('<br>',unsafe_allow_html=True)

        # ── Prediction table ─────────────────────────────────
        st.markdown('<div class="section-header">ALL PREDICTIONS</div>', unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:12px;color:#444;margin-bottom:16px;'>"
                    f"Showing {total} verified predictions</div>", unsafe_allow_html=True)

        for e in reversed(surprise_log):
            r = e['pred_vpd']/max(e['actual_vpd'],1)
            if 0.5<=r<=2.0:    row_cls="pred-row-good"; badge="✅ On Target"; col="#4ADE80"
            elif 0.25<=r<=4.0: row_cls="pred-row-ok";   badge="⚠️ Close";    col="#FFD600"
            else:              row_cls="pred-row-bad";  badge="❌ Off";       col="#FF3B3B"
            
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