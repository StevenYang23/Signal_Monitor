import sys

def modify_app():
    with open("app.py", "r", encoding="utf-8") as f:
        content = f.read()

    # Find the start of structure_metrics
    sm_start = content.find("structure_metrics = build_structure_metrics_insights(")
    
    # Find the end of USE_DEEPSEEK block
    ds_end = content.find("hmm_regime_str = \"unknown\"")
    
    if sm_start == -1 or ds_end == -1:
        print("Could not find blocks")
        return
        
    sm_block = content[sm_start:ds_end]
    
    # Find the end of the HMM block
    hmm_end = content.find("return {", ds_end)
    
    hmm_block = content[ds_end:hmm_end]
    
    new_sm_block = """
        # Calculate SSR
        realized_ssr = np.nan
        implied_ssr_val = np.nan
        try:
            es, _ = fetch_es_futures(lookback_days=90)
            fut = es["close"]
            if hmm_history_dates and hmm_history_iv:
                iv_series = pd.Series(hmm_history_iv, index=pd.to_datetime(hmm_history_dates))
                iv_series.index = iv_series.index.tz_localize(None).normalize()
                slope = atmf_skew_slope(aiv, psk, dte=30)
                ssr_df = rolling_ssr(fut, iv_series, slope, window=10)
                if not ssr_df.empty:
                    realized_ssr = float(ssr_df.dropna(subset=["ssr"]).iloc[-1].ssr)
            
            imp = implied_ssr(x_ks, y_dte, sv_data, min_dte=14.0)
            if not imp.empty and not imp["ssr"].dropna().empty:
                implied_ssr_val = float(np.interp(30.0, imp["dte"], imp["ssr"]))
        except Exception as ssr_exc:
            print(f"[pipeline] {short_name} SSR: {ssr_exc}", flush=True)

        structure_metrics = build_structure_metrics_insights(
            today=today_f,
            vix_ctx=result.get("vix_context"),
            anchor_ctx=anchor_ctx,
            changes=result.get("changes"),
        )
        if USE_DEEPSEEK:
            structure_metrics = deepseek_enhance_structure_insights(
                structure_metrics,
                context={
                    "index": short_name,
                    "date": str(today_d),
                    "spot": spot,
                    "term_slope": tsl,
                    "vrp": vrp_val,
                    "hmm_regime": hmm_regime_str,
                    "hmm_prob_today": hmm_prob_today,
                    "realized_ssr": realized_ssr,
                    "implied_ssr": implied_ssr_val,
                    "gex_net_label": gex_payload.get("buckets", {}).get(gex_payload.get("default_bucket", "0"), {}).get("net_label", "n/a"),
                    "gex_regime": gex_payload.get("buckets", {}).get(gex_payload.get("default_bucket", "0"), {}).get("regime", "unknown"),
                    "anomalies_count": len(anomalies)
                },
            )
        
"""

    new_content = content[:sm_start] + hmm_block + new_sm_block + content[hmm_end:]
    
    with open("app.py", "w", encoding="utf-8") as f:
        f.write(new_content)

modify_app()
