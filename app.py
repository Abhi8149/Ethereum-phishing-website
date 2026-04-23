from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from inference import fetch_etherscan_transaction_record, load_predictor


st.set_page_config(page_title="Ethereum Wallet Risk Checker", page_icon="ETH", layout="wide")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Manrope:wght@400;500;600;700&display=swap');

        :root {
            --bg-0: #070b12;
            --bg-1: #0b1220;
            --panel: rgba(14, 23, 40, 0.78);
            --panel-strong: rgba(17, 28, 48, 0.92);
            --border: rgba(148, 163, 184, 0.22);
            --text-main: #e2e8f0;
            --text-dim: #94a3b8;
            --accent-cyan: #22d3ee;
            --accent-emerald: #34d399;
            --accent-amber: #f59e0b;
            --accent-rose: #fb7185;
            --shadow: 0 24px 60px rgba(2, 6, 23, 0.45);
        }

        .stApp {
            background:
                radial-gradient(1100px 520px at 6% -5%, rgba(34, 211, 238, 0.16), transparent 52%),
                radial-gradient(850px 420px at 95% 4%, rgba(251, 113, 133, 0.14), transparent 53%),
                linear-gradient(160deg, var(--bg-0), var(--bg-1) 55%);
            color: var(--text-main);
            font-family: 'Manrope', sans-serif;
        }

        .block-container {
            padding-top: 2.25rem;
            padding-bottom: 2.5rem;
            max-width: 1100px;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 1.75rem 1.75rem 1.45rem;
            border-radius: 18px;
            background: linear-gradient(150deg, rgba(17, 28, 48, 0.92), rgba(9, 16, 30, 0.92));
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
            margin-bottom: 1.4rem;
            animation: slide-up 0.45s ease-out;
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 330px;
            height: 330px;
            right: -95px;
            top: -160px;
            background: radial-gradient(circle, rgba(52, 211, 153, 0.28), transparent 65%);
            pointer-events: none;
        }

        .hero h1 {
            margin: 0;
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(1.8rem, 2.8vw, 2.45rem);
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #f8fafc;
        }

        .hero p {
            margin: 0.6rem 0 0;
            color: #bfd0e5;
            font-size: 0.98rem;
            line-height: 1.5;
            max-width: 760px;
        }

        .hero-badges {
            margin-top: 1rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
        }

        .hero-badge {
            border-radius: 999px;
            padding: 0.35rem 0.72rem;
            font-size: 0.74rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            border: 1px solid rgba(148, 163, 184, 0.32);
            color: #dbeafe;
            background: rgba(30, 41, 59, 0.55);
        }

        .input-container {
            padding: 1.05rem 1.15rem 0.8rem;
            border-radius: 14px;
            background: var(--panel);
            backdrop-filter: blur(6px);
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
            margin-bottom: 0.75rem;
        }

        .input-label {
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            color: #cbd5e1;
            margin-bottom: 0.4rem;
            display: block;
        }

        .input-hint {
            color: var(--text-dim);
            font-size: 0.85rem;
            margin: 0;
        }

        .result-card {
            position: relative;
            overflow: hidden;
            padding: 1.1rem 1rem;
            border-radius: 14px;
            border: 1px solid var(--border);
            background: var(--panel-strong);
            box-shadow: var(--shadow);
            animation: slide-up 0.5s ease-out;
        }

        .result-card::before {
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, rgba(34, 211, 238, 0.95), rgba(52, 211, 153, 0.95));
        }

        .result-title {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #9fb0c6;
            margin-bottom: 0.8rem;
            font-weight: 700;
        }

        .result-value {
            font-size: 1.55rem;
            font-weight: 700;
            margin: 0.2rem 0 0.6rem;
            letter-spacing: -0.02em;
            color: #f8fafc;
        }

        .result-sub {
            color: #9fb0c6;
            font-size: 0.85rem;
            font-weight: 400;
        }

        .result-probability {
            font-size: 0.9rem;
            color: #cbd5e1;
            margin-top: 0.5rem;
        }

        .good {
            color: var(--accent-emerald);
            font-weight: 700;
        }

        .bad {
            color: #f87171;
            font-weight: 700;
        }

        .neutral {
            color: var(--accent-amber);
        }

        .metric-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }

        .metric-box {
            padding: 1rem;
            border-radius: 14px;
            background: var(--panel);
            border: 1px solid var(--border);
            text-align: center;
            box-shadow: 0 10px 24px rgba(2, 6, 23, 0.35);
        }

        .metric-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #9fb0c6;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .metric-value {
            font-size: 1.45rem;
            font-weight: 700;
            color: #f8fafc;
        }

        .dataframe {
            background-color: transparent !important;
            border-radius: 12px;
        }

        .section-title {
            font-size: 1.1rem;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 650;
            color: #e2e8f0;
            margin: 1.8rem 0 1rem 0;
            padding-bottom: 0.72rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.24);
        }

        .stButton > button {
            width: 100%;
            border-radius: 12px;
            padding: 0.81rem 1.5rem;
            font-weight: 600;
            font-size: 0.95rem;
            letter-spacing: 0;
            background: linear-gradient(120deg, #22d3ee, #06b6d4 45%, #10b981);
            border: 1px solid rgba(34, 211, 238, 0.55);
            color: white;
            box-shadow: 0 10px 22px rgba(13, 148, 136, 0.35);
            transition: transform 0.16s ease, filter 0.16s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            filter: brightness(1.06);
            border-color: rgba(34, 211, 238, 0.7);
        }

        .stTextInput > div > div > input {
            border-radius: 10px;
            border: 1px solid rgba(148, 163, 184, 0.36);
            background: rgba(15, 23, 42, 0.78);
            color: #e2e8f0;
            padding: 0.75rem 1rem;
            font-size: 0.95rem;
        }

        .stTextInput > div > div > input:focus {
            border-color: rgba(34, 211, 238, 0.65);
            box-shadow: 0 0 0 2px rgba(34, 211, 238, 0.18);
        }

        .stSidebar {
            background: #070d1a;
            border-right: 1px solid rgba(148, 163, 184, 0.2);
        }

        .stAlert {
            border-radius: 12px;
        }

        .stMarkdown strong {
            color: #dbeafe;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 24px rgba(2, 6, 23, 0.3);
        }

        @keyframes slide-up {
            from {
                opacity: 0;
                transform: translateY(8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @media (max-width: 760px) {
            .block-container {
                padding-top: 1.3rem;
                padding-bottom: 2rem;
            }

            .hero {
                padding: 1.3rem 1.1rem 1.2rem;
                border-radius: 14px;
            }

            .hero-badges {
                gap: 0.45rem;
            }

            .hero-badge {
                font-size: 0.69rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_predictor():
    return load_predictor(str(Path.cwd()))


inject_css()

st.markdown(
    """
    <div class="hero">
        <h1>Ethereum Wallet Risk Checker</h1>
        <p>Analyze wallet behavior with graph learning signals and transaction patterns. Enter an address to run both trained models and review a confidence-based verdict.</p>
        <div class="hero-badges">
            <span class="hero-badge">DA-HGNN Intelligence</span>
            <span class="hero-badge">GAE + Random Forest</span>
            <span class="hero-badge">Live Etherscan Data</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="input-container">
        <label class="input-label">Wallet address</label>
        <p class="input-hint">Use a full Ethereum address. The app fetches recent transactions and computes phishing risk.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_input, col_button = st.columns([4, 1], gap="small")

with col_input:
    address = st.text_input(
        "Wallet / account address",
        placeholder="0x...",
        label_visibility="collapsed",
    )

with col_button:
    run_button = st.button("Analyze", use_container_width=True, type="primary")


if run_button:
    if not address.strip():
        st.error("Please enter a valid Ethereum wallet address.")
    else:
        with st.spinner("Analyzing wallet security..."):
            try:
                transaction_record = None
                transaction_error = None
                try:
                    transaction_record = fetch_etherscan_transaction_record(address, limit=50)
                except Exception as error:
                    transaction_error = str(error)

                prediction_result = None
                prediction_error = None
                if transaction_record is None:
                    prediction_error = transaction_error or "Prediction requires fetched transactions from Etherscan."
                else:
                    try:
                        predictor = get_predictor()
                        prediction_result = predictor.predict(address, tx_record=transaction_record)
                    except Exception as error:
                        prediction_error = str(error)

                if prediction_result is not None:
                    verdict_color = "#f87171" if prediction_result.final_label == "Phishing" else "#34d399"
                    st.markdown(
                        f"""
                        <div style="padding: 1.25rem; border-radius: 14px; background: rgba(17, 28, 48, 0.9); border: 1px solid rgba(148, 163, 184, 0.24); text-align: center; margin: 1.1rem 0; border-left: 4px solid {verdict_color}; box-shadow: 0 18px 42px rgba(2, 6, 23, 0.4);">
                            <p style="margin: 0; font-size: 0.78rem; color: #9fb0c6; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700;">
                                Final Verdict
                            </p>
                            <p style="margin: 0.5rem 0 0; font-size: 2rem; font-weight: 700; color: {verdict_color}; font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.02em;">
                                {prediction_result.final_label}
                            </p>
                            <p style="margin: 0.6rem 0 0; font-size: 0.97rem; color: #cbd5e1; font-weight: 500;">
                                Risk score: <span style="color: {verdict_color}; font-weight: 700;">{prediction_result.final_probability:.1%}</span>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown("<h3 class='section-title'>Model predictions</h3>", unsafe_allow_html=True)

                    col_model1, col_model2, col_consensus = st.columns(3, gap="medium")

                    with col_model1:
                        label_color = "bad" if prediction_result.dahgnn_label == "Phishing" else "good"
                        st.markdown(
                            f"""
                            <div class="result-card">
                                <div class="result-title">DA-HGNN</div>
                                <div class="result-value {label_color}">{prediction_result.dahgnn_label}</div>
                                <div class="result-probability">Probability: <strong>{prediction_result.dahgnn_probability:.2%}</strong></div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with col_model2:
                        label_color = "bad" if prediction_result.gae_rf_label == "Phishing" else "good"
                        st.markdown(
                            f"""
                            <div class="result-card">
                                <div class="result-title">GAE + Random Forest</div>
                                <div class="result-value {label_color}">{prediction_result.gae_rf_label}</div>
                                <div class="result-probability">Probability: <strong>{prediction_result.gae_rf_probability:.2%}</strong></div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with col_consensus:
                        label_color = "bad" if prediction_result.final_label == "Phishing" else "good"
                        st.markdown(
                            f"""
                            <div class="result-card" style="border-left: 4px solid rgba(34, 211, 238, 0.8);">
                                <div class="result-title">Consensus</div>
                                <div class="result-value {label_color}">{prediction_result.final_label}</div>
                                <div class="result-probability" style="color: #cbd5e1;">Weighted score: <strong>{prediction_result.final_probability:.2%}</strong></div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                
                elif prediction_error:
                    st.warning(prediction_error)

                st.markdown("<h3 class='section-title'>Transaction history</h3>", unsafe_allow_html=True)
                
                if transaction_record is not None and not transaction_record.transactions.empty:
                    summary = transaction_record.summary
                    
                    metric_cols = st.columns(4, gap="small")
                    metrics = [
                        ("Sent", f"{summary['sent_transactions']}"),
                        ("Received", f"{summary['received_transactions']}"),
                        ("Total tx", f"{summary['total_transactions']}"),
                        ("Counterparties", f"{summary['unique_counterparties']}"),
                    ]
                    
                    for col, (label, value) in zip(metric_cols, metrics):
                        with col:
                            st.markdown(
                                f"""
                                <div class="metric-box">
                                    <div class="metric-label">{label}</div>
                                    <div class="metric-value">{value}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    
                    time_cols = st.columns(2, gap="medium")
                    with time_cols[0]:
                        st.markdown(f"**First transaction:** {summary['first_transaction'] or 'N/A'}")
                    with time_cols[1]:
                        st.markdown(f"**Last transaction:** {summary['last_transaction'] or 'N/A'}")
                    
                    st.write("")
                    display_df = transaction_record.transactions.copy()
                    if "timeStamp" in display_df.columns:
                        display_df["timeStamp"] = pd.to_datetime(display_df["timeStamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                    
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                else:
                    st.info(transaction_error or "No transaction history available for this wallet.")

                if prediction_result is None and not prediction_error:
                    with st.expander("Why no prediction is shown"):
                        st.write(
                            "A prediction couldn't be generated if:\n"
                            "- Required model artifacts are missing (best_dahgnn_model.pt, gae_pdna_weights.pth, rf_classifier.pkl)\n"
                            "- Etherscan API is unreachable\n"
                            "- The wallet has no usable transaction history\n"
                            "- The address format is invalid"
                        )

            except FileNotFoundError as error:
                st.error(f"File error: {error}")
            except KeyError as error:
                st.error(f"Configuration error: {error}")
            except Exception as error:
                st.error(f"Analysis failed: {error}")
