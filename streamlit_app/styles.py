def get_custom_css():
    return """
    <style>
        .main {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            color: #f8fafc;
        }

        .stApp {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        }

        section[data-testid="stSidebar"] {
            background: #0b1220;
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        .block-container {
            padding-top: 2.4rem;
            padding-bottom: 2.4rem;
            max-width: 1280px;
        }

        .hero-card {
            background: linear-gradient(135deg, rgba(37,99,235,0.18), rgba(14,165,233,0.10));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 24px;
            padding: 34px;
            margin-bottom: 24px;
            box-shadow: 0 12px 34px rgba(0,0,0,0.25);
        }

        .metric-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;

            padding: 28px;              /* 🔥 arttı */
            margin-bottom: 8px;         /* 🔥 azaldı */

            box-shadow: 0 10px 28px rgba(0,0,0,0.22);

            min-height: 220px;          /* 🔥 büyüdü */
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .metric-card h4 {
            font-size: 1.3rem;
            margin-bottom: 10px;
        }

        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 16px 36px rgba(0,0,0,0.35);
            transition: all 0.25s ease;
        }

        .metric-card h2 {
            font-size: 1.9rem;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .metric-card p {
            font-size: 1.05rem;
            color: #cbd5e1;
        }

        .info-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 26px;
            margin-bottom: 20px;
        }

        .result-good {
            background: rgba(34,197,94,0.12);
            border: 1px solid rgba(34,197,94,0.35);
            border-radius: 18px;
            padding: 20px;
            margin-top: 12px;
        }

        .result-warn {
            background: rgba(239,68,68,0.12);
            border: 1px solid rgba(239,68,68,0.35);
            border-radius: 18px;
            padding: 20px;
            margin-top: 12px;
        }

        h1 {
            color: #f8fafc !important;
            font-size: 2.9rem !important;
            font-weight: 800 !important;
        }

        h2 {
            color: #f8fafc !important;
            font-size: 2.15rem !important;
            font-weight: 700 !important;
        }

        h3 {
            color: #f8fafc !important;
            font-size: 1.55rem !important;
            font-weight: 700 !important;
        }

        h4 {
            color: #f8fafc !important;
            font-size: 1.22rem !important;
            font-weight: 600 !important;
        }

        p, li, label, div {
            color: #e5e7eb;
            font-size: 1.12rem !important;
            line-height: 1.75 !important;
        }

        ul {
            padding-left: 1.2rem;
        }
    </style>
    """
def get_upload_css():
    return """
    <style>
    /* Upload container */
    .upload-box {
        border: 2px dashed #3b82f6;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        background: rgba(59, 130, 246, 0.08);
        transition: all 0.3s ease;
    }

    .upload-box:hover {
        border-color: #60a5fa;
        background: rgba(59, 130, 246, 0.12);
    }

    .upload-title {
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .upload-sub {
        font-size: 14px;
        color: #cbd5e1;
        margin-bottom: 20px;
    }

    /* Streamlit default uploaderı güzelleştir */
    div[data-testid="stFileUploader"] {
        border: none;
    }

    div[data-testid="stFileUploader"] section {
        border: none !important;
        background: transparent !important;
    }

    div[data-testid="stFileUploader"] button {
        border-radius: 10px !important;
        padding: 8px 16px !important;
        font-size: 14px !important;
    }
    </style>
    """