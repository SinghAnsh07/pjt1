"""
Design system and custom CSS styling for the V7 Audio Authenticity Analyzer.

Implements a refined, dark audio-workspace aesthetic with purposeful semantic hierarchy,
glazed glassmorphism ("frosted glass") containers, high text contrast, and responsive layout
across desktop and mobile.
"""

def get_custom_css() -> str:
    """Returns the centralized CSS string for injection into the Streamlit application."""
    return """
    <style>
    /* =========================================================================
       CSS Custom Properties (Theme Design System & Glazed Glassmorphism)
       ========================================================================= */
    :root {
        --bg-main: #0C0F14;
        
        /* Glazed Glass Surface Tokens (Semi-opaque with high blur contrast) */
        --surface-glass-base: rgba(16, 21, 31, 0.88);
        --surface-glass-elevated: rgba(22, 28, 42, 0.90);
        --surface-glass-subtle: rgba(12, 16, 24, 0.85);
        --surface-glass-dense: rgba(14, 18, 27, 0.94);
        
        --surface-primary: rgba(16, 21, 31, 0.88);
        --surface-elevated: rgba(22, 28, 42, 0.90);
        --surface-subtle: rgba(28, 36, 52, 0.88);
        
        /* Glass Borders & Highlights */
        --border-glass: rgba(255, 255, 255, 0.12);
        --border-glass-subtle: rgba(255, 255, 255, 0.08);
        --border-glass-highlight: rgba(255, 255, 255, 0.22);
        --border-default: #2A3344;
        --border-subtle: rgba(42, 51, 68, 0.7);
        
        /* Shadows & Specular Insets */
        --shadow-glass: 0 10px 32px 0 rgba(0, 0, 0, 0.45), inset 0 1px 0 0 rgba(255, 255, 255, 0.14);
        --shadow-glass-sm: 0 4px 16px 0 rgba(0, 0, 0, 0.35), inset 0 1px 0 0 rgba(255, 255, 255, 0.10);
        --shadow-glass-lg: 0 16px 48px 0 rgba(0, 0, 0, 0.55), inset 0 1px 0 0 rgba(255, 255, 255, 0.18);
        
        /* Typography */
        --text-primary: #FFFFFF;
        --text-secondary: #CBD5E1;
        --text-muted: #94A3B8;
        
        /* Semantic Accents */
        --accent-primary: #8B9CFF;
        --accent-secondary: #65D6C4;
        
        /* Semantic Status (Real / Bona Fide) - Glazed Emerald */
        --color-positive: #4CC9A6;
        --color-positive-bg: linear-gradient(135deg, rgba(14, 28, 26, 0.90) 0%, rgba(11, 22, 21, 0.88) 100%);
        --color-positive-border: rgba(76, 201, 166, 0.40);
        
        /* Semantic Status (Fake / Caution) - Glazed Amber */
        --color-caution: #F5C06B;
        --color-caution-bg: linear-gradient(135deg, rgba(32, 23, 15, 0.92) 0%, rgba(22, 17, 18, 0.90) 100%);
        --color-caution-border: rgba(245, 192, 107, 0.45);
        
        /* Error */
        --color-error: #F87171;
        --color-error-bg: linear-gradient(135deg, rgba(34, 16, 18, 0.92) 0%, rgba(24, 13, 16, 0.90) 100%);
        --color-error-border: rgba(248, 113, 113, 0.45);
        
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --font-stack: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* =========================================================================
       Full-Bleed Outer Background Support
       ========================================================================= */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* =========================================================================
       Global Container & Typography
       ========================================================================= */
    html, body, [class*="css"] {
        font-family: var(--font-stack);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3.5rem;
        width: min(72vw, 1380px) !important;
        max-width: min(72vw, 1380px) !important;
        margin-left: auto !important;
        margin-right: auto !important;
        position: relative;
        z-index: 1;
    }

    /* =========================================================================
       Application Header (Glazed Glass)
       ========================================================================= */
    .app-header {
        margin-bottom: 1.25rem;
        padding: 1.15rem 1.4rem 1.05rem 1.4rem;
        background: var(--surface-glass-base);
        backdrop-filter: blur(18px) saturate(180%);
        -webkit-backdrop-filter: blur(18px) saturate(180%);
        border: 1px solid var(--border-glass);
        box-shadow: var(--shadow-glass);
        border-radius: var(--radius-md);
    }
    .app-header-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-bottom: 0.35rem;
    }
    .app-title-group {
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }
    .app-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 38px;
        height: 38px;
        background: rgba(139, 156, 255, 0.15);
        border: 1px solid rgba(139, 156, 255, 0.30);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.15);
        border-radius: var(--radius-sm);
        font-size: 1.25rem;
        color: var(--accent-primary);
    }
    .app-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.2;
    }
    .header-badges {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(22, 28, 42, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        color: var(--text-secondary);
        border: 1px solid var(--border-glass);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
        border-radius: 9999px;
        padding: 0.25rem 0.75rem;
        font-size: 0.80rem;
        font-weight: 500;
    }
    .badge-local-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: var(--color-positive);
        box-shadow: 0 0 8px rgba(76, 201, 166, 0.6);
        display: inline-block;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: var(--text-secondary);
        line-height: 1.5;
        margin: 0.35rem 0 0 0;
    }

    /* =========================================================================
       Introductory Section (Glazed Glass)
       ========================================================================= */
    .intro-card {
        background: var(--surface-glass-dense);
        backdrop-filter: blur(18px) saturate(180%);
        -webkit-backdrop-filter: blur(18px) saturate(180%);
        border: 1px solid var(--border-glass);
        box-shadow: var(--shadow-glass);
        border-radius: var(--radius-md);
        padding: 1rem 1.35rem;
        margin-bottom: 1.25rem;
    }
    .intro-headline {
        font-size: 1.02rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.3rem;
    }
    .intro-text {
        font-size: 0.90rem;
        color: var(--text-secondary);
        line-height: 1.5;
        margin: 0;
    }

    /* =========================================================================
       Upload & File Summary Cards (Glazed Glass)
       ========================================================================= */
    [data-testid="stFileUploader"] section {
        background: var(--surface-glass-base) !important;
        backdrop-filter: blur(18px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(18px) saturate(180%) !important;
        border: 1px dashed rgba(255, 255, 255, 0.24) !important;
        box-shadow: var(--shadow-glass) !important;
        border-radius: var(--radius-md) !important;
        padding: 1.5rem !important;
        transition: border-color 0.2s ease, background 0.2s ease;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: rgba(139, 156, 255, 0.5) !important;
        background: var(--surface-glass-elevated) !important;
    }
    
    div[data-testid="stAudio"] {
        background: var(--surface-glass-base) !important;
        backdrop-filter: blur(18px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(18px) saturate(180%) !important;
        border: 1px solid var(--border-glass) !important;
        box-shadow: var(--shadow-glass-sm) !important;
        border-radius: var(--radius-md) !important;
        padding: 0.85rem 1rem !important;
        margin-bottom: 1rem !important;
    }

    .upload-summary-card, .metadata-card {
        background: var(--surface-glass-base);
        backdrop-filter: blur(18px) saturate(180%);
        -webkit-backdrop-filter: blur(18px) saturate(180%);
        border: 1px solid var(--border-glass);
        box-shadow: var(--shadow-glass);
        border-radius: var(--radius-md);
        padding: 1.15rem 1.35rem;
        margin-top: 0.85rem;
        margin-bottom: 1.25rem;
    }
    .upload-summary-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.85rem;
        border-bottom: 1px solid var(--border-glass-subtle);
        padding-bottom: 0.55rem;
    }
    .upload-summary-title, .metadata-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        color: var(--text-muted);
        letter-spacing: 0.05em;
    }
    .upload-status-ready {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.80rem;
        color: var(--color-positive);
        font-weight: 600;
    }
    .metadata-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.85rem;
    }
    .metadata-item-label {
        font-size: 0.76rem;
        color: var(--text-muted);
        margin-bottom: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metadata-item-value {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-primary);
        font-family: monospace;
    }

    /* =========================================================================
       Results & Plain-Language Verdict Banners (Glazed Glass)
       ========================================================================= */
    .verdict-card {
        position: relative;
        border-radius: var(--radius-md);
        padding: 1.4rem 1.6rem;
        margin-top: 1.25rem;
        margin-bottom: 1.25rem;
        backdrop-filter: blur(20px) saturate(190%);
        -webkit-backdrop-filter: blur(20px) saturate(190%);
    }
    .verdict-real {
        background: var(--color-positive-bg);
        border: 1px solid var(--color-positive-border);
        box-shadow: 0 12px 36px 0 rgba(0, 0, 0, 0.45), inset 0 1px 0 0 rgba(76, 201, 166, 0.30);
    }
    .verdict-fake {
        background: var(--color-caution-bg);
        border: 1px solid var(--color-caution-border);
        box-shadow: 0 12px 36px 0 rgba(0, 0, 0, 0.45), inset 0 1px 0 0 rgba(245, 192, 107, 0.35);
    }
    .verdict-header-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-bottom: 0.65rem;
    }
    .verdict-title-real {
        color: var(--color-positive);
        font-size: 1.42rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin: 0;
        line-height: 1.25;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
    }
    .verdict-title-fake {
        color: var(--color-caution);
        font-size: 1.42rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin: 0;
        line-height: 1.25;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
    }
    .verdict-status-pill {
        display: inline-block;
        padding: 0.28rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }
    .pill-real {
        background: rgba(76, 201, 166, 0.20);
        color: #5EEAD4;
        border: 1px solid rgba(76, 201, 166, 0.50);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
    }
    .pill-fake {
        background: rgba(245, 192, 107, 0.20);
        color: #FDE047;
        border: 1px solid rgba(245, 192, 107, 0.50);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
    }
    .verdict-desc {
        color: var(--text-primary);
        font-size: 0.98rem;
        line-height: 1.6;
        margin-bottom: 0.85rem;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.6);
    }
    .verdict-disclaimer {
        color: #E2E8F0;
        font-size: 0.86rem;
        line-height: 1.55;
        background: rgba(10, 14, 22, 0.72);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.10);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
        border-radius: var(--radius-sm);
        padding: 0.75rem 1rem;
        margin-top: 0.85rem;
    }
    .verdict-disclaimer strong {
        color: #FFFFFF;
        font-weight: 600;
    }

    /* Score Presentation Strip (Glazed Glass Nested Module) */
    .score-strip {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 1.35rem;
        background: rgba(10, 14, 22, 0.82);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: var(--radius-sm);
        padding: 0.85rem 1.15rem;
        margin: 0.95rem 0;
        box-shadow: inset 0 1px 0 0 rgba(255, 255, 255, 0.10), 0 4px 16px rgba(0, 0, 0, 0.35);
    }
    .score-item {
        display: flex;
        flex-direction: column;
    }
    .score-label {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.15rem;
    }
    .score-val {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
        font-family: monospace;
    }
    .score-explanation {
        font-size: 0.84rem;
        color: var(--text-secondary);
        flex: 1 1 200px;
        line-height: 1.45;
    }

    /* Results Summary Grid (Glazed Glass) */
    .results-summary-card {
        background: var(--surface-glass-base);
        backdrop-filter: blur(18px) saturate(180%);
        -webkit-backdrop-filter: blur(18px) saturate(180%);
        border: 1px solid var(--border-glass);
        box-shadow: var(--shadow-glass);
        border-radius: var(--radius-md);
        padding: 1.15rem 1.4rem;
        margin-top: 1rem;
        margin-bottom: 1.25rem;
    }
    .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 0.85rem;
    }
    .summary-cell-label {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.2rem;
    }
    .summary-cell-value {
        font-size: 1.02rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    /* =========================================================================
       Streamlit Metric Cards (Glazed Glass Tiles)
       ========================================================================= */
    div[data-testid="stMetric"] {
        background: var(--surface-glass-base) !important;
        backdrop-filter: blur(18px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(18px) saturate(180%) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: var(--radius-md) !important;
        padding: 0.95rem 1.15rem !important;
        box-shadow: var(--shadow-glass-sm) !important;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(139, 156, 255, 0.40) !important;
        transform: translateY(-1px);
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        font-size: 1.55rem !important;
        font-family: monospace !important;
    }
    
    div[data-testid="stCaptionContainer"] {
        background: var(--surface-glass-dense) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid var(--border-glass-subtle) !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.75rem 1rem !important;
        margin-top: 0.75rem !important;
        margin-bottom: 1rem !important;
        color: var(--text-secondary) !important;
        line-height: 1.55 !important;
        box-shadow: var(--shadow-glass-sm) !important;
    }

    div[data-testid="stDataFrame"] {
        background: var(--surface-glass-base) !important;
        backdrop-filter: blur(18px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(18px) saturate(180%) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: var(--radius-md) !important;
        box-shadow: var(--shadow-glass) !important;
        overflow: hidden !important;
        padding: 0.25rem !important;
    }

    div[data-testid="stExpander"] {
        background: var(--surface-glass-base) !important;
        backdrop-filter: blur(18px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(18px) saturate(180%) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: var(--radius-md) !important;
        box-shadow: var(--shadow-glass) !important;
        margin-top: 1rem !important;
        margin-bottom: 1.25rem !important;
        overflow: hidden !important;
    }
    div[data-testid="stExpander"] summary {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        padding: 0.85rem 1.15rem !important;
    }

    /* =========================================================================
       Audio Signal Insights Section (Glazed Glass Tabs)
       ========================================================================= */
    .insights-header {
        margin-top: 1.75rem;
        margin-bottom: 0.65rem;
    }
    .insights-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }
    .insights-subtitle {
        font-size: 0.90rem;
        color: var(--text-secondary);
        margin-bottom: 0.85rem;
    }
    .insights-context-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.65rem;
        background: var(--surface-glass-dense);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-glass);
        box-shadow: var(--shadow-glass-sm);
        border-radius: 9999px;
        padding: 0.35rem 0.95rem;
        font-size: 0.82rem;
        color: var(--text-secondary);
        margin-bottom: 0.95rem;
    }

    /* Streamlit Tab Customization (Glazed Glass Tab Panel) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 1px solid var(--border-glass);
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.6rem 1.1rem;
        font-size: 0.90rem;
        font-weight: 500;
        color: var(--text-secondary);
        border-radius: var(--radius-sm) var(--radius-sm) 0 0;
        background: transparent;
        transition: color 0.2s ease, background 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
        background: rgba(255, 255, 255, 0.04);
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent-primary) !important;
        border-bottom: 2px solid var(--accent-primary) !important;
        font-weight: 600 !important;
        background: rgba(139, 156, 255, 0.08) !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: var(--surface-glass-base);
        backdrop-filter: blur(18px) saturate(180%);
        -webkit-backdrop-filter: blur(18px) saturate(180%);
        border: 1px solid var(--border-glass);
        border-top: none;
        border-radius: 0 0 var(--radius-md) var(--radius-md);
        padding: 1.35rem 1.45rem;
        box-shadow: var(--shadow-glass);
    }

    /* =========================================================================
       Export & Action Cards (Glazed Glass)
       ========================================================================= */
    .export-card {
        background: var(--surface-glass-base);
        backdrop-filter: blur(18px) saturate(180%);
        -webkit-backdrop-filter: blur(18px) saturate(180%);
        border: 1px solid var(--border-glass);
        box-shadow: var(--shadow-glass);
        border-radius: var(--radius-md);
        padding: 1.25rem 1.4rem;
        margin-top: 1.35rem;
        margin-bottom: 1.25rem;
    }
    .export-title {
        font-size: 1.20rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.3rem;
    }
    .export-desc {
        font-size: 0.88rem;
        color: var(--text-secondary);
        margin-bottom: 1.15rem;
        line-height: 1.5;
    }

    /* =========================================================================
       Privacy and Limitations Information Cards (Glazed Glass)
       ========================================================================= */
    .info-dual-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.15rem;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
    }
    .info-card {
        background: var(--surface-glass-base);
        backdrop-filter: blur(18px) saturate(180%);
        -webkit-backdrop-filter: blur(18px) saturate(180%);
        border: 1px solid var(--border-glass);
        box-shadow: var(--shadow-glass);
        border-radius: var(--radius-md);
        padding: 1.15rem 1.35rem;
    }
    .info-card-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.45rem;
    }
    .info-card-text {
        font-size: 0.85rem;
        color: var(--text-secondary);
        line-height: 1.55;
        margin: 0;
    }

    .footer-note {
        margin-top: 2rem;
        border-top: 1px solid var(--border-glass-subtle);
        padding-top: 1.15rem;
        color: var(--text-muted);
        font-size: 0.84rem;
        line-height: 1.5;
    }

    /* =========================================================================
       Alerts & Notifications (Glazed Glass)
       ========================================================================= */
    div[data-testid="stAlert"] {
        background: var(--surface-glass-dense) !important;
        backdrop-filter: blur(18px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(18px) saturate(180%) !important;
        border: 1px solid var(--border-glass) !important;
        box-shadow: var(--shadow-glass) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
    }

    /* =========================================================================
       SpecularButton & Native Trigger Styles
       ========================================================================= */
    .specular-native-trigger {
        position: absolute !important;
        width: 0px !important;
        height: 0px !important;
        overflow: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .specular-native-trigger div[data-testid="stButton"] {
        margin: 0 !important;
        padding: 0 !important;
        height: 0px !important;
    }

    /* =========================================================================
       MagicBento Methodology Section (Glazed Glass)
       ========================================================================= */
    .methodology-header {
        margin-top: 1.85rem;
        margin-bottom: 0.85rem;
        padding: 1.05rem 1.35rem;
        background: var(--surface-glass-base);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-md);
        backdrop-filter: blur(18px) saturate(180%);
        -webkit-backdrop-filter: blur(18px) saturate(180%);
        box-shadow: var(--shadow-glass);
    }
    .methodology-title {
        font-size: 1.12rem;
        font-weight: 700;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 0 0 0.25rem 0;
    }
    .methodology-subtitle {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin: 0;
        line-height: 1.5;
    }

    /* =========================================================================
       Responsive Rules
       ========================================================================= */
    @media (max-width: 1024px) {
        .block-container {
            width: 92vw !important;
            max-width: 92vw !important;
            padding-left: 1.25rem;
            padding-right: 1.25rem;
        }
    }

    @media (max-width: 768px) {
        .block-container {
            width: 95vw !important;
            max-width: 95vw !important;
            padding-top: 1rem;
            padding-bottom: 2rem;
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }
        .app-title {
            font-size: 1.45rem;
        }
        .app-header-top {
            flex-direction: column;
            align-items: flex-start;
        }
        .header-badges {
            margin-top: 0.25rem;
        }
        .info-dual-grid {
            grid-template-columns: 1fr;
        }
        .metadata-grid, .summary-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    @media (max-width: 480px) {
        .block-container {
            width: 96vw !important;
            max-width: 96vw !important;
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
        .metadata-grid, .summary-grid {
            grid-template-columns: 1fr;
        }
        .score-strip {
            flex-direction: column;
            align-items: flex-start;
            gap: 0.65rem;
        }
    }
    </style>
    """
