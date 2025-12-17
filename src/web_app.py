"""
FastAPI Web Application for Phishing Detection.
Serves a monochrome, Pinterest-style interface and provides an inference endpoint.
"""

from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.inference import predict_email_bert
from src.config import config

app = FastAPI(title="Phishing Detector Web App")

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Ensure directories exist
TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Setup templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze_email(email_text: str = Form(...)) -> Dict[str, Any]:
    """
    Analyze email text using the BERT model.
    Returns JSON with prediction, explanation, and detailed analysis.
    """
    from src.features import _extract_metadata_features
    import re
    
    label, proba, explanation = predict_email_bert(email_text)
    
    # Extract metadata for detailed analysis
    metadata = _extract_metadata_features([email_text])[0]
    
    # Analyze risk factors
    risk_factors = []
    safe_indicators = []
    
    # URL analysis
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_text)
    if urls:
        risk_factors.append({"factor": "Contains URLs", "count": len(urls), "severity": "high" if len(urls) > 2 else "medium"})
    else:
        safe_indicators.append({"indicator": "No suspicious URLs", "type": "links"})
    
    # Urgency keywords
    urgency_words = ['urgent', 'immediate', 'verify', 'suspended', 'expire', 'confirm', 'click here', 'act now']
    found_urgency = [word for word in urgency_words if word.lower() in email_text.lower()]
    if found_urgency:
        risk_factors.append({"factor": "Urgency language detected", "examples": found_urgency[:3], "severity": "high"})
    
    # Special characters and digits ratio
    if metadata[1] > 15:  # digit_ratio index
        risk_factors.append({"factor": "High digit count", "value": f"{metadata[1]:.0f}%", "severity": "medium"})
    
    if metadata[2] > 20:  # special_char_ratio index
        risk_factors.append({"factor": "Unusual special characters", "value": f"{metadata[2]:.0f}%", "severity": "medium"})
    
    # Text length analysis
    if metadata[3] < 50:
        risk_factors.append({"factor": "Unusually short email", "value": f"{int(metadata[3])} chars", "severity": "low"})
    else:
        safe_indicators.append({"indicator": "Reasonable length", "type": "content"})
    
    # Greeting and professional tone
    if any(word in email_text.lower() for word in ['dear customer', 'dear user', 'dear member']):
        risk_factors.append({"factor": "Generic greeting", "severity": "medium"})
    elif any(word in email_text.lower() for word in ['hi', 'hello', 'dear']):
        safe_indicators.append({"indicator": "Personalized greeting", "type": "tone"})
    
    # Recommendation
    if label == 1:  # Phishing
        recommendation = "⚠️ This email shows multiple signs of phishing. Do not click any links, download attachments, or provide personal information."
        action = "Delete this email immediately and report it as phishing."
    else:
        recommendation = "✓ This email appears legitimate based on our analysis."
        action = "Always verify sender identity and be cautious with unexpected requests."
    
    return {
        "is_phishing": bool(label == 1),
        "probability": float(proba),
        "confidence_level": "High" if abs(proba - 0.5) > 0.3 else "Medium" if abs(proba - 0.5) > 0.15 else "Low",
        "explanation": explanation,
        "risk_factors": risk_factors,
        "safe_indicators": safe_indicators,
        "metadata": {
            "url_count": int(metadata[0]),
            "text_length": int(metadata[3]),
            "avg_word_length": round(metadata[4], 1)
        },
        "recommendation": recommendation,
        "action": action
    }
