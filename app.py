import imaplib
import smtplib
import email
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import email.utils
import requests
import re
import time
import html
import io
import os
from concurrent.futures import ThreadPoolExecutor
import streamlit as st

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

# ═══════════════════════════ INITIALIZATION & STATE GUARD ════════════════════════════════════
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "ui_logs" not in st.session_state:
    st.session_state.ui_logs = ["🌐 Autonomous Background Engine Framework Activated. Standing by for unread data streams..."]

def ui_print(text: str):
    """Appends live processing data directly to the Streamlit UI memory array with timestamps."""
    print(text)
    timestamp = time.strftime('%H:%M:%S')
    st.session_state.ui_logs.append(f"[{timestamp}] {text}")

# ═══════════════════════════ SECURED INFRASTRUCTURE SETTINGS ════════════════════════════════════
try:
    EMAIL_USER = st.secrets["EMAIL_USER"]
    EMAIL_PASS = st.secrets["EMAIL_PASS"]
    NVAPI_KEY  = st.secrets["NVAPI_KEY"]
except Exception:
    st.error("🔒 Security Alert: Configuration parameters are absent. Please define EMAIL_USER, EMAIL_PASS, and NVAPI_KEY inside your Streamlit Secrets Management Tab.")
    st.stop()

NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
FAST_MODEL     = "meta/llama-3.1-8b-instruct"   
STRONG_MODEL   = "meta/llama-3.3-70b-instruct"  

POLL_INTERVAL_SECONDS = 15  
PROCESSED_IDS_FILE    = "processed_email_ids.txt"

SKIP_SENDER_PATTERNS = [
    str(EMAIL_USER).lower(), "noreply", "no-reply", "donotreply", "do-not-reply",
    "mailer-daemon", "postmaster", "bounce", "notifications@", "alert@",
    "support@", "automated@", "newsletter@",
]

_session = requests.Session()
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────── Romanized Fingerprints ──────────────────────────
ROMANIZED_FINGERPRINTS = {
    "Telugu": [
        "unaaru", "unnaru", "unnaav", "ela unav", "chestunav", "chestunnav",
        "naku", "meeru", "emi chestunav", "chey", "cheppandi", "ledu", "undi",
        "avutundi", "chesanu", "vachanu", "veltanu", "chudandi",
        "manchi", "samacharam", "kadha", "kaadu", "aite", "aithe",
        "ante", "antey", "meeku", "mee ku", "ela unnav",
        "WB:naku", "WB:mee", "WB:oka", "WB:mari",
    ],
    "Hindi": [
        "kya haal", "theek hoon", "namaste", "tumhara", "mujhe",
        "kaisa hai", "kaise ho", "bhai yaar", "batao", "dekho", "kyunki",
        "WB:kya", "WB:hai", "WB:hain", "WB:nahi", "WB:bhai", "WB:yaar",
        "WB:acha", "WB:theek", "WB:tum", "WB:hoon", "WB:aap",
        "WB:mere", "WB:mera", "WB:woh", "WB:hoga", "WB:phir",
    ],
    "Tamil": [
        "eppadi", "irukkeenga", "irukkinga", "vanakkam", "irukken",
        "theriyum", "theriyala", "sollanga", "mudiyuma", "paakalam",
        "ungaluku", "enakku",
        "WB:nalla", "WB:sollu", "WB:paar", "WB:thambi", "WB:akka",
        "WB:enna", "WB:romba", "WB:konjam", "WB:vaanga", "WB:ponga",
        "WB:seri",
    ],
}

def _kw_score(text_lower: str, kw: str) -> int:
    if kw.startswith("WB:"):
        word = kw[3:]
        return 1 if re.search(r'\b' + re.escape(word) + r'\b', text_lower) else 0
    return 1 if kw in text_lower else 0

def local_romanized_detect(text: str) -> str | None:
    text_lower = text.lower()
    scores = {}
    for lang, keywords in ROMANIZED_FINGERPRINTS.items():
        score = sum(_kw_score(text_lower, kw) for kw in keywords)
        if score > 0: scores[lang] = score
    if not scores: return None
    best_lang  = max(scores, key=scores.get)
    if scores[best_lang] < 2: return None
    return best_lang

def _clean_language_string(raw: str) -> str:
    raw = re.sub(r'\s*\(.*?\)', '', raw)      
    raw = re.sub(r'\s*[-–].*$', '', raw)       
    raw = re.sub(r'\s+', ' ', raw).strip()
    parts = [p.strip().title() for p in raw.split(',') if p.strip()]
    return ', '.join(parts)

def _call_api(model: str, messages: list, max_tokens: int = 512, temperature: float = 0.0) -> str | None:
    headers = {"Authorization": f"Bearer {NVAPI_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    try:
        r = _session.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    return None

def detect_language_and_tone(text: str) -> tuple[str, str]:
    local_lang = local_romanized_detect(text)
    system_prompt = (
        "You are an expert linguist. Analyze the text and return EXACTLY two lines:\n"
        "LANGUAGE: <comma-separated language names>\n"
        "TONE: <Friendly or Formal>\n\n"
        "Return ONLY the two lines. No extra text."
    )
    hint = f"\n[Local pre-scan detected: {local_lang}]" if local_lang else ""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "ela unaaru bro? naku call chey free unapudu."},
        {"role": "assistant", "content": "LANGUAGE: Telugu\nTONE: Friendly"},
        {"role": "user", "content": f"Analyze:{hint}\n\n{text[:6000]}"},
    ]
    result = _call_api(FAST_MODEL, messages, max_tokens=60)
    language, tone = "English", "Formal"   
    if result:
        for line in result.strip().splitlines():
            upper = line.upper()
            if upper.startswith("LANGUAGE:"): language = _clean_language_string(line.split(":", 1)[1].strip())
            elif upper.startswith("TONE:"):
                raw_tone = line.split(":", 1)[1].strip().rstrip(".")
                tone = "Friendly" if "friend" in raw_tone.lower() else "Formal"
    if local_lang and language.lower() == "english": language = _clean_language_string(local_lang)
    return language, tone

def translate_to_english(text: str, detected_languages: str) -> str | None:
    if detected_languages.strip().lower() == "english": return text
    messages = [
        {"role": "system", "content": "You are an elite translator specializing in South Asian languages. Translate ALL non-English content into fluent, natural English. Output ONLY the English translation — no labels, no notes."},
        {"role": "user", "content": text[:12000]},
    ]
    return _call_api(STRONG_MODEL, messages, max_tokens=1500)

def draft_english_reply(english_text: str, tone: str) -> str | None:
    persona_prompt = (
        "You are a real human writing a direct personal email response. You must act completely as the primary profile defined below:\n\n"
        "YOUR PERSONAL DATA ENGINE:\n"
        "- Name: Vishnu (Full name: Boddu Vishnu Vardhan Reddy)\n"
        "- Occupation: Computer Science & Engineering student at Amity University\n"
        "- Academic Focus: Python algorithms, complexity analysis, text scraping, and Natural Language Processing (NLP)\n"
        "- Plan Parameters: Actively preparing for the GATE examination\n"
        "- Hobbies: Major follower of T20 cricket and live IPL statistics (RCB, SRH, KKR), tracking regional Telugu cinema movie releases, and playing Free Fire MAX\n"
        "- Daily Food Habits: Eats South Indian breakfast (Idli/Dosa/Poha) with coffee in the morning; Rice, dal, roti, and curries for lunch and dinner\n\n"
        f"STYLISTIC ALIGNMENT: Match formatting rules to the calculated tone parameter: **{tone}**. Output ONLY the response text payload itself."
    )
    messages = [
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content": "Hey, what are you up to? Did you have breakfast?"},
        {"role": "assistant", "content": "Hey! Just wrapping up some Python scripting for an NLP lab assignment and looking over some sorting algorithm complexity metrics. Yeah, just had some idli and coffee a bit ago. What's going on with you?"},
        {"role": "user", "content": f"Draft a personal response to this message:\n\n{english_text[:5000]}"},
    ]
    return _call_api(STRONG_MODEL, messages, max_tokens=600, temperature=0.5)

def translate_to_native(english_reply: str, target_language: str, tone: str) -> str | None:
    if "english" in target_language.lower() and "," not in target_language: return english_reply
    messages = [
        {"role": "system", "content": f"You are a native speaker of {target_language}. Translate the English reply into natural, idiomatic {target_language} matching a {tone} register. Output ONLY the final clean translation payload."},
        {"role": "user", "content": english_reply},
    ]
    return _call_api(STRONG_MODEL, messages, max_tokens=1000)

def run_qa_audit(english_draft: str, native_reply: str, target_tone: str, target_lang: str) -> tuple[int, str]:
    messages = [
        {"role": "system", "content": f"You are a QA compliance bot checking translation chains. Compare the original English draft response with its translation into {target_lang}. The specified stylistic parameter target was **{target_tone}**. Format exactly as:\nSCORE: <integer 1-5>\nANALYSIS: <one sentence feedback string>"},
        {"role": "user", "content": f"English draft:\n{english_draft}\n\nTranslated reply:\n{native_reply}"},
    ]
    result = _call_api(FAST_MODEL, messages, max_tokens=80)
    score, analysis = 5, "Audit verification completed successfully."
    if result:
        for line in result.strip().splitlines():
            if line.upper().startswith("SCORE:"):
                try: score = int(re.search(r'\d', line.split(":", 1)[1]).group())
                except Exception: pass
            elif line.upper().startswith("ANALYSIS:"): analysis = line.split(":", 1)[1].strip()
    return score, analysis

def clean_html(html_text: str) -> str:
    text = html.unescape(html_text)
    text = re.sub(r"<style[^>]*>[\s\S]*?</style>|<script[^>]*>[\s\S]*?</script>|<[^>]+>", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()

def parse_email_body(msg) -> tuple[str, list]:
    body, html_body, images = "", "", []
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            raw = part.get_payload(decode=True)
            if ct.startswith("image/") and raw: images.append(raw)
            elif ct == "text/plain" and not body and raw: body = raw.decode(errors="ignore")
            # FIX: Removed the rogue 'biases' syntax token below
            elif ct == "text/html" and not html_body and raw: html_body = raw.decode(errors="ignore")
    else:
        ct, raw = msg.get_content_type(), msg.get_payload(decode=True)
        if raw:
            if ct.startswith("image/"): images.append(raw)
            elif ct == "text/html": html_body = raw.decode(errors="ignore")
            else: body = raw.decode(errors="ignore")
    return (clean_html(html_body) if html_body.strip() else clean_html(body)).strip(), images

def send_reply(recipient: str, subject: str, body_text: str):
    try:
        msg = MIMEMultipart()
        msg["From"], msg["To"], msg["Subject"] = EMAIL_USER, recipient, subject
        msg.attach(MIMEText(body_text, "plain", "utf-8"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.sendmail(EMAIL_USER, recipient, msg.as_string())
        ui_print(f"🚀 Outbound Reply sent successfully to → {recipient}")
    except Exception as e:
        ui_print(f"❌ Outbound SMTP Delivery Core Failure: {e}")

def load_processed_ids() -> set:
    if not os.path.exists(PROCESSED_IDS_FILE): return set()
    with open(PROCESSED_IDS_FILE, "r") as f: return set(line.strip() for line in f if line.strip())

def save_processed_id(uid: str):
    with open(PROCESSED_IDS_FILE, "a") as f: f.write(uid + "\n")

# =====================================================================
# SYSTEM LIFE-CYCLE ORCHESTRATION PIPELINE
# =====================================================================
def process_email(msg, uid_str: str):
    sender_name, sender_addr = email.utils.parseaddr(msg.get("From", ""))
    raw_subj, enc = decode_header(msg.get("Subject", "No Subject"))[0]
    if isinstance(raw_subj, bytes): raw_subj = raw_subj.decode(enc or "utf-8", errors="ignore")
    reply_subj = raw_subj if raw_subj.lower().startswith("re:") else f"Re: {raw_subj}"

    ui_print(f"==========================================")
    ui_print(f"📧 EMAIL CONTENT INGESTED (UID {uid_str})")
    ui_print(f"==========================================")

    body, images = parse_email_body(msg)
    ocr          = extract_ocr(images) if images else ""
    full_text    = f"{body}\n{ocr}".strip()
    
    if not full_text:
        ui_print("⚠️ Inbound message text array empty — pipeline skipped.")
        return

    if should_skip_sender(sender_addr):
        ui_print(f"⛔ Skipping circuit processing logic loop for automated target address: {sender_addr}")
        return

    ui_print(full_text)
    ui_print(f"==========================================\n")

    ui_print("Processing deep text scanning via NVIDIA Llama 3.1 8B...")
    t0 = time.time()
    language, tone = detect_language_and_tone(full_text)
    ui_print(f"   👉 Detected Language: **{language}**")
    ui_print(f"   👉 Evaluated Email Tone: **{tone}** ({time.time()-t0:.2f}s)\n")

    ui_print("Generating English Translation...")
    t0 = time.time()
    english_text = translate_to_english(full_text, language)
    if not english_text: return
    ui_print(f"   👉 English Translation:\n   \"{english_text}\" ({time.time()-t0:.2f}s)\n-----------------------")

    ui_print("🤖 AI Drafting Persona-Based Support Reply (in English)...")
    t0 = time.time()
    english_reply = draft_english_reply(english_text, tone)
    if not english_reply: return
    ui_print(f"   👉 Generated English Draft:\n   {english_reply} ({time.time()-t0:.2f}s)\n-----------------------")

    ui_print(f"🔄 Translating response framework → {language} & initializing QA audit matrices...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_trans = executor.submit(translate_to_native, english_reply, language, tone)
        f_qa    = executor.submit(run_qa_audit, english_reply, english_reply, tone, language)
        native_reply = f_trans.result()
        qa_score, qa_analysis = f_qa.result()

    if not native_reply: return
    ui_print(f"   👉 Final Customer-Facing Response:\n   \"{native_reply}\"")
    ui_print(f"   👉 QA Tone Evaluation Compliance Audit Output: Score {qa_score}/5 — {qa_analysis} ({time.time()-t0:.2f}s)\n-----------------------")

    if qa_score < 3:
        ui_print(f"⚠️ QA Score threshold alert. Re-running transformation prompt arrays...")
        improved = translate_to_native(english_reply, language, tone + " (Enforce strict politeness constraints)")
        if improved: native_reply = improved

    send_reply(sender_addr, reply_subj, native_reply)

# ═══════════════════════════ STREAMLIT INTERFACE FRAMEWORK ════════════════════════════════════
st.set_page_config(layout="wide")
st.title("📬 Intelligent Multilingual Support Middleware Engine")
st.markdown("---")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.header("⚙️ Control Node")
    st.info(f"📧 **Active Mailbox:** `{EMAIL_USER}`")
    
    if not st.session_state.is_running:
        if st.button("🚀 Launch Background Engine", type="primary", use_container_width=True):
            st.session_state.is_running = True
            st.rerun()
    else:
        if st.button("🛑 Terminate Background Engine", type="secondary", use_container_width=True):
            st.session_state.is_running = False
            st.rerun()
            
    st.markdown("---")
    st.subheader("📊 Engine Matrix")
    st.text(f"Processor: {FAST_MODEL}")
    st.text(f"Reasoning: {STRONG_MODEL}")

with col_right:
    st.header("🖥️ Live Telemetry Streams")
    
    if st.button("🗑️ Clear Log Console History", use_container_width=True):
        st.session_state.ui_logs = ["Buffer memory purged. Standing by..."]
        st.rerun()

    log_placeholder = st.empty()
    log_placeholder.text_area(
        label="Operational Traces",
        value="\n".join(st.session_state.ui_logs),
        height=580,
        disabled=True
    )

# ═══════════════════════════ MAIN RUNTIME POLLING LOOP ════════════════════════════════════
if st.session_state.is_running:
    processed_ids = load_processed_ids()
    mail = None
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(EMAIL_USER, EMAIL_PASS)
        mail.select("inbox")
        status, data = mail.uid("search", None, "UNSEEN")
        
        if status == "OK" and data[0]:
            unread_uids = data[0].split()
            new_uids    = [u for u in unread_uids if u.decode() not in processed_ids]
            
            if new_uids:
                for uid_bytes in new_uids:
                    uid_str = uid_bytes.decode()
                    _, msg_data = mail.uid("fetch", uid_bytes, "(RFC822)")
                    process_email(email.message_from_bytes(msg_data[0][1]), uid_str)
                    processed_ids.add(uid_str)
                    save_processed_id(uid_str)
            else:
                ui_print("Scanning inbox target workspace... No unseen messages located.")
        else:
            ui_print("Scanning inbox target workspace... No unseen messages located.")
        mail.logout()
    except Exception as e:
        ui_print(f"❌ Connection error: {e}")
        if mail:
            try: mail.logout()
            except: pass

    time.sleep(POLL_INTERVAL_SECONDS)
    st.rerun()
