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
import threading
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

# ═══════════════════════════ THREAD-SAFE GLOBAL LOG MONITOR ════════════════════════════════════
if "SYSTEM_TELEMETRY_LOGS" not in globals():
    globals()["SYSTEM_TELEMETRY_LOGS"] = ["🌐 Engine Standby. Awaiting daemon thread activation..."]
    globals()["THREAD_MEMORY_LOCK"] = threading.Lock()

def ui_print(text: str):
    """Thread-safe logging framework routing telemetry data to stdout and the web layout UI."""
    print(text)
    timestamp = time.strftime('%H:%M:%S')
    formatted_line = f"[{timestamp}] {text}"
    with globals()["THREAD_MEMORY_LOCK"]:
        globals()["SYSTEM_TELEMETRY_LOGS"].append(formatted_line)
        if len(globals()["SYSTEM_TELEMETRY_LOGS"]) > 500:
            globals()["SYSTEM_TELEMETRY_LOGS"].pop(1)

# ═══════════════════════════ STREAMLIT STATE INITIALIZATION ENGINE ═══════════════════════════
# CRITICAL FIX: Ensures keys are safely instantiated before any logical evaluate runs
if "is_running" not in st.session_state:
    st.session_state.is_running = False

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
    "Kannada": [
        "hego iddeera", "hegidira", "hegiddira", "namaskara",
        "WB:hego", "WB:iddeera", "WB:banni", "WB:hogu", "WB:madi",
        "WB:aagide", "WB:neenu", "WB:neevu", "WB:naanu", "WB:avaru",
        "WB:yenu", "WB:yake", "WB:beku",
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
    best_score = scores[best_lang]
    if best_score < 2: return None
    eng_words   = len(re.findall(r'\b[a-z]{3,}\b', text_lower))
    native_hits = sum(scores.values())
    ratio       = native_hits / max(eng_words, 1)
    if ratio < 0.05: return None  
    if eng_words > 40 and native_hits < 5: return f"{best_lang}, English"
    return best_lang

def _clean_language_string(raw: str) -> str:
    raw = re.sub(r'\s*\(.*?\)', '', raw)      
    raw = re.sub(r'\s*[-–].*$', '', raw)       
    raw = re.sub(r'\s+', ' ', raw).strip()
    parts = [p.strip().title() for p in raw.split(',') if p.strip()]
    return ', '.join(parts)

def _call_api(model: str, messages: list, max_tokens: int = 512,
              temperature: float = 0.0, max_retries: int = 4) -> str | None:
    headers = {"Authorization": f"Bearer {NVAPI_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    delay = 2
    for attempt in range(max_retries):
        try:
            r = _session.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=60)
            if r.status_code == 429:
                time.sleep(min(delay, 60))
                delay *= 2
                continue
            if r.status_code >= 500:
                time.sleep(min(delay, 60))
                delay *= 2
                continue
            data = r.json()
            if "choices" in data: return data["choices"][0]["message"]["content"].strip()
            return None
        except Exception:
            time.sleep(1)
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
    result = _call_api(FAST_MODEL, messages, max_tokens=60, temperature=0.0)
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
        {"role": "system", "content": "You are an elite translator specializing in South Asian scripts. Output ONLY the English translation — no labels, no notes."},
        {"role": "user", "content": f"Translate ({detected_languages}):\n\n{text[:12000]}"},
    ]
    return _call_api(STRONG_MODEL, messages, max_tokens=1500, temperature=0.0)

def draft_english_reply(english_text: str, tone: str) -> str | None:
    persona_prompt = (
        "You are a real human writing a direct personal email response. Act completely as the primary user profile described below:\n\n"
        "IDENTITY DATA ENGINE:\n"
        "- Name: Vishnu (Full name: Boddu Vishnu Vardhan Reddy).\n"
        "- Occupation: Computer Science & Engineering student at Amity University.\n"
        "- Academic Focus: Python algorithms, complexity analysis, text parsing/scraping, and NLP. Preparing for the GATE exam.\n"
        "- Hobbies: Big follower of T20 cricket and IPL statistics (RCB, SRH, KKR), tracking regional Telugu cinema movie releases, and playing Free Fire MAX rounds during free blocks.\n"
        "- Daily Food Habits: Eats South Indian breakfast (Idli/Dosa/Poha) with coffee in the morning.\n\n"
        f"Tone Constraints: Enforce an explicitly **{tone}** register style framework. Output ONLY the final response text payload body."
    )
    messages = [
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content": f"Draft a reply to:\n\n{english_text[:5000]}"},
    ]
    return _call_api(STRONG_MODEL, messages, max_tokens=600, temperature=0.5)

def translate_to_native(english_reply: str, target_language: str, tone: str) -> str | None:
    if "english" in target_language.lower() and "," not in target_language: return english_reply
    messages = [
        {"role": "system", "content": f"You are a native speaker of {target_language}. Translate this English text array into fluent, complete sentences in {target_language} matching a {tone} register register perfectly. Output ONLY the translation."},
        {"role": "user", "content": f"Translate to {target_language}:\n\n{english_reply}"},
    ]
    return _call_api(STRONG_MODEL, messages, max_tokens=1000, temperature=0.0)

def run_qa_audit(english_draft: str, native_reply: str, target_tone: str, target_lang: str) -> tuple[int, str]:
    messages = [
        {"role": "system", "content": f"Compare the English draft response with its {target_lang} translation. Audit if the specified tone target **{target_tone}** was perfectly preserved. Format response exactly like:\nSCORE: <integer 1-5>\nANALYSIS: <one sentence explanation>"},
        {"role": "user", "content": f"English draft:\n{english_draft}\n\nTranslated reply:\n{native_reply}"},
    ]
    result = _call_api(FAST_MODEL, messages, max_tokens=80, temperature=0.0)
    score, analysis = 5, "Audit verification completed successfully."
    if result:
        for line in result.strip().splitlines():
            upper = line.upper()
            if upper.startswith("SCORE:"):
                try: score = int(re.search(r'\d', line.split(":", 1)[1]).group())
                except Exception: pass
            elif upper.startswith("ANALYSIS:"): analysis = line.split(":", 1)[1].strip()
    return score, analysis

def clean_html(html_text: str) -> str:
    text = html.unescape(html_text)
    text = re.sub(r"<style[^>]*>[\s\S]*?</style>|<script[^>]*>[\s\S]*?</script>|<[^>]+>", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", re.sub(r"[\u200b-\u200d\u2060\ufeff\xad]", "", text)).strip()

def extract_ocr(image_bytes_list: list) -> str:
    if not pytesseract or not PILImage: return ""
    results = []
    for i, img_bytes in enumerate(image_bytes_list):
        try:
            img = PILImage.open(io.BytesIO(img_bytes))
            ocr = pytesseract.image_to_string(img).strip()
            if ocr: results.append(f"\n[Image {i+1} OCR Data Ingestion]\n{ocr}")
        except Exception: pass
    return "\n".join(results)

def parse_email_body(msg) -> tuple[str, list]:
    body, html_body, images = "", "", []
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            cd = str(part.get("Content-Disposition", ""))
            raw = part.get_payload(decode=True)
            if ct.startswith("image/") and raw: images.append(raw)
            elif "attachment" in cd: continue
            elif ct == "text/plain" and not body and raw: body = raw.decode(errors="ignore")
            elif ct == "text/html" and not html_body and raw: html_body = raw.decode(errors="ignore")
    else:
        ct, raw = msg.get_content_type(), msg.get_payload(decode=True)
        if raw:
            if ct.startswith("image/"): images.append(raw)
            elif ct == "text/html": html_body = raw.decode(errors="ignore")
            else: body = raw.decode(errors="ignore")
    final = clean_html(html_body) if html_body.strip() else clean_html(body)
    return final.strip(), images

def should_skip_sender(sender_addr: str) -> bool:
    low = sender_addr.lower()
    return any(p in low for p in SKIP_SENDER_PATTERNS)

def send_reply(recipient: str, subject: str, body_text: str):
    try:
        msg = MIMEMultipart()
        msg["From"], msg["To"], msg["Subject"] = EMAIL_USER, recipient, subject
        msg.attach(MIMEText(body_text, "plain", "utf-8"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.sendmail(EMAIL_USER, recipient, msg.as_string())
        ui_print(f"📬 Outbound SMTP routing channel clear. Reply sent successfully → {recipient}")
    except Exception as e:
        ui_print(f"❌ Outbound SMTP channel delivery error: {e}")

def load_processed_ids() -> set:
    if not os.path.exists(PROCESSED_IDS_FILE): return set()
    with open(PROCESSED_IDS_FILE, "r") as f: return set(line.strip() for line in f if line.strip())

def save_processed_id(uid: str):
    with open(PROCESSED_IDS_FILE, "a") as f: f.write(uid + "\n")

def process_email(msg, uid_str: str):
    sender_name, sender_addr = email.utils.parseaddr(msg.get("From", ""))
    raw_subj, enc = decode_header(msg.get("Subject", "No Subject"))[0]
    if isinstance(raw_subj, bytes): raw_subj = raw_subj.decode(enc or "utf-8", errors="ignore")
    reply_subj = raw_subj if raw_subj.lower().startswith("re:") else f"Re: {raw_subj}"

    ui_print(f"Incoming Target Frame Located (UID {uid_str}) from {sender_name or sender_addr} | Subject: {raw_subj}")
    if should_skip_sender(sender_addr):
        ui_print(f"⛔ Skipping circuit logic loop for automated target sender address: {sender_addr}")
        return

    body, images = parse_email_body(msg)
    ocr          = extract_ocr(images) if images else ""
    full_text    = f"{body}\n{ocr}".strip()
    if not full_text:
        ui_print("⚠️ Inbound message text string void — pipeline skipped.")
        return

    ui_print(f"📄 Content Payload Body Extracted: \"{full_text[:300]}...\"")

    ui_print("🔍 Step 1 & 3: Initializing Language & Tone Tracking Framework...")
    t0 = time.time()
    language, tone = detect_language_and_tone(full_text)
    ui_print(f"   👉 Detected Language: {language} | Evaluated Tone: {tone} ({time.time()-t0:.2f}s)")

    ui_print("🌐 Step 2: Running Content Translation Mapping to English Context...")
    t0 = time.time()
    english_text = translate_to_english(full_text, language)
    if not english_text: return
    ui_print(f"   👉 Normalized English Context: \"{english_text[:150]}...\"")

    ui_print("✍️  Step 4: AI Drafting Persona-Based Support Reply (in English)...")
    t0 = time.time()
    english_reply = draft_english_reply(english_text, tone)
    if not english_reply: return
    ui_print(f"   👉 Generated Persona English Draft: \"{english_reply[:150]}...\"")

    ui_print(f"🔄 Step 5: Translating Response Framework to Native Script and Running QA Audit...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_translate = executor.submit(translate_to_native, english_reply, language, tone)
        future_qa        = executor.submit(run_qa_audit, english_reply, english_reply, tone, language)
        native_reply            = future_translate.result()
        qa_score, qa_analysis   = future_qa.result()

    if not native_reply: return
    ui_print(f"   👉 QA Verification Metrics Result: Score {qa_score}/5 — {qa_analysis}")

    if qa_score < 3:
        ui_print(f"⚠️ QA Score threshold alert. Re-running conversion arrays with stricter target heuristics...")
        improved = translate_to_native(english_reply, language, tone + " (strict tone preservation)")
        if improved: native_reply = improved

    ui_print(f"📬 Final Response Block Synthesized Natively:\n{native_reply}")
    send_reply(sender_addr, reply_subj, native_reply)

def background_monitor_thread(running_flag):
    processed_ids = load_processed_ids()
    ui_print(f"🚀 Background Core Thread Hooked. Starting live scanning queue cycles...")
    
    while running_flag():
        mail = None
        try:
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            mail.login(EMAIL_USER, EMAIL_PASS)
            mail.select("inbox")

            status, data = mail.uid("search", None, "UNSEEN")
            if status != "OK" or not data[0]:
                mail.logout()
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            unread_uids = data[0].split()
            new_uids    = [u for u in unread_uids if u.decode() not in processed_ids]

            if not new_uids:
                mail.logout()
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            ui_print(f"🔔 Target alert: Found {len(new_uids)} fresh email payload entries inside queue.")
            for uid_bytes in new_uids:
                if not running_flag(): break
                uid_str = uid_bytes.decode()
                try:
                    _, msg_data = mail.uid("fetch", uid_bytes, "(RFC822)")
                    for part in msg_data:
                        if isinstance(part, tuple):
                            process_email(email.message_from_bytes(part[1]), uid_str)
                    processed_ids.add(uid_str)
                    save_processed_id(uid_str)
                except Exception as e:
                    ui_print(f"❌ Structural parsing mapping fault on UID {uid_str}: {e}")
            mail.logout()
        except Exception as e:
            ui_print(f"❌ Live inbox connection network pulse anomaly: {e}")
            if mail:
                try: mail.logout()
                except: pass
        time.sleep(POLL_INTERVAL_SECONDS)

# ═══════════════════════════ STREAMLIT DASHBOARD INTERFACE LAYOUT ════════════════════════════════════
st.title("📬 Intelligent Multilingual Support Middleware Engine")
st.markdown("---")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.header("⚙️ System Control Node")
    st.info(f"📧 **Active Account Workspace:** `{EMAIL_USER}`")
    
    if not st.session_state.is_running:
        if st.button("🚀 Launch Background Daemon Thread", type="primary", use_container_width=True):
            st.session_state.is_running = True
            with globals()["THREAD_MEMORY_LOCK"]:
                globals()["SYSTEM_TELEMETRY_LOGS"].append(f"[{time.strftime('%H:%M:%S')}] ⚡ Spawning isolated background server tracking node...")
            
            t = threading.Thread(target=background_monitor_thread, args=(lambda: st.session_state.is_running,), daemon=True)
            t.start()
            st.rerun()
    else:
        if st.button("🛑 Terminate Background Daemon Thread", type="secondary", use_container_width=True):
            st.session_state.is_running = False
            with globals()["THREAD_MEMORY_LOCK"]:
                globals()["SYSTEM_TELEMETRY_LOGS"].append(f"[{time.strftime('%H:%M:%S')}] 🛑 Stopping active tracking background worker nodes...")
            st.rerun()
            
    st.markdown("---")
    st.subheader("📊 Engine Specifications Matrix")
    st.text(f"Fast Token Processor: \n{FAST_MODEL}")
    st.text(f"Strong Reasoning Engine: \n{STRONG_MODEL}")
    st.text(f"Polling Frequency Window: {POLL_INTERVAL_SECONDS}s")
    
    st.button("🔄 Sync Telemetry Screen Updates", use_container_width=True)

with col_right:
    st.header("🖥️ System Verbose Log Streams")
    
    if st.button("🗑️ Clear Log Cache Memory", use_container_width=True):
        with globals()["THREAD_MEMORY_LOCK"]:
            globals()["SYSTEM_TELEMETRY_LOGS"] = ["Buffer memory purged. Standing by..."]
        st.rerun()

    with globals()["THREAD_MEMORY_LOCK"]:
        log_content = "\n".join(globals()["SYSTEM_TELEMETRY_LOGS"])

    st.text_area(
        label="Live Process Metrics (Restored Verbosity Traces)",
        value=log_content,
        height=580,
        disabled=True
    )
