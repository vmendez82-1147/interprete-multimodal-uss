import os
import re
import tempfile
import urllib.parse
import numpy as np
import scipy.io.wavfile as wavfile
import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

st.set_page_config(page_title="Intérprete Multimodal USS", page_icon="🗣️", layout="wide")

device = -1
torch_dtype = torch.float32

# Modelos adaptados para 1GB RAM (CPU Cloud)
OPUS_MODELS = {
    ("Español", "Inglés"): "Helsinki-NLP/opus-mt-es-en",
    ("Inglés", "Español"): "Helsinki-NLP/opus-mt-en-es",
    ("Español", "Francés"): "Helsinki-NLP/opus-mt-es-fr",
    ("Francés", "Español"): "Helsinki-NLP/opus-mt-fr-es",
    ("Español", "Portugués"): "Helsinki-NLP/opus-mt-es-pt",
    ("Portugués", "Español"): "Helsinki-NLP/opus-mt-ROMANCE-en",
    ("Español", "Alemán"): "Helsinki-NLP/opus-mt-es-de",
    ("Alemán", "Español"): "Helsinki-NLP/opus-mt-de-es",
    ("Español", "Italiano"): "Helsinki-NLP/opus-mt-es-it",
    ("Italiano", "Español"): "Helsinki-NLP/opus-mt-it-es",
}

LANGUAGES = {
    "Español": {"whisper": "spanish", "tts": "facebook/mms-tts-spa"},
    "Inglés": {"whisper": "english", "tts": "facebook/mms-tts-eng"},
    "Portugués": {"whisper": "portuguese", "tts": "facebook/mms-tts-por"},
    "Francés": {"whisper": "french", "tts": "facebook/mms-tts-fra"},
    "Alemán": {"whisper": "german", "tts": "facebook/mms-tts-deu"},
    "Italiano": {"whisper": "italian", "tts": "facebook/mms-tts-ita"},
}

CHILEAN_SLANG_MAP = {
    r"\bech(é|e|arse|ó|aron)\s+el\s+ramo\b": "reprobar la asignatura",
    r"\bramo\b": "asignatura",
    r"\bramos\b": "asignaturas",
    r"\bprofe\b": "profesor",
    r"\bpega\b": "trabajo",
    r"\bpegas\b": "trabajos",
    r"\bal\s+tiro\b": "inmediatamente",
    r"\bcachai\b": "¿entiendes?",
    r"\bcachar\b": "entender",
    r"\bcacho\b": "entiendo",
    r"\bla\s+raja\b": "excelente",
    r"\bhacer\s+una\s+vaquita\b": "reunir dinero entre todos",
    r"\bestar\s+colgado\b": "no entender nada",
}

def canonicalize_chilean_spanish(text: str):
    if not text:
        return "", []
    normalized, detected = text, []
    for pattern, rep in CHILEAN_SLANG_MAP.items():
        m = re.search(pattern, normalized, flags=re.IGNORECASE)
        if m:
            detected.append(f"'{m.group(0)}' ➔ '{rep}'")
            normalized = re.sub(pattern, rep, normalized, flags=re.IGNORECASE)
    return normalized, detected

@st.cache_resource
def get_asr():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=device)

@st.cache_resource
def get_translator(src_lang, tgt_lang):
    pair = (src_lang, tgt_lang)
    model_id = OPUS_MODELS.get(pair, "Helsinki-NLP/opus-mt-es-en")
    tok = AutoTokenizer.from_pretrained(model_id)
    mod = AutoModelForSeq2SeqLM.from_pretrained(model_id, low_cpu_mem_usage=True)
    mod.eval()
    return tok, mod

@st.cache_resource
def get_tts(lang_name):
    model_id = LANGUAGES[lang_name]["tts"]
    return pipeline("text-to-speech", model=model_id, device=device)

@st.cache_resource
def get_ocr():
    import easyocr
    return easyocr.Reader(["es", "en"], gpu=False)

def synthesize_speech(text, target_lang="Español"):
    try:
        tts_pipe = get_tts(target_lang)
        clean_text = text[:250].strip() or "Texto no disponible."
        tts_out = tts_pipe(clean_text)
        sr = int(tts_out["sampling_rate"])
        raw_audio = tts_out["audio"]
        if isinstance(raw_audio, torch.Tensor):
            raw_audio = raw_audio.squeeze().cpu().numpy()
        audio_data = np.array(raw_audio, dtype=np.float32)
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wavfile.write(temp_wav.name, sr, (audio_data * 32767).astype(np.int16))
        return temp_wav.name
    except Exception as e:
        return None

def translate_text(text, src_lang, tgt_lang):
    if not text or not text.strip():
        return ""
    if src_lang == tgt_lang:
        return text.strip()
    tok, mod = get_translator(src_lang, tgt_lang)
    inp = tok(text.strip(), return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        out = mod.generate(**inp, max_new_tokens=100)
    return tok.batch_decode(out, skip_special_tokens=True)[0]

# Interfaz
st.title("🗣️ Intérprete Multimodal & Multilingüe USS")
st.caption("Magíster en Data Science • Universidad San Sebastián")

tab1, tab2 = st.tabs(["🌐 Intérprete Universal", "🤖 Asistente del Proyecto"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Entrada y Configuración")
        mode = st.radio("Fuente de entrada", ["🎙️ Audio", "📄 Documento (PDF/DOCX)", "📷 Imagen/Foto (OCR)"])
        src_lang = st.selectbox("Idioma Origen", list(LANGUAGES.keys()), index=0)
        tgt_lang = st.selectbox("Idioma Destino", list(LANGUAGES.keys()), index=1)
        use_slang = st.checkbox("Normalizar modismos chilenos (Solo origen Español)", value=True)

        audio_file, doc_file, img_file = None, None, None
        if mode == "🎙️ Audio":
            audio_file = st.file_uploader("Sube un audio (WAV, MP3)", type=["wav", "mp3"])
        elif mode == "📄 Documento (PDF/DOCX)":
            doc_file = st.file_uploader("Sube un PDF o DOCX", type=["pdf", "docx"])
        else:
            img_file = st.file_uploader("Sube una foto legible", type=["png", "jpg", "jpeg"])

        btn_run = st.button("✨ Procesar y Traducir ✨", type="primary")

    with col2:
        st.subheader("2. Resultados")
        if btn_run:
            with st.spinner("Procesando entrada con modelos fundacionales..."):
                raw_text = ""
                if mode == "🎙️ Audio" and audio_file:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    tfile.write(audio_file.read())
                    asr = get_asr()
                    lang_code = LANGUAGES[src_lang]["whisper"]
                    res = asr(tfile.name, generate_kwargs={"language": lang_code})
                    raw_text = res["text"].strip()
                elif mode == "📄 Documento (PDF/DOCX)" and doc_file:
                    ext = os.path.splitext(doc_file.name)[-1].lower()
                    if ext == ".pdf":
                        import pypdf
                        reader = pypdf.PdfReader(doc_file)
                        raw_text = " ".join([p.extract_text() for p in reader.pages[:2] if p.extract_text()])
                    elif ext == ".docx":
                        import docx
                        doc = docx.Document(doc_file)
                        raw_text = " ".join([p.text for p in doc.paragraphs if p.text])
                elif mode == "📷 Imagen/Foto (OCR)" and img_file:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    tfile.write(img_file.read())
                    ocr = get_ocr()
                    raw_text = " ".join(ocr.readtext(tfile.name, detail=0)).strip()

                if raw_text:
                    st.text_area("Texto Detectado:", value=raw_text, height=90)
                    clean_text = raw_text
                    if src_lang == "Español" and use_slang:
                        clean_text, detected = canonicalize_chilean_spanish(raw_text)
                        if detected:
                            st.info("💡 Modismos normalizados: " + ", ".join(detected))

                    translation = translate_text(clean_text, src_lang, tgt_lang)
                    st.success("**Traducción Final:** " + translation)

                    audio_path = synthesize_speech(translation, tgt_lang)
                    if audio_path:
                        st.audio(audio_path, format="audio/wav")

                    wsp_msg = f"🎙️ Traducción USS ({src_lang} ➔ {tgt_lang}):\nOriginal: {raw_text}\nTraducción: {translation}"
                    url_wsp = f"https://api.whatsapp.com/send?text={urllib.parse.quote(wsp_msg)}"
                    st.markdown(f"[📲 Compartir por WhatsApp]({url_wsp})")
                else:
                    st.warning("No se pudo extraer texto o no adjuntaste archivo.")

with tab2:
    st.subheader("Asistente del Proyecto")
    st.markdown("""
    - **Capacidades integradas:** Speech-to-Text (Whisper), OCR (EasyOCR), Traducción (OPUS / NLLB) y Síntesis de voz (MMS-TTS).
    - **Filtro de jerga:** Normaliza modismos chilenos para evitar errores semánticos en el traductor.
    - **Optimización Cloud:** Pipeline configurado con baja latencia y carga diferida.
    """)
