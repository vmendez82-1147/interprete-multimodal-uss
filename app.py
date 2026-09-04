import os
import re
import tempfile
import urllib.parse
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

st.set_page_config(page_title="Intérprete Multimodal USS", page_icon="🗣️", layout="wide")

device = -1
device_str = "cpu"
torch_dtype = torch.float32

LANGUAGES = {
    "Español": {"nllb": "spa_Latn", "whisper": "spanish", "tts_model": "facebook/mms-tts-spa"},
    "Inglés": {"nllb": "eng_Latn", "whisper": "english", "tts_model": "facebook/mms-tts-eng"},
    "Portugués": {"nllb": "por_Latn", "whisper": "portuguese", "tts_model": "facebook/mms-tts-por"},
    "Francés": {"nllb": "fra_Latn", "whisper": "french", "tts_model": "facebook/mms-tts-fra"},
    "Alemán": {"nllb": "deu_Latn", "whisper": "german", "tts_model": "facebook/mms-tts-deu"},
    "Italiano": {"nllb": "ita_Latn", "whisper": "italian", "tts_model": "facebook/mms-tts-ita"},
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
def load_base_models():
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device, torch_dtype=torch_dtype)
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    model.eval()
    return asr, tokenizer, model

asr, tokenizer_nllb, model_nllb = load_base_models()

@st.cache_resource
def get_tts(lang_name):
    model_id = LANGUAGES[lang_name]["tts_model"]
    return pipeline("text-to-speech", model=model_id, device=device, torch_dtype=torch_dtype)

@st.cache_resource
def get_ocr():
    import easyocr
    return easyocr.Reader(["es", "en", "pt", "fr", "de", "it"], gpu=False)

def synthesize_speech(text, target_lang="Español"):
    tts_pipe = get_tts(target_lang)
    clean_text = text[:300].strip() or "Texto no disponible."
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

def translate_text(text, src_lang_name, tgt_lang_name):
    if not text or not text.strip():
        return ""
    src_code = LANGUAGES[src_lang_name]["nllb"]
    tgt_code = LANGUAGES[tgt_lang_name]["nllb"]
    tokenizer_nllb.src_lang = src_code
    inputs = tokenizer_nllb(text.strip(), return_tensors="pt", truncation=True, max_length=256)
    with torch.inference_mode():
        outputs = model_nllb.generate(
            **inputs,
            forced_bos_token_id=tokenizer_nllb.convert_tokens_to_ids(tgt_code),
            num_beams=1,
            max_new_tokens=128,
            use_cache=True,
        )
    return tokenizer_nllb.batch_decode(outputs, skip_special_tokens=True)[0]

st.title("🗣️ Intérprete Multimodal & Multilingüe USS")
tab1, tab2 = st.tabs(["🌐 Intérprete Universal", "🤖 Asistente Guía"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Entrada y Configuración")
        mode = st.radio("Fuente de entrada", ["🎙️ Audio", "📄 Documento (PDF/DOCX)", "📷 Imagen/Foto (OCR)"])
        src_lang = st.selectbox("Idioma Origen", list(LANGUAGES.keys()), index=0)
        tgt_lang = st.selectbox("Idioma Destino", list(LANGUAGES.keys()), index=1)
        use_slang = st.checkbox("Normalizar modismos chilenos", value=True)

        audio_file, doc_file, img_file = None, None, None
        if mode == "🎙️ Audio":
            audio_file = st.file_uploader("Sube un archivo de audio (WAV, MP3)", type=["wav", "mp3"])
        elif mode == "📄 Documento (PDF/DOCX)":
            doc_file = st.file_uploader("Sube un archivo PDF o DOCX", type=["pdf", "docx"])
        else:
            img_file = st.file_uploader("Sube una imagen con texto", type=["png", "jpg", "jpeg"])

        btn_run = st.button("✨ Procesar y Traducir ✨", type="primary")

    with col2:
        st.subheader("2. Resultados")
        if btn_run:
            raw_text = ""
            if mode == "🎙️ Audio" and audio_file:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(audio_file.read())
                lang_whisper = LANGUAGES[src_lang]["whisper"]
                raw_text = asr(tfile.name, generate_kwargs={"language": lang_whisper})["text"].strip()
            elif mode == "📄 Documento (PDF/DOCX)" and doc_file:
                ext = os.path.splitext(doc_file.name)[-1].lower()
                if ext == ".pdf":
                    import pypdf
                    reader = pypdf.PdfReader(doc_file)
                    raw_text = " ".join([p.extract_text() for p in reader.pages[:3] if p.extract_text()])
                elif ext == ".docx":
                    import docx
                    doc = docx.Document(doc_file)
                    raw_text = " ".join([p.text for p in doc.paragraphs if p.text])
            elif mode == "📷 Imagen/Foto (OCR)" and img_file:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tfile.write(img_file.read())
                reader = get_ocr()
                raw_text = " ".join(reader.readtext(tfile.name, detail=0)).strip()

            if raw_text:
                st.write("**Texto Detectado:**", raw_text)
                clean_text = raw_text
                if src_lang == "Español" and use_slang:
                    clean_text, detected = canonicalize_chilean_spanish(raw_text)
                    if detected:
                        st.info("Modismos detectados: " + ", ".join(detected))

                translation = translate_text(clean_text, src_lang, tgt_lang)
                st.success("**Traducción Final:** " + translation)

                audio_path = synthesize_speech(translation, tgt_lang)
                st.audio(audio_path, format="audio/wav")

                wsp_msg = f"🎙️ Traducción USS ({src_lang} ➔ {tgt_lang}):\nOriginal: {raw_text}\nTraducción: {translation}"
                url_wsp = f"https://api.whatsapp.com/send?text={urllib.parse.quote(wsp_msg)}"
                st.markdown(f"[📲 Compartir en WhatsApp]({url_wsp})")
            else:
                st.warning("No se pudo procesar la entrada o no se cargó ningún archivo.")

with tab2:
    st.subheader("Asistente del Proyecto")
    user_q = st.text_input("Haz una pregunta sobre el sistema:")
    if user_q:
        q = user_q.lower()
        if "idioma" in q:
            st.write("Idiomas disponibles: Español, Inglés, Portugués, Francés, Alemán e Italiano.")
        elif "modismo" in q:
            st.write("Normaliza expresiones coloquiales chilenas antes de pasar el texto al modelo de traducción.")
        elif "modelo" in q:
            st.write("Arquitectura: Whisper Small (ASR), EasyOCR (Visión), NLLB-200 (Traducción) y MMS-TTS (Voz).")
        else:
            st.write("Puedes consultar sobre idiomas soportados, modelos de IA o el filtro de modismos chilenos.")