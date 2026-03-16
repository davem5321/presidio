"""Streamlit UI for Presidio Analyzer - Document & Folder Processing."""

import os
import json
import tempfile
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Set

import io

import dotenv
import streamlit as st
import pandas as pd
import fitz  # pymupdf
from docx import Document
from presidio_analyzer import AnalyzerEngine, RecognizerResult, PatternRecognizer, Pattern, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider

dotenv.load_dotenv()


def _load_allowed_extensions() -> Set[str]:
    """Load allowed file extensions from .env or use defaults."""
    env_val = os.getenv("ALLOWED_EXTENSIONS", "")
    if env_val.strip():
        return {ext.strip().lower() for ext in env_val.split(",") if ext.strip()}
    return {".txt", ".md", ".csv", ".log", ".json", ".xml", ".html", ".yml", ".yaml", ".pdf", ".docx"}


ALLOWED_EXTENSIONS = _load_allowed_extensions()


def _load_excluded_filenames() -> Set[str]:
    """Load excluded filenames from .env."""
    env_val = os.getenv("EXCLUDED_FILENAMES", "")
    if env_val.strip():
        return {name.strip() for name in env_val.split(",") if name.strip()}
    return set()


EXCLUDED_FILENAMES = _load_excluded_filenames()


def _load_excluded_folders() -> Set[str]:
    """Load excluded folder names from .env."""
    env_val = os.getenv("EXCLUDED_FOLDERS", "")
    if env_val.strip():
        return {name.strip() for name in env_val.split(",") if name.strip()}
    return set()


EXCLUDED_FOLDERS = _load_excluded_folders()


st.set_page_config(
    page_title="Presidio Analyzer - Document Scanner",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "https://microsoft.github.io/presidio/"},
)


def _create_company_recognizer() -> PatternRecognizer:
    """Create a pattern recognizer for company names based on common suffixes."""
    company_patterns = [
        Pattern(
            name="company_suffix",
            regex=r"\b(?:[A-Z][A-Za-z&\'\-]+(?:\s+[A-Z][A-Za-z&\'\-]+){0,4})\s+(?:Inc\.|Incorporated|Ltd\.|Limited|Corp\.|Corporation|LLC|LLP|L\.L\.C\.|PLC|plc|Co\.|GmbH|S\.A\.|N\.V\.|B\.V\.)\b",
            score=0.7,
        ),
    ]
    return PatternRecognizer(
        supported_entity="ORGANIZATION",
        name="CompanyNameRecognizer",
        patterns=company_patterns,
        supported_language="en",
    )


NER_MODELS = {
    "spaCy / en_core_web_lg (fast, default)": {
        "engine": "spacy",
        "model": "en_core_web_lg",
    },
    "spaCy / en_core_web_trf (transformer-based, most accurate)": {
        "engine": "spacy",
        "model": "en_core_web_trf",
    },
    "spaCy / en_core_web_sm (fastest, less accurate)": {
        "engine": "spacy",
        "model": "en_core_web_sm",
    },
    "HuggingFace / obi/deid_roberta_i2b2 (best for deidentification)": {
        "engine": "transformers",
        "model": "obi/deid_roberta_i2b2",
    },
    "HuggingFace / StanfordAIMI/stanford-deidentifier-base": {
        "engine": "transformers",
        "model": "StanfordAIMI/stanford-deidentifier-base",
    },
}

SPACY_ENTITY_MAPPING = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "NORP": "NRP",
    "FAC": "LOCATION",
    "LOC": "LOCATION",
    "GPE": "LOCATION",
    "LOCATION": "LOCATION",
    "ORG": "ORGANIZATION",
    "ORGANIZATION": "ORGANIZATION",
    "DATE": "DATE_TIME",
    "TIME": "DATE_TIME",
}

TRANSFORMERS_ENTITY_MAPPING = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "LOC": "LOCATION",
    "LOCATION": "LOCATION",
    "GPE": "LOCATION",
    "ORG": "ORGANIZATION",
    "ORGANIZATION": "ORGANIZATION",
    "NORP": "NRP",
    "AGE": "AGE",
    "ID": "ID",
    "EMAIL": "EMAIL",
    "PATIENT": "PERSON",
    "STAFF": "PERSON",
    "HOSP": "ORGANIZATION",
    "PATORG": "ORGANIZATION",
    "DATE": "DATE_TIME",
    "TIME": "DATE_TIME",
    "PHONE": "PHONE_NUMBER",
    "HCW": "PERSON",
    "HOSPITAL": "ORGANIZATION",
    "FACILITY": "LOCATION",
}


@st.cache_resource
def get_analyzer(model_key: str) -> AnalyzerEngine:
    """Create and cache the Presidio AnalyzerEngine for a given model."""
    model_config = NER_MODELS[model_key]
    engine_name = model_config["engine"]
    model_name = model_config["model"]

    if engine_name == "spacy":
        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": model_name}],
            "ner_model_configuration": {
                "model_to_presidio_entity_mapping": SPACY_ENTITY_MAPPING,
                "low_confidence_score_multiplier": 0.4,
                "low_score_entity_names": ["ORG", "ORGANIZATION"],
            },
        }
    else:
        nlp_configuration = {
            "nlp_engine_name": "transformers",
            "models": [{
                "lang_code": "en",
                "model_name": {"spacy": "en_core_web_sm", "transformers": model_name},
            }],
            "ner_model_configuration": {
                "model_to_presidio_entity_mapping": TRANSFORMERS_ENTITY_MAPPING,
                "low_confidence_score_multiplier": 0.4,
                "low_score_entity_names": ["ID"],
                "labels_to_ignore": [
                    "CARDINAL", "EVENT", "LANGUAGE", "LAW", "MONEY",
                    "ORDINAL", "PERCENT", "PRODUCT", "QUANTITY", "WORK_OF_ART",
                ],
            },
        }

    nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)
    registry.add_recognizer(_create_company_recognizer())

    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
    return analyzer


def read_file_text(file_bytes: bytes, filename: str) -> str:
    """Extract text from uploaded file bytes based on extension."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf_text(file_bytes)
    elif ext == ".docx":
        return _extract_docx_text(file_bytes)
    else:
        return file_bytes.decode("utf-8", errors="replace")


def _extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using pymupdf."""
    text_parts = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def _extract_docx_text(file_bytes: bytes) -> str:
    """Extract text from DOCX bytes using python-docx."""
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)


MAX_CHUNK_SIZE = 900000  # Stay under spaCy's 1M char limit


def analyze_text(text: str, language: str, entities: List[str], score_threshold: float, model_key: str = "") -> List[RecognizerResult]:
    """Run Presidio Analyzer on text, chunking if necessary."""
    analyzer = get_analyzer(model_key)
    if len(text) <= MAX_CHUNK_SIZE:
        return analyzer.analyze(
            text=text,
            language=language,
            entities=entities if entities else None,
            score_threshold=score_threshold,
        )

    # Chunk large texts
    all_results = []
    for start in range(0, len(text), MAX_CHUNK_SIZE):
        chunk = text[start:start + MAX_CHUNK_SIZE]
        chunk_results = analyzer.analyze(
            text=chunk,
            language=language,
            entities=entities if entities else None,
            score_threshold=score_threshold,
        )
        # Adjust offsets to match original text positions
        for r in chunk_results:
            r.start += start
            r.end += start
        all_results.extend(chunk_results)
    return all_results


def results_to_records(results: List[RecognizerResult], text: str) -> List[Dict[str, Any]]:
    """Convert analyzer results to a list of dicts for display."""
    records = []
    for r in results:
        records.append({
            "Entity Type": r.entity_type,
            "Text": text[r.start:r.end],
            "Start": r.start,
            "End": r.end,
            "Score": round(r.score, 2),
        })
    return records


def highlight_text(text: str, results: List[RecognizerResult]) -> str:
    """Create an HTML string with PII entities highlighted."""
    if not results:
        return f"<pre style='white-space: pre-wrap;'>{text}</pre>"

    sorted_results = sorted(results, key=lambda x: x.start)
    html_parts = []
    last_end = 0

    colors = {
        "PERSON": "#ff6b6b",
        "EMAIL_ADDRESS": "#ffa94d",
        "PHONE_NUMBER": "#69db7c",
        "LOCATION": "#74c0fc",
        "DATE_TIME": "#b197fc",
        "CREDIT_CARD": "#ff8787",
        "CRYPTO": "#ffd43b",
        "IBAN_CODE": "#a9e34b",
        "IP_ADDRESS": "#66d9e8",
        "NRP": "#e599f7",
        "MEDICAL_LICENSE": "#ff922b",
        "URL": "#20c997",
        "ORGANIZATION": "#da77f2",
    }
    default_color = "#ffc9c9"

    for r in sorted_results:
        if r.start > last_end:
            html_parts.append(_escape_html(text[last_end:r.start]))
        color = colors.get(r.entity_type, default_color)
        entity_text = _escape_html(text[r.start:r.end])
        html_parts.append(
            f'<mark style="background-color: {color}; padding: 2px 4px; border-radius: 3px;" '
            f'title="{r.entity_type} ({r.score:.2f})">'
            f'{entity_text} <sup style="font-size: 0.7em;">{r.entity_type}</sup></mark>'
        )
        last_end = r.end

    if last_end < len(text):
        html_parts.append(_escape_html(text[last_end:]))

    return f"<div style='white-space: pre-wrap; font-family: monospace; line-height: 1.6;'>{''.join(html_parts)}</div>"


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _is_supported_file(filename: str) -> bool:
    """Check if a file has an allowed extension and is not excluded."""
    name = Path(filename).name
    if name in EXCLUDED_FILENAMES:
        return False
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _is_in_excluded_folder(file_path: Path, root: Path) -> bool:
    """Check if a file is inside an excluded folder."""
    rel = file_path.relative_to(root)
    return any(part in EXCLUDED_FOLDERS for part in rel.parts[:-1])


def process_zip(uploaded_zip) -> Dict[str, bytes]:
    """Extract files from a zip archive."""
    files = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getvalue())
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Guard against zip slip
            for info in zf.infolist():
                if info.is_dir():
                    continue
                member_path = os.path.normpath(info.filename)
                if member_path.startswith("..") or os.path.isabs(member_path):
                    continue
                if _is_supported_file(info.filename) and not any(part in EXCLUDED_FOLDERS for part in Path(info.filename).parts[:-1]):
                    files[info.filename] = zf.read(info.filename)
    return files


def collect_folder_files(folder_path: str) -> Dict[str, bytes]:
    """Recursively collect all supported files from a folder."""
    files = {}
    root = Path(folder_path)
    if not root.is_dir():
        return files
    for file_path in sorted(root.rglob("*")):
        if file_path.is_file() and _is_supported_file(file_path.name) and not _is_in_excluded_folder(file_path, root):
            rel_path = str(file_path.relative_to(root))
            files[rel_path] = file_path.read_bytes()
    return files


# ─── Sidebar ───────────────────────────────────────────────
st.sidebar.header("Presidio Analyzer")
st.sidebar.markdown("Scan documents for PII entities using [Microsoft Presidio](https://microsoft.github.io/presidio/).")

st_model_key = st.sidebar.selectbox(
    "NER model",
    options=list(NER_MODELS.keys()),
    index=0,
    help="Select the Named Entity Recognition model. Transformer models are more accurate but slower and require downloading from HuggingFace on first use.",
)

language = st.sidebar.selectbox("Language", ["en"], index=0)

with st.sidebar.expander("Model info"):
    cfg = NER_MODELS[st_model_key]
    st.write(f"**Engine:** {cfg['engine']}")
    st.write(f"**Model:** {cfg['model']}")
    if cfg["engine"] == "transformers":
        st.info("Transformer models download on first use (~500MB-1GB). First analysis will be slower.")

analyzer = get_analyzer(st_model_key)
all_entities = sorted(analyzer.get_supported_entities())

selected_entities = st.sidebar.multiselect(
    "Entities to detect",
    options=["All"] + all_entities,
    default=["All"],
    help="Select which PII entity types to scan for.",
)

if "All" in selected_entities or not selected_entities:
    entities_to_use = None  # Analyzer will use all
    entities_list = []
else:
    entities_to_use = selected_entities
    entities_list = selected_entities

score_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    step=0.05,
    help="Minimum confidence score for a detection to be included.",
)

# ─── Main Area ─────────────────────────────────────────────
st.title("Presidio Document Analyzer")
st.markdown("Upload documents, a zip archive, or point to a local folder to scan for PII entities.")

upload_mode = st.radio("Upload mode", ["Files", "Zip archive", "Local folder"], horizontal=True)

if upload_mode == "Files":
    uploaded_files = st.file_uploader(
        "Drop files here",
        accept_multiple_files=True,
        type=["txt", "md", "csv", "log", "json", "xml", "html", "yml", "yaml", "pdf", "docx"],
        help="Upload text files, PDFs, or Word documents.",
    )
elif upload_mode == "Zip archive":
    uploaded_zip = st.file_uploader(
        "Drop a .zip archive here",
        accept_multiple_files=False,
        type=["zip"],
        help="Upload a zip archive containing text files, PDFs, or Word documents.",
    )
    uploaded_files = None
else:
    folder_path = st.text_input(
        "Folder path",
        placeholder=r"C:\Users\you\Documents\contracts",
        help="Enter the full path to a local folder. All supported files will be scanned recursively.",
    )
    uploaded_files = None

# Collect files to process
files_to_process: Dict[str, bytes] = {}

if upload_mode == "Files" and uploaded_files:
    for f in uploaded_files:
        files_to_process[f.name] = f.getvalue()
elif upload_mode == "Zip archive" and "uploaded_zip" in dir() and uploaded_zip is not None:
    files_to_process = process_zip(uploaded_zip)
elif upload_mode == "Local folder" and "folder_path" in dir() and folder_path:
    resolved = Path(folder_path).resolve()
    if resolved.is_dir():
        files_to_process = collect_folder_files(str(resolved))
        if not files_to_process:
            st.warning(f"No supported files found in `{folder_path}`. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    else:
        st.error(f"Folder not found: `{folder_path}`")

# Process
if files_to_process:
    if upload_mode == "Local folder":
        st.info(f"Found **{len(files_to_process)}** supported files in `{folder_path}`")
    col_analyze, col_stop = st.columns([1, 1])
    with col_analyze:
        analyze_clicked = st.button("Analyze", type="primary")
    with col_stop:
        stop_pressed = st.button("Stop", type="secondary")

    if analyze_clicked:
        all_records = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        total = len(files_to_process)
        stopped = False

        for idx, (filename, file_bytes) in enumerate(files_to_process.items()):
            if stop_pressed:
                stopped = True
                st.warning(f"Stopped after processing {idx} of {total} files.")
                break

            status_text.markdown(f"**Processing:** `{filename}` ({idx + 1}/{total})")

            try:
                text = read_file_text(file_bytes, filename)

                if not text.strip():
                    st.warning(f"**{filename}** is empty, skipping.")
                    progress_bar.progress((idx + 1) / total)
                    continue

                results = analyze_text(text, language, entities_list, score_threshold, st_model_key)

                with st.expander(f"**{filename}** — {len(results)} PII entities found", expanded=(len(files_to_process) == 1)):
                    if results:
                        # Highlighted text
                        st.markdown("#### Highlighted Text")
                        st.markdown(highlight_text(text, results), unsafe_allow_html=True)

                        # Results table
                        st.markdown("#### Detected Entities")
                        records = results_to_records(results, text)
                        df = pd.DataFrame(records)
                        st.dataframe(df, use_container_width=True)

                        for rec in records:
                            rec["File"] = filename
                        all_records.extend(records)
                    else:
                        st.success("No PII entities detected.")
            except Exception as e:
                st.error(f"**{filename}** — Error: {e}")

            progress_bar.progress((idx + 1) / total)

        progress_bar.empty()
        status_text.empty()

        # Summary
        st.divider()
        if stopped:
            st.subheader("Partial Results (stopped)")
        else:
            st.subheader("Summary")
        col1, col2 = st.columns(2)
        files_processed = idx + 1 if not stopped else idx
        col1.metric("Files scanned", f"{files_processed}/{total}")
        col2.metric("Total PII entities found", len(all_records))

        if all_records:
            summary_df = pd.DataFrame(all_records)
            st.markdown("#### All Detections")
            st.dataframe(summary_df, use_container_width=True)

            # Entity type breakdown
            st.markdown("#### By Entity Type")
            type_counts = summary_df["Entity Type"].value_counts()
            st.bar_chart(type_counts)

            # Download results as JSON
            json_str = json.dumps(all_records, indent=2)
            st.download_button(
                label="Download results as JSON",
                data=json_str,
                file_name="presidio_analysis_results.json",
                mime="application/json",
            )
elif upload_mode == "Files":
    st.info("Upload files using the panel above to get started.")
elif upload_mode == "Zip archive":
    st.info("Upload a zip archive using the panel above to get started.")
elif upload_mode == "Local folder":
    st.info("Enter a folder path above to get started.")
