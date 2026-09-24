import streamlit as st
from pathlib import Path
import sys
import subprocess
import re
import torch


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(
    "/mnt/c/Users/rohan/MTech_Sanskrit_NMT"
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Sanskrit → English Translator",
    page_icon="🕉️",
    layout="wide"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ============================================================
# MODELS
# ============================================================

INDICTRANS_MODEL = (
    "krpraveen/indictrans2-sanskrit-en-finetuned"
)


# ============================================================
# WINDOWS TESSERACT
# ============================================================

# Tesseract itself is installed on Windows.
# Streamlit is running inside WSL.

# ============================================================
# WINDOWS TESSERACT
# ============================================================

# Tesseract executable is installed on Windows.
# Streamlit is running inside WSL.

TESSERACT = "/mnt/c/Program Files/Tesseract-OCR/tesseract.exe"

# Windows path passed to the Windows Tesseract executable
TESSDATA = r"C:\Program Files\Tesseract-OCR\tessdata"


# ============================================================
# TITLE
# ============================================================

st.title(
    "🕉️ Sanskrit → English Translator"
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = text.replace(
        "\n",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def contains_devanagari(text):

    return bool(
        re.search(
            r"[\u0900-\u097F]",
            text
        )
    )


# ============================================================
# OCR
# ============================================================

def run_ocr(image_path):

    # --------------------------------------------------------
    # WSL paths
    # --------------------------------------------------------

    tesseract_path = (
        "/mnt/c/Program Files/"
        "Tesseract-OCR/tesseract.exe"
    )

    tessdata_path = (
        r"C:\Program Files\Tesseract-OCR\tessdata"
    )

    # --------------------------------------------------------
    # Check Tesseract executable
    # --------------------------------------------------------

    if not Path(tesseract_path).exists():
        raise FileNotFoundError(
            f"Tesseract not found: {tesseract_path}"
        )

    # --------------------------------------------------------
    # Check Sanskrit language data
    # --------------------------------------------------------

    san_data = Path(
        "/mnt/c/Program Files/"
        "Tesseract-OCR/tessdata/san.traineddata"
    )

    if not san_data.exists():
        raise FileNotFoundError(
            "san.traineddata was not found."
        )

    # --------------------------------------------------------
    # Convert WSL image path to Windows path
    # --------------------------------------------------------

    result = subprocess.run(
        [
            "wslpath",
            "-w",
            str(image_path)
        ],
        capture_output=True,
        text=True,
        check=True
    )

    windows_image_path = result.stdout.strip()

    # --------------------------------------------------------
    # Tesseract command
    # --------------------------------------------------------

    command = [
        tesseract_path,
        windows_image_path,
        "stdout",
        "--tessdata-dir",
        tessdata_path,
        "-l",
        "san",
        "--psm",
        "6"
    ]

    # --------------------------------------------------------
    # Run Tesseract
    # --------------------------------------------------------

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    # --------------------------------------------------------
    # Check error
    # --------------------------------------------------------

    if result.returncode != 0:

        raise RuntimeError(
            result.stderr.strip()
        )

    # --------------------------------------------------------
    # Return OCR text
    # --------------------------------------------------------

    return clean_text(
        result.stdout
    )
# ============================================================
# INDIC TRANS2 LOADING
# ============================================================

@st.cache_resource
def load_indictrans2():

    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM
    )

    from IndicTransToolkit import (
        IndicProcessor
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        INDICTRANS_MODEL,
        trust_remote_code=True
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = AutoModelForSeq2SeqLM.from_pretrained(
        INDICTRANS_MODEL,
        trust_remote_code=True
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    processor = IndicProcessor(
        inference=True
    )

    return (
        tokenizer,
        model,
        processor
    )


# ============================================================
# INDIC TRANS2 TRANSLATION
# ============================================================

def translate_indictrans2(
    sanskrit_text
):

    from indic_transliteration import (
        sanscript
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    tokenizer, model, processor = (
        load_indictrans2()
    )

    # --------------------------------------------------------
    # Convert IAST → Devanagari
    #
    # IndicTrans2 expects Sanskrit in
    # Devanagari for san_Deva.
    # --------------------------------------------------------

    if contains_devanagari(
        sanskrit_text
    ):

        devanagari = sanskrit_text

    else:

        devanagari = sanscript.transliterate(
            sanskrit_text,
            sanscript.IAST,
            sanscript.DEVANAGARI
        )

    # --------------------------------------------------------
    # IndicTrans preprocessing
    # --------------------------------------------------------

    preprocessed = processor.preprocess_batch(
        [devanagari],
        src_lang="san_Deva",
        tgt_lang="eng_Latn"
    )

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    inputs = tokenizer(
        preprocessed,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )

    # --------------------------------------------------------
    # Move tensors to GPU
    # --------------------------------------------------------

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    # --------------------------------------------------------
    # Translation
    #
    # beam=2 is intentionally used because
    # RTX 3050 has 6 GB VRAM.
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            num_beams=2,

            max_length=128,

            length_penalty=1.2,

            no_repeat_ngram_size=3
        )

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    translations = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )

    # --------------------------------------------------------
    # IndicTrans postprocessing
    # --------------------------------------------------------

    translations = processor.postprocess_batch(
        translations,
        lang="eng_Latn"
    )

    return translations[0]


# ============================================================
# INPUT TYPE
# ============================================================

st.subheader(
    "📥 Input"
)

input_type = st.radio(
    "Choose input type",
    [
        "Text",
        "Image"
    ],
    horizontal=True
)


sanskrit_text = ""


# ============================================================
# TEXT INPUT
# ============================================================

if input_type == "Text":

    sanskrit_text = st.text_area(
        "Enter Sanskrit Text",
        height=160,
        placeholder=(
            "रामः वनं गच्छति"
        )
    )


# ============================================================
# IMAGE INPUT
# ============================================================

else:

    uploaded_file = st.file_uploader(
        "Upload Sanskrit Image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp"
        ]
    )

    if uploaded_file:

        # ----------------------------------------------------
        # Display image
        # ----------------------------------------------------

        st.image(
            uploaded_file,
            caption="Uploaded Sanskrit Image",
            use_container_width=True
        )

        # ----------------------------------------------------
        # Temporary directory
        # ----------------------------------------------------

        temp_dir = (
            PROJECT_ROOT
            / "application"
            / "temp"
        )

        temp_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Save uploaded image
        # ----------------------------------------------------

        image_path = (
            temp_dir
            / uploaded_file.name
        )

        with open(
            image_path,
            "wb"
        ) as file:

            file.write(
                uploaded_file.getbuffer()
            )

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        if st.button(
            "🔍 Extract Sanskrit Text"
        ):

            with st.spinner(
                "Running Sanskrit OCR..."
            ):

                try:

                    ocr_text = run_ocr(
                        image_path
                    )

                    if ocr_text:

                        st.session_state[
                            "ocr_text"
                        ] = ocr_text

                        st.success(
                            "OCR completed successfully."
                        )

                    else:

                        st.warning(
                            "OCR did not detect Sanskrit text."
                        )

                except Exception as e:

                    st.error(
                        f"OCR Failed: {e}"
                    )

    # --------------------------------------------------------
    # Retrieve OCR text
    # --------------------------------------------------------

    if "ocr_text" in st.session_state:

        sanskrit_text = (
            st.session_state["ocr_text"]
        )

        st.subheader(
            "📜 OCR Extracted Sanskrit"
        )

        st.text_area(
            "Detected text",
            sanskrit_text,
            height=120
        )


# ============================================================
# TRANSLATE BUTTON
# ============================================================

st.divider()

translate_button = st.button(
    "🔄 Translate",
    type="primary",
    use_container_width=True
)


# ============================================================
# TRANSLATION
# ============================================================

if translate_button:

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    sanskrit_text = clean_text(
        sanskrit_text
    )

    if not sanskrit_text:

        st.warning(
            "Please enter Sanskrit text "
            "or upload an image and extract the text."
        )

        st.stop()

    # --------------------------------------------------------
    # Display Sanskrit
    # --------------------------------------------------------

    st.subheader(
        "📜 Sanskrit Text"
    )

    st.info(
        sanskrit_text
    )

    # --------------------------------------------------------
    # Translation
    # --------------------------------------------------------

    with st.spinner(
        "Translating with IndicTrans2..."
    ):

        try:

            english = translate_indictrans2(
                sanskrit_text
            )

        except torch.cuda.OutOfMemoryError:

            torch.cuda.empty_cache()

            st.error(
                "GPU memory is full. "
                "Close other GPU applications "
                "and try again."
            )

            st.stop()

        except Exception as e:

            st.error(
                f"Translation failed: {e}"
            )

            st.stop()

    # --------------------------------------------------------
    # English output
    # --------------------------------------------------------

    st.subheader(
        "English Translation"
    )

    st.success(
        english
    )

