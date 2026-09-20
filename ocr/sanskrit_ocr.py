from pathlib import Path
import sys
import cv2
import pytesseract
import subprocess

# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSDATA_PATH = r"C:\Program Files\Tesseract-OCR\tessdata"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    # Grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Upscale 3x
    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # Mild denoising
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    return gray
# def preprocess_image(image_path):

#     image = cv2.imread(str(image_path))

#     if image is None:
#         raise FileNotFoundError(
#             f"Could not read image:\n{image_path}"
#         )

#     # Convert to grayscale
#     gray = cv2.cvtColor(
#         image,
#         cv2.COLOR_BGR2GRAY
#     )

#     # Upscale image
#     gray = cv2.resize(
#         gray,
#         None,
#         fx=2,
#         fy=2,
#         interpolation=cv2.INTER_CUBIC
#     )

#     # Denoise
#     gray = cv2.GaussianBlur(
#         gray,
#         (3, 3),
#         0
#     )

#     # Adaptive threshold
#     binary = cv2.adaptiveThreshold(
#         gray,
#         255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         31,
#         11
#     )

#     return binary


# ============================================================
# SANSKRIT OCR
# ============================================================
# def extract_sanskrit_text(image_path):

#     command = [
#         TESSERACT_PATH,
#         "--tessdata-dir",
#         TESSDATA_PATH,
#         str(image_path),
#         "stdout",
#         "-l",
#         "san",
#         "--psm",
#         "6"
#     ]

#     result = subprocess.run(
#         command,
#         capture_output=True,
#         text=True,
#         encoding="utf-8"
#     )

#     if result.returncode != 0:
#         raise RuntimeError(result.stderr)

#     return result.stdout.strip()
def extract_sanskrit_text(image_path):

    command = [
        TESSERACT_PATH,
        "--tessdata-dir",
        TESSDATA_PATH,
        str(image_path),
        "stdout",
        "-l",
        "san",
        "--psm",
        "6"
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout.strip()
# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:

        print("Usage:")
        print(
            r"python ocr\sanskrit_ocr.py ocr\images\sanskrit.png"
        )

        return

    image_path = Path(sys.argv[1])

    if not image_path.exists():

        print(
            f"ERROR: Image not found:\n{image_path}"
        )

        return

    print("=" * 70)
    print("SANSKRIT OCR")
    print("=" * 70)

    print("\nImage:", image_path)

    print(
        "Tesseract:",
        TESSERACT_PATH
    )

    print(
        "Tessdata:",
        TESSDATA_PATH
    )

    print(
        "Language: Sanskrit (san)"
    )

    print("\nProcessing...")

    try:

        text = extract_sanskrit_text(
            image_path
        )

        print("\n" + "=" * 70)
        print("EXTRACTED SANSKRIT TEXT")
        print("=" * 70)

        if text:
            print("\n" + text)
        else:
            print(
                "\nNo Sanskrit text detected."
            )

    except Exception as e:

        print("\nOCR ERROR:")
        print(e)


if __name__ == "__main__":
    main()