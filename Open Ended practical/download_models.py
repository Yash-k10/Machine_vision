import os
import urllib.request
import zipfile

def download_easyocr_models():
    model_dir = os.path.expanduser('~/.EasyOCR/model')
    os.makedirs(model_dir, exist_ok=True)
    
    craft_pth = os.path.join(model_dir, 'craft_mlt_25k.pth')
    eng_pth = os.path.join(model_dir, 'english_g2.pth')

    if not os.path.exists(craft_pth):
        print("Downloading CRAFT detection model (~17MB)...")
        craft_zip = os.path.join(model_dir, 'craft.zip')
        urllib.request.urlretrieve('https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip', craft_zip)
        with zipfile.ZipFile(craft_zip, 'r') as z:
            z.extractall(model_dir)
        if os.path.exists(craft_zip):
            os.remove(craft_zip)
        print("[OK] CRAFT model ready!")
    else:
        print("[OK] CRAFT model already exists.")

    if not os.path.exists(eng_pth):
        print("Downloading English recognition model (~44MB)...")
        eng_zip = os.path.join(model_dir, 'eng.zip')
        urllib.request.urlretrieve('https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip', eng_zip)
        with zipfile.ZipFile(eng_zip, 'r') as z:
            z.extractall(model_dir)
        if os.path.exists(eng_zip):
            os.remove(eng_zip)
        print("[OK] English recognition model ready!")
    else:
        print("[OK] English recognition model already exists.")

    print("SUCCESS: All EasyOCR model weights are installed and ready!")

if __name__ == "__main__":
    download_easyocr_models()
