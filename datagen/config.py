# config.py - SUGGESTED FIX
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))
SEGMENT_LENGTH = 15
SNIPPET_LENGTH = 5
BATCH_SIZE = 200 # Moved from sampler.py

# Default to folders inside the current project directory!
BASE_DIR = Path(__file__).resolve().parent
STEMS_ROOT = Path(os.getenv("STEMS_ROOT", BASE_DIR / "data" / "stems"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "data" / "output"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", BASE_DIR / "data" / "cache"))
CSV_PATH = CACHE_DIR / "track_metadata_grouped.csv" # Moved here for global access

STEM_INSTRUMENTS = ["Drums", "Other", "Guitar", "Piano", "Vocals", "Bass"]
STEM_INSTRUMENTS_VIRTUAL = ["Drums", "Harmony", "Guitar", "Piano", "Vocals"]

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
CACHE_DIR.mkdir(exist_ok=True, parents=True)