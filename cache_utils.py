import json
from config import CACHE_DIR
from functools import lru_cache

@lru_cache(maxsize=1000) # Prevents constant disk reads!
def load_cached_features(track_id: str):
    cache_path = CACHE_DIR / f"{track_id}.json"
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load cache for {track_id}: {e}")
    return None

def get_cached_tempo_and_key(track_id: str):
    """
    Retrieve tempo and key from cache, or return None if not available.
    """
    cached = load_cached_features(track_id)
    if cached and "bpm" in cached and "key" in cached:
        return float(cached["bpm"]), cached["key"]
    return None, None

def get_cached_top_regions(track_id: str, region_duration: float, top_k: int = 5):
    """
    Retrieve top-k scored regions from cache based on region duration.
    """
    cached = load_cached_features(track_id)
    if cached:
        key = "5" if region_duration <= 5.1 else "15"
        if key in cached:
            regions = sorted(cached[key], key=lambda x: -x[1])[:top_k]
            return [(float(start), float(score)) for start, score in regions]
    return None