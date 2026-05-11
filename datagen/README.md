Audio Stem Sampler & Pair Generator

This tool acts as a data-engineering pipeline. It takes a folder of audio stems, precomputes harmonic and rhythmic features using caching, and generates artificial target/source audio pairs formatted as PyTorch .pt tensors for later training.


Folder Structure & Setup

Place your stems in the default directory data/stems/.
Files must follow the naming convention: TrackID_(Instrument).wav.
For the script to properly package your audio data into pairs before training, 
please do a stem separation with google's demucs 6s separation track into 6 stems 
Use these names after a song id {"Drums", "Other", "Guitar", "Piano", "Vocals", "Bass"}
(e.g., Song123_(Vocals).wav, Song123_(Bass).wav)

The project will automatically create data/output/ and data/cache/ directories relative to the config.py file. If you wish to use custom directories, simply create a .env file in the root folder with the following contents:

STEMS_ROOT=/path/to/your/stems
OUTPUT_DIR=/path/to/your/output
CACHE_DIR=/path/to/your/cache
SAMPLE_RATE=16000


Usage

This is a two-step process:

Step 1: Precompute Features
This script creates a grouped metadata CSV and local JSON caches so that your ML sampler doesn't get bottlenecked reading full .wav files continuously.

python precompute.py


Step 2: Generate Samples
Run the multiprocessing sampler to start churning out your source and target paired .pt files. It will automatically zip batches of 200 files.

python sampler.py
