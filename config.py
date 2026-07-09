from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# project root
ROOT_DIR = Path(__file__).resolve().parent

# data paths
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA = DATA_DIR / 'raw'
INTERIM_DATA = DATA_DIR / 'interim'
PROCESSED_DATA = DATA_DIR / 'processed'



