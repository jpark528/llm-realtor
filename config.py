import os
from decouple import Config, RepositoryEnv
from pathlib import Path

root_dir = Path().resolve()

config = Config(RepositoryEnv(root_dir / '.env'))  # Explicitly load .env

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = config('LANGSMITH_API_KEY')
