module load 2023
module load CUDA/12.4.0
module load Python/3.11.3-GCCcore-12.3.0

python -m venv venv
source venv/bin/activate
pip install -U pip
pip install pandas tqdm MarkItDown torch torchvision torchaudio transformers bitsandbytes langchain python-dotenv