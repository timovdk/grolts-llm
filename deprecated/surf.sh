module load 2023 CUDA/12.4.0 Python/3.11.3-GCCcore-12.3.0

python -m venv venv
source venv/bin/activate
pip install -U pip
pip install pandas tqdm MarkItDown torch torchvision torchaudio transformers bitsandbytes langchain python-dotenv langchain-chroma langchain-community langchain-openai langchain-groq accelerate pypdf FlagEmbedding
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True