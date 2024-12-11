# Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements (excluding torch)
pip install -r requirements.txt

# Install PyTorch and check CUDA availability
# pip install torch
python -c "import torch; print(torch.cuda.is_available())"
