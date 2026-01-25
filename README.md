# Fish Recognition API

## Setup
```bash
# Clone & install
git clone <repo>
cd fish-recognition-api
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

pip install -r requirements.txt

# Add your model
cp /path/to/your/fish_model.keras models/

# Run
python app.py
# or
bash run.sh