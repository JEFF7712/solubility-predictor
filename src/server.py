from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from .model import GNN
from .utils import smile_to_data
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "gnn_solubility.pth")

app = FastAPI(title="Solubility Predictor")

app.mount("/static", StaticFiles(directory="static"), name="static")

# Load Model
device = torch.device('cpu')
model = GNN(hidden_dim=128)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except FileNotFoundError:
    print("Model file not found!")
model.to(device)
model.eval()

class Request(BaseModel):
    smiles: str

@app.get("/")
def read_root():
    return FileResponse('static/index.html')

@app.post("/predict")
async def predict(req: Request):
    data = smile_to_data(req.smiles)
    if data is None:
        raise HTTPException(400, "Invalid SMILES string")
    
    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
    data = data.to(device)
    
    with torch.no_grad():
        pred = model(data).item()
    
    mol = Chem.MolFromSmiles(req.smiles)
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol_block = Chem.MolToMolBlock(mol)
    except:
        mol_block = ""

    return {
        "molecule": req.smiles,
        "logS": round(pred, 4),
        "solubility": "Soluble" if pred > -2.5 else "Insoluble",
        "mol_block": mol_block
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)