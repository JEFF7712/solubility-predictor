from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from .model import GNN
from .utils import smile_to_data
import os
import logging
import re
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "gnn_solubility.pth")

app = FastAPI(title="Solubility Predictor")

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda req, exc: HTTPException(429, "Rate limit exceeded"))

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Load Model
device = torch.device('cpu')
model = GNN(hidden_dim=128)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    logger.info("Model loaded successfully")
except FileNotFoundError:
    logger.error(f"Model file not found at {MODEL_PATH}")
    raise RuntimeError("Model file not found. Cannot start server.")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise RuntimeError(f"Error loading model: {str(e)}")
model.to(device)
model.eval()

class Request(BaseModel):
    smiles: str
    
    @field_validator('smiles')
    @classmethod
    def validate_smiles(cls, v):
        if len(v) > 500:
            raise ValueError("SMILES string too long (max 500 characters)")
        if not v or not v.strip():
            raise ValueError("SMILES string cannot be empty")
        v = v.strip()
        if not re.match(r'^[A-Za-z0-9@+\-()[\]=/#\\]+$', v):
            raise ValueError("Invalid characters in SMILES string")
        return v

@app.get("/")
def read_root():
    return FileResponse('static/index.html')

@app.post("/predict")
@limiter.limit("60/minute")
async def predict(request: FastAPIRequest, req: Request):
    try:
        data = smile_to_data(req.smiles)
        if data is None:
            logger.warning(f"Invalid SMILES string: {req.smiles}")
            raise HTTPException(400, "Invalid SMILES string")
        
        if data.x.shape[0] > 500:
            logger.warning(f"SMILES string too complex (>500 atoms): {req.smiles}")
            raise HTTPException(400, "Molecule too large (max 500 atoms)")
        
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
        data = data.to(device)
        
        with torch.no_grad():
            pred = model(data).item()
        
        if not (-20 < pred < 20):
            logger.error(f"Prediction out of expected range: {pred}")
            raise HTTPException(500, "Model produced invalid prediction")
        
        mol = Chem.MolFromSmiles(req.smiles)
        if mol is None:
            mol_block = ""
        else:
            mol = Chem.AddHs(mol)
            try:
                AllChem.EmbedMolecule(mol, randomSeed=42)
                mol_block = Chem.MolToMolBlock(mol)
                if len(mol_block) > 50000:
                    mol_block = mol_block[:50000]
            except Exception as e:
                logger.warning(f"Error generating mol block: {str(e)}")
                mol_block = ""
        
        logger.info(f"Prediction successful for SMILES: {req.smiles}")
        return {
            "molecule": req.smiles,
            "logS": round(pred, 4),
            "solubility": "Soluble" if pred > -2.5 else "Insoluble",
            "mol_block": mol_block
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        raise HTTPException(500, "Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)