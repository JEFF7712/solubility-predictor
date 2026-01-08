from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
from model import GNN
from utils import smile_to_data

app = FastAPI(title="Solubility Predictor")

# Load Model
device = torch.device('cpu')
model = GNN(hidden_dim=128)
model.load_state_dict(torch.load("gnn_solubility.pth", map_location=device))
model.to(device)
model.eval()

class Request(BaseModel):
    smiles: str

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Solubility Predictor</title>
        </head>
        <body>
            <h1>Welcome to the Solubility Predictor API</h1>
        </body>
    </html>
    """

@app.post("/predict")
async def predict(req: Request):
    data = smile_to_data(req.smiles)
    
    if data is None:
        raise HTTPException(400, "Invalid SMILES string")
    
    # Create batch index for single item
    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
    data = data.to(device)
    
    with torch.no_grad():
        pred = model(data).item()
        
    return {
        "molecule": req.smiles,
        "logS": round(pred, 4),
        "solubility": "Soluble" if pred > -2.5 else "Insoluble"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)