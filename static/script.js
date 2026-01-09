// Initialize 3D Viewer
let element = document.getElementById('viewer');
let viewer = $3Dmol.createViewer(element, { backgroundColor: 'black' });

async function predict() {
    const smiles = document.getElementById('smilesInput').value;
    const statusDiv = document.getElementById('statusText');
    const btn = document.getElementById('analyzeBtn');
    
    // UI Loading State
    statusDiv.innerText = "Computing...";
    statusDiv.className = "";
    btn.disabled = true;
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ smiles: smiles })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Update Status Text
            statusDiv.innerText = `${data.solubility} (Log(S): ${data.logS})`;
            statusDiv.className = data.solubility === 'Soluble' ? 'soluble' : 'insoluble';

            // Render 3D Molecule
            viewer.clear();
            if (data.mol_block) {
                viewer.addModel(data.mol_block, "mol");
                viewer.setStyle({}, { stick: {radius: 0.15}, sphere: {scale: 0.2} });
                viewer.zoomTo();
                viewer.render();
            } else {
                statusDiv.innerText += " (No 3D data)";
            }
        } else {
            statusDiv.innerText = "Error: " + data.detail;
            statusDiv.className = "insoluble";
        }
    } catch (e) {
        console.error(e);
        statusDiv.innerText = "Network Error";
        statusDiv.className = "insoluble";
    }
    
    btn.disabled = false;
}

// Event Listeners
document.getElementById('analyzeBtn').addEventListener('click', predict);

// Allow pressing "Enter" in the input box
document.getElementById('smilesInput').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') predict();
});

predict();