// Initialize 3D Viewer
let element = document.getElementById('viewer');
let viewer = $3Dmol.createViewer(element, { backgroundColor: 'black' });

// Map a solubility class to one of three colour tiers.
function solubilityTier(cls) {
    if (cls === 'Highly soluble' || cls === 'Soluble') return 'soluble';
    if (cls === 'Slightly soluble') return 'slightly';
    return 'insoluble';
}

async function predict() {
    const smiles = document.getElementById('smilesInput').value;
    const statusDiv = document.getElementById('statusText');
    const warningDiv = document.getElementById('warningText');
    const btn = document.getElementById('analyzeBtn');

    // UI Loading State
    statusDiv.innerText = "Computing...";
    statusDiv.className = "";
    warningDiv.innerText = "";
    btn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ smiles: smiles })
        });

        const data = await response.json();

        if (response.ok) {
            statusDiv.innerText = `${data.solubility_class}   Log S = ${data.logS} ± ${data.logS_std}`;
            statusDiv.className = solubilityTier(data.solubility_class);

            if (data.warnings && data.warnings.length) {
                warningDiv.innerText = data.warnings.join(' ');
            }

            // Render 3D Molecule
            viewer.clear();
            if (data.mol_block) {
                viewer.addModel(data.mol_block, "mol");
                viewer.setStyle({}, { stick: {radius: 0.15}, sphere: {scale: 0.2} });
                viewer.zoomTo();
                viewer.render();
            } else {
                warningDiv.innerText = ("No 3D structure available. " + warningDiv.innerText).trim();
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

// Example molecule buttons fill the input and predict.
document.querySelectorAll('.example').forEach(function (el) {
    el.addEventListener('click', function () {
        document.getElementById('smilesInput').value = el.dataset.smiles;
        predict();
    });
});

predict();