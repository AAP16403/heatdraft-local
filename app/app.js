let APP_DATA = null;

function loadData() {
  APP_DATA = APP_DATA_RAW;
  populateDropdowns();
}

function populateDropdowns() {
  const contSel = document.getElementById('contaminant-select');
  const mbSel = document.getElementById('membrane-select');

  APP_DATA.contaminants
    .sort((a, b) => a['Types of contaminants'].localeCompare(b['Types of contaminants']))
    .forEach(c => {
      const opt = document.createElement('option');
      opt.value = c['Types of contaminants'];
      opt.textContent = c['Types of contaminants'];
      contSel.appendChild(opt);
    });

  APP_DATA.membranes
    .sort((a, b) => a['Name of MB'].localeCompare(b['Name of MB']))
    .forEach(m => {
      const opt = document.createElement('option');
      opt.value = m['Name of MB'];
      opt.textContent = `${m['Name of MB']} (${m['Type of MB']})`;
      mbSel.appendChild(opt);
    });

  contSel.addEventListener('change', onContaminantChange);
  mbSel.addEventListener('change', onMembraneChange);
}

function onContaminantChange() {
  const name = document.getElementById('contaminant-select').value;
  if (!name) return;
  const c = APP_DATA.contaminants.find(x => x['Types of contaminants'] === name);
  if (!c) return;

  const smilesEl = document.getElementById('smiles-display');
  if (smilesEl) smilesEl.textContent = c['SMILES'] || 'N/A';
}

function onMembraneChange() {
  // Auto-fill membrane type badge if needed
}

function fmt(val, decimals = 4) {
  if (val === null || val === undefined) return 'N/A';
  if (typeof val === 'number') return Number(val.toFixed(decimals)).toString();
  return String(val);
}

function predict() {
  const contName = document.getElementById('contaminant-select').value;
  const mbName = document.getElementById('membrane-select').value;
  const userPressure = parseFloat(document.getElementById('pressure').value) || null;
  const userTime = parseFloat(document.getElementById('time').value) || null;
  const userInitConc = parseFloat(document.getElementById('init-conc').value) || null;
  const userPh = parseFloat(document.getElementById('ph').value) || null;

  if (!contName && !mbName) {
    alert('Please select at least a contaminant or membrane.');
    return;
  }

  // Get selected profiles
  const contProfile = contName ? APP_DATA.contaminants.find(x => x['Types of contaminants'] === contName) : null;
  const mbProfile = mbName ? APP_DATA.membranes.find(x => x['Name of MB'] === mbName) : null;

  // Normalization denominators for distance calculation
  const NORM = { mw: 1000, size: 1.5, kow: 10, mwco: 500, pore: 1.0, angle: 100, pressure: 2000, ph: 14, time: 1440, conc: 100 };

  // Helper to safely get normalized diff
  const diff = (val1, val2, normScale) => {
    if (val1 === null || val2 === null || val1 === undefined || val2 === undefined) return 0; // Ignore missing features in distance
    return Math.pow((val1 - val2) / normScale, 2);
  };

  // Evaluate similarity of EVERY record against the user's input (K-Nearest Neighbors Algorithm)
  let scoredRecords = APP_DATA.records.map(r => {
    const rCont = APP_DATA.contaminants.find(x => x['Types of contaminants'] === r['Types of contaminants']) || {};
    const rMb = APP_DATA.membranes.find(x => x['Name of MB'] === r['Name of MB']) || {};
    
    let distSq = 0;
    
    // Contaminant distances (only if user selected a contaminant)
    if (contProfile) {
      distSq += diff(contProfile['Compound Mw (g/mol)'], rCont['Compound Mw (g/mol)'], NORM.mw);
      distSq += diff(contProfile['Compound size (nm)'], rCont['Compound size (nm)'], NORM.size);
      distSq += diff(contProfile['Compound log K ow'], rCont['Compound log K ow'], NORM.kow);
    }
    
    // Membrane distances (only if user selected a membrane)
    if (mbProfile) {
      distSq += diff(mbProfile['MB MWCO (Da)'], rMb['MB MWCO (Da)'], NORM.mwco);
      distSq += diff(mbProfile['MB pore radius rp (nm)'], rMb['MB pore radius rp (nm)'], NORM.pore);
      distSq += diff(mbProfile['MB contact angle (°)'], rMb['MB contact angle (°)'], NORM.angle);
    }
    
    // Conditions distances
    if (userPressure !== null) distSq += diff(userPressure, r['Pressure (kPa)'], NORM.pressure);
    if (userPh !== null) distSq += diff(userPh, r['pH'], NORM.ph);
    if (userTime !== null) distSq += diff(userTime, r['Measurement time (min)'], NORM.time);
    if (userInitConc !== null) distSq += diff(userInitConc, r['Initial concentration of compound (mg/L)'], NORM.conc);
    
    // Check if it's an exact identity match for extra weight
    let isExactMatch = true;
    if (contName && r['Types of contaminants'] !== contName) isExactMatch = false;
    if (mbName && r['Name of MB'] !== mbName) isExactMatch = false;

    const distance = Math.sqrt(distSq);
    // Similarity is inversely proportional to distance (max 1.0, minimum near 0)
    const similarity = isExactMatch ? 1.0 : 1.0 / (1.0 + distance * 10);
    
    return { ...r, similarity, isExactMatch };
  });

  // Sort by similarity descending
  scoredRecords.sort((a, b) => b.similarity - a.similarity);

  // Take top 15 closest neighbors
  const matches = scoredRecords.slice(0, 15);

  // Predict Removal Rate using K-Nearest Neighbors weighted average
  let predictedRemoval = null;
  let confidence = 'low';

  const validMatches = matches.filter(m => m['Removal rate (%)'] !== null);
  if (validMatches.length > 0) {
    let totalWeight = 0;
    let weightedSum = 0;
    
    validMatches.forEach(m => {
      // Weight is heavily biased towards highly similar records
      const weight = Math.pow(m.similarity, 2);
      weightedSum += m['Removal rate (%)'] * weight;
      totalWeight += weight;
    });
    
    predictedRemoval = totalWeight > 0 ? (weightedSum / totalWeight) : null;

    // Confidence heuristic
    const topSim = validMatches[0].similarity;
    if (topSim > 0.9) confidence = 'high';
    else if (topSim > 0.5) confidence = 'medium';
    else confidence = 'low';
  }

  renderResults(predictedRemoval, confidence, contProfile, mbProfile, matches);
}

function renderResults(predicted, confidence, contProfile, mbProfile, matches) {
  const area = document.getElementById('results-area');

  let html = '';

  // Hero result
  if (predicted !== null) {
    const confClass = confidence === 'high' ? 'confidence-high' : confidence === 'medium' ? 'confidence-medium' : 'confidence-low';
    const confLabel = confidence === 'high' ? '● High Confidence' : confidence === 'medium' ? '◐ Medium Confidence' : '○ Low Confidence';
    html += `
      <div class="result-hero animate-in">
        <div class="result-value">${predicted.toFixed(1)}%</div>
        <div class="result-label">Predicted Removal Rate</div>
        <div class="result-confidence ${confClass}">${confLabel} · ${matches.length} matches</div>
      </div>`;
  }

  // Contaminant properties card
  if (contProfile) {
    html += `
      <div class="card animate-in" style="animation-delay:0.1s">
        <div class="card-title"><div class="icon">🧪</div> Contaminant Properties</div>
        <div class="data-grid">
          ${dataCell('SMILES', contProfile['SMILES'])}
          ${dataCell('MW (g/mol)', contProfile['Compound Mw (g/mol)'], 2)}
          ${dataCell('Size (nm)', contProfile['Compound size (nm)'], 4)}
          ${dataCell('Stokes Radius (nm)', contProfile['Compound stokes radius rs (nm)'], 4)}
          ${dataCell('Charge', contProfile['Compound charge'], 3)}
          ${dataCell('Log Kow', contProfile['Compound log K ow'], 3)}
          ${dataCell('Density (g/cm³)', contProfile['Density (g·cm-3)'], 4)}
          ${dataCell('pKa1', contProfile['pKa1 '], 2)}
          ${dataCell('pKa2', contProfile['pKa2'], 2)}
          ${dataCell('Water Sol. (mg/L)', contProfile['WS (mg/L)'], 1)}
        </div>
      </div>`;
  }

  // Membrane properties card
  if (mbProfile) {
    html += `
      <div class="card animate-in" style="animation-delay:0.15s">
        <div class="card-title"><div class="icon">🔬</div> Membrane Properties</div>
        <div class="data-grid">
          ${dataCell('Type', mbProfile['Type of MB'])}
          ${dataCell('MWCO (Da)', mbProfile['MB MWCO (Da)'], 0)}
          ${dataCell('Pore Radius (nm)', mbProfile['MB pore radius rp (nm)'], 4)}
          ${dataCell('Contact Angle (°)', mbProfile['MB contact angle (°)'], 1)}
          ${dataCell('Surface Energy (J/m²)', mbProfile['MB surface energy, γm (J·m-2)'], 2)}
          ${dataCell('Zeta Potential (mV)', mbProfile['MB zeta potential (mV)'], 2)}
        </div>
      </div>`;
  }

  // Matching experiments table
  if (matches.length > 0) {
    html += `
      <div class="card animate-in" style="animation-delay:0.2s">
        <div class="card-title"><div class="icon">📊</div> Matching Experimental Records</div>
        <div style="overflow-x:auto;">
          <table class="match-table">
            <thead><tr>
              <th>Contaminant</th><th>Membrane</th><th>Type</th>
              <th>Pressure (kPa)</th><th>pH</th><th>Conc. (mg/L)</th>
              <th>Time (min)</th><th>Removal %</th>
            </tr></thead>
            <tbody>
              ${matches.map(m => {
                const rv = m['Removal rate (%)'];
                const pillClass = rv >= 80 ? 'removal-high' : rv >= 50 ? 'removal-med' : 'removal-low';
                return `<tr>
                  <td>${m['Types of contaminants'] || '-'}</td>
                  <td>${m['Name of MB'] || '-'}</td>
                  <td>${m['Type of MB'] || '-'}</td>
                  <td>${fmt(m['Pressure (kPa)'], 0)}</td>
                  <td>${fmt(m['pH'], 1)}</td>
                  <td>${fmt(m['Initial concentration of compound (mg/L)'], 1)}</td>
                  <td>${fmt(m['Measurement time (min)'], 0)}</td>
                  <td><span class="removal-pill ${pillClass}">${fmt(rv, 1)}%</span></td>
                </tr>`;
              }).join('')}
            </tbody>
          </table>
        </div>
      </div>`;
  }

  if (!html) {
    html = `
      <div class="card empty-state">
        <div class="icon">🔍</div>
        <h3>No matching records found</h3>
        <p>Try a different combination of inputs.</p>
      </div>`;
  }

  area.innerHTML = html;
}

function dataCell(label, value, decimals) {
  const isNA = value === null || value === undefined;
  const display = isNA ? 'N/A' : (typeof value === 'number' ? Number(value.toFixed(decimals ?? 4)).toString() : String(value));
  return `
    <div class="data-cell">
      <div class="label">${label}</div>
      <div class="value ${isNA ? 'na' : ''}">${display}</div>
    </div>`;
}

document.addEventListener('DOMContentLoaded', loadData);
