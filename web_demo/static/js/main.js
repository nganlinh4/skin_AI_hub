// Main JS for the web demo
let selectedFile = null;
let imageBase64 = null;
let reportReader = null;

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const classifyBtn = document.getElementById('classifyBtn');
const generateBtn = document.getElementById('generateBtn');
const reportContent = document.getElementById('reportContent');
const loadingSpinner = document.getElementById('loadingSpinner');
const downloadBtn = document.getElementById('downloadBtn');
const statusMessage = document.getElementById('statusMessage');
const classificationInput = document.getElementById('classification');
const confidenceInput = document.getElementById('confidence');
const classResults = document.getElementById('classificationResults');
const classDetails = document.getElementById('classDetails');

function showError(message) {
  document.getElementById('errorMessage').innerText = message;
  const modal = new bootstrap.Modal(document.getElementById('errorModal'));
  modal.show();
}

function resetUI() {
  reportContent.textContent = '';
  loadingSpinner.classList.add('d-none');
  downloadBtn.classList.add('d-none');
  statusMessage.innerHTML = '<i class="bi bi-info-circle"></i> Ready';
}

function clearReport() {
  reportContent.textContent = '';
  downloadBtn.classList.add('d-none');
  statusMessage.innerHTML = '<i class="bi bi-info-circle"></i> Cleared.';
}

function toBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      // Strip the prefix: data:image/...;base64,
      const result = reader.result;
      const commaIndex = result.indexOf(',');
      resolve(commaIndex >= 0 ? result.substring(commaIndex + 1) : result);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function updatePreview(file) {
  imagePreview.innerHTML = '';
  const img = document.createElement('img');
  img.className = 'img-thumbnail';
  img.style.maxWidth = '100%';
  img.style.maxHeight = '240px';
  img.src = URL.createObjectURL(file);
  imagePreview.appendChild(img);
}

// Drag-and-drop handlers
['dragenter', 'dragover'].forEach(evt => {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add('dragover');
  });
});
['dragleave', 'drop'].forEach(evt => {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('dragover');
  });
});

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('drop', async (e) => {
  const dt = e.dataTransfer;
  if (!dt || !dt.files || dt.files.length === 0) return;
  const file = dt.files[0];
  if (!file.type.startsWith('image/')) {
    showError('Please drop an image file.');
    return;
  }
  selectedFile = file;
  updatePreview(file);
  imageBase64 = await toBase64(file);
  classifyBtn.disabled = false;
  generateBtn.disabled = false;
  statusMessage.innerHTML = '<i class="bi bi-info-circle"></i> Image ready.';
});

fileInput.addEventListener('change', async (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  if (!file.type.startsWith('image/')) {
    showError('Please select an image file.');
    return;
  }
  selectedFile = file;
  updatePreview(file);
  imageBase64 = await toBase64(file);
  classifyBtn.disabled = false;
  generateBtn.disabled = false;
  statusMessage.innerHTML = '<i class="bi bi-info-circle"></i> Image ready.';
});

// Classification
classifyBtn.addEventListener('click', async () => {
  if (!selectedFile) {
    showError('Please upload an image first.');
    return;
  }
  classifyBtn.disabled = true;
  statusMessage.innerHTML = '<i class="bi bi-cpu"></i> Running classification...';
  
  try {
    const formData = new FormData();
    formData.append('file', selectedFile);
    const res = await fetch('/classify', {
      method: 'POST',
      body: formData
    });
    if (!res.ok) throw new Error(`Classification failed: ${res.status}`);
    const data = await res.json();

    if (data.success) {
      classificationInput.value = data.predicted_class;
      confidenceInput.value = (data.confidence * 100).toFixed(1) + '%';
      classDetails.innerHTML = '';
      Object.entries(data.probabilities).sort((a,b) => b[1]-a[1]).forEach(([k,v]) => {
        const pct = (v * 100).toFixed(1);
        classDetails.innerHTML += `<div>${k}: <strong>${pct}%</strong></div>`;
      });
      classResults.classList.remove('d-none');
      statusMessage.innerHTML = '<i class="bi bi-check-circle"></i> Classification complete.';
    } else {
      showError('Classification failed.');
    }
  } catch (err) {
    showError(err.message || 'Classification error.');
  } finally {
    classifyBtn.disabled = false;
  }
});

// Report generation (stream)
generateBtn.addEventListener('click', async () => {
  if (!selectedFile || !imageBase64) {
    showError('Please upload an image first.');
    return;
  }

  // Basic form validation
  const name = document.getElementById('patientName').value.trim();
  const age = parseInt(document.getElementById('patientAge').value, 10);
  const gender = document.getElementById('patientGender').value;
  const history = document.getElementById('medicalHistory').value.trim();
  const classification = classificationInput.value || 'Skin Condition';
  const language = document.getElementById('language').value || 'en';

  if (!name || !age || !gender) {
    showError('Please fill in Patient Name, Age, and Gender.');
    return;
  }

  resetUI();
  loadingSpinner.classList.remove('d-none');
  statusMessage.innerHTML = '<i class="bi bi-hourglass"></i> Generating report...';

  const payload = {
    image_data: imageBase64,
    patient_name: name,
    patient_age: age,
    patient_gender: gender,
    classification: classification,
    history: history,
    language: language,
    comprehensive: true
  };

  try {
    const res = await fetch('/generate_report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error(`Report generation failed: ${res.status}`);

    // Stream the response
    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');

    let buffer = '';
    let pdfPath = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Parse SSE lines (data: {...})
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep last partial line
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const jsonStr = line.slice(6);
        try {
          const msg = JSON.parse(jsonStr);
          if (msg.type === 'token') {
            // token may contain embedded files marker
            const token = msg.token || '';
            const filesMatch = token.match(/\[FILES\](.*)\[\/FILES\]/);
            if (filesMatch) {
              try {
                const filesObj = JSON.parse(filesMatch[1]);
                if (filesObj.pdf_report) {
                  pdfPath = filesObj.pdf_report; // like /api/v1/reports/xxx.pdf
                }
              } catch (e) { /* ignore */ }
            }
            reportContent.textContent += token;
            reportContent.scrollTop = reportContent.scrollHeight;
          } else if (msg.type === 'complete') {
            loadingSpinner.classList.add('d-none');
            statusMessage.innerHTML = '<i class="bi bi-check-circle"></i> Report generation complete.';
            if (pdfPath) {
              downloadBtn.classList.remove('d-none');
              downloadBtn.onclick = () => {
                const url = '/download_pdf?path=' + encodeURIComponent(pdfPath);
                window.open(url, '_blank');
              };
            }
          } else if (msg.type === 'error') {
            loadingSpinner.classList.add('d-none');
            showError(msg.error || 'Streaming error');
          }
        } catch (e) {
          // ignore parse errors
        }
      }
    }

    // Ensure complete state if not already
    loadingSpinner.classList.add('d-none');
    statusMessage.innerHTML = '<i class="bi bi-check-circle"></i> Report generation complete.';
    if (pdfPath) {
      downloadBtn.classList.remove('d-none');
      downloadBtn.onclick = () => {
        const url = '/download_pdf?path=' + encodeURIComponent(pdfPath);
        window.open(url, '_blank');
      };
    }

  } catch (err) {
    loadingSpinner.classList.add('d-none');
    showError(err.message || 'Failed to generate report.');
  }
});

