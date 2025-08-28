# Test script for Skin Diagnosis API

Write-Host "Testing Skin Diagnosis API..." -ForegroundColor Green

# Test 1: Quick Analysis (Non-streaming)
Write-Host "`nTest 1: Testing /analyze endpoint (Quick Analysis)..." -ForegroundColor Yellow

$boundary = [System.Guid]::NewGuid().ToString()
$filePath = "C:\work\skin_AI_hub\skin-diagnosis-generation\test.jpg"
$fileBytes = [System.IO.File]::ReadAllBytes($filePath)
$fileEnc = [System.Text.Encoding]::GetEncoding('iso-8859-1').GetString($fileBytes)

$bodyLines = (
    "--$boundary",
    'Content-Disposition: form-data; name="image"; filename="test.jpg"',
    'Content-Type: image/jpeg',
    '',
    $fileEnc,
    "--$boundary",
    'Content-Disposition: form-data; name="patient_name"',
    '',
    'Susan',
    "--$boundary",
    'Content-Disposition: form-data; name="patient_age"',
    '',
    '38',
    "--$boundary",
    'Content-Disposition: form-data; name="patient_gender"',
    '',
    'Female',
    "--$boundary",
    'Content-Disposition: form-data; name="classification"',
    '',
    'Infectious Disease - Folliculitis',
    "--$boundary",
    'Content-Disposition: form-data; name="history"',
    '',
    'Patient has a history of recurrent folliculitis episodes, particularly during summer months.',
    "--$boundary",
    'Content-Disposition: form-data; name="language"',
    '',
    'en',
    "--$boundary--"
) -join "`r`n"

$bodyBytes = [System.Text.Encoding]::GetEncoding('iso-8859-1').GetBytes($bodyLines)

try {
    $response = Invoke-WebRequest -Uri "http://localhost:3026/api/v1/analyze" `
        -Method POST `
        -ContentType "multipart/form-data; boundary=$boundary" `
        -Body $bodyBytes
    
    $result = $response.Content | ConvertFrom-Json
    Write-Host "Success! Quick Analysis Result:" -ForegroundColor Green
    Write-Host "Patient Name: $($result.patient_name)"
    Write-Host "Analysis: $($result.concise_analysis)"
    Write-Host "Generation Time: $($result.generation_time) seconds"
    if ($result.files.analysis_text) {
        Write-Host "Analysis File: $($result.files.analysis_text)"
    }
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}

# Test 2: Check Health Endpoint
Write-Host "`nTest 2: Testing /health endpoint..." -ForegroundColor Yellow
try {
    $health = Invoke-WebRequest -Uri "http://localhost:3026/health" -Method GET
    Write-Host "Health Check: $($health.Content)" -ForegroundColor Green
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}

# Test 3: List Models
Write-Host "`nTest 3: Testing /models endpoint..." -ForegroundColor Yellow
try {
    $models = Invoke-WebRequest -Uri "http://localhost:3026/api/v1/models" -Method GET
    $modelsData = $models.Content | ConvertFrom-Json
    Write-Host "Available Models:" -ForegroundColor Green
    $modelsData.available_models | ForEach-Object {
        Write-Host "  - $($_.name): $($_.description)"
    }
    Write-Host "Current Model: $($modelsData.current_model)"
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}

Write-Host "`nAll tests completed!" -ForegroundColor Green
