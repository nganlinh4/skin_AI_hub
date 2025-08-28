# Comprehensive Test Script for Skin Diagnosis API
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "COMPREHENSIVE API TEST SUITE" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Test 4: Full Diagnosis (Non-streaming)
Write-Host "Test 4: Testing /diagnose endpoint (Full Diagnosis with Reports)..." -ForegroundColor Yellow

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
    'John Doe',
    "--$boundary",
    'Content-Disposition: form-data; name="patient_age"',
    '',
    '45',
    "--$boundary",
    'Content-Disposition: form-data; name="patient_gender"',
    '',
    'Male',
    "--$boundary",
    'Content-Disposition: form-data; name="classification"',
    '',
    'Dermatitis',
    "--$boundary",
    'Content-Disposition: form-data; name="history"',
    '',
    'Patient has sensitive skin with occasional eczema flare-ups.',
    "--$boundary",
    'Content-Disposition: form-data; name="language"',
    '',
    'en',
    "--$boundary--"
) -join "`r`n"

$bodyBytes = [System.Text.Encoding]::GetEncoding('iso-8859-1').GetBytes($bodyLines)

try {
    Write-Host "Generating full diagnosis (this may take a moment)..." -ForegroundColor Gray
    $response = Invoke-WebRequest -Uri "http://localhost:3026/api/v1/diagnose" `
        -Method POST `
        -ContentType "multipart/form-data; boundary=$boundary" `
        -Body $bodyBytes
    
    $result = $response.Content | ConvertFrom-Json
    Write-Host "Success! Full Diagnosis Generated:" -ForegroundColor Green
    Write-Host "  Patient: $($result.patient_name)"
    Write-Host "  Concise Analysis: $($result.analysis.concise_analysis)"
    Write-Host "  Generation Time: $($result.analysis.generation_time) seconds"
    Write-Host "  Files Generated:"
    Write-Host "    - Markdown: $($result.files.markdown_report)"
    Write-Host "    - PDF: $($result.files.pdf_report)"
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}

# Test 5: Streaming Analysis with Base64
Write-Host "`nTest 5: Testing /analyze-stream-simple endpoint (Streaming with Base64)..." -ForegroundColor Yellow

# Convert image to base64
$imageBytes = [System.IO.File]::ReadAllBytes($filePath)
$imageBase64 = [System.Convert]::ToBase64String($imageBytes)

$requestBody = @{
    image_data = $imageBase64
    filename = "test.jpg"
    patient_name = "Alice Smith"
    patient_age = 32
    patient_gender = "Female"
    classification = "Skin Condition"
    history = "No significant medical history"
    language = "en"
} | ConvertTo-Json

try {
    Write-Host "Initiating streaming analysis..." -ForegroundColor Gray
    
    # For streaming endpoints, we need to handle Server-Sent Events
    $headers = @{
        "Content-Type" = "application/json"
        "Accept" = "text/event-stream"
    }
    
    $response = Invoke-WebRequest -Uri "http://localhost:3026/api/v1/analyze-stream-simple" `
        -Method POST `
        -Headers $headers `
        -Body $requestBody
    
    Write-Host "Streaming Response Received:" -ForegroundColor Green
    # Parse SSE data
    $lines = $response.Content -split "`n"
    $fullText = ""
    foreach ($line in $lines) {
        if ($line -match "^data: (.+)") {
            try {
                $eventData = $Matches[1] | ConvertFrom-Json
                if ($eventData.type -eq "complete") {
                    $fullText = $eventData.full_text
                    Write-Host "  Complete Analysis: $fullText" -ForegroundColor Green
                    if ($eventData.files.analysis_text) {
                        Write-Host "  Analysis File: $($eventData.files.analysis_text)" -ForegroundColor Green
                    }
                }
            } catch {}
        }
    }
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}

# Test 6: Vietnamese Language Test
Write-Host "`nTest 6: Testing Vietnamese Language Support..." -ForegroundColor Yellow

$bodyLinesVi = (
    "--$boundary",
    'Content-Disposition: form-data; name="image"; filename="test.jpg"',
    'Content-Type: image/jpeg',
    '',
    $fileEnc,
    "--$boundary",
    'Content-Disposition: form-data; name="patient_name"',
    '',
    'Nguyen Van A',
    "--$boundary",
    'Content-Disposition: form-data; name="patient_age"',
    '',
    '35',
    "--$boundary",
    'Content-Disposition: form-data; name="patient_gender"',
    '',
    'Nam',
    "--$boundary",
    'Content-Disposition: form-data; name="classification"',
    '',
    'Viêm nang lông',
    "--$boundary",
    'Content-Disposition: form-data; name="history"',
    '',
    'Bệnh nhân có tiền sử viêm nang lông tái phát',
    "--$boundary",
    'Content-Disposition: form-data; name="language"',
    '',
    'vi',
    "--$boundary--"
) -join "`r`n"

$bodyBytesVi = [System.Text.Encoding]::GetEncoding('utf-8').GetBytes($bodyLinesVi)

try {
    $response = Invoke-WebRequest -Uri "http://localhost:3026/api/v1/analyze" `
        -Method POST `
        -ContentType "multipart/form-data; boundary=$boundary" `
        -Body $bodyBytesVi
    
    $result = $response.Content | ConvertFrom-Json
    Write-Host "Success! Vietnamese Analysis:" -ForegroundColor Green
    Write-Host "  Patient: $($result.patient_name)"
    Write-Host "  Analysis: $($result.concise_analysis)"
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "ALL COMPREHENSIVE TESTS COMPLETED!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
