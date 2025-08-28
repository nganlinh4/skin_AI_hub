# Kill any existing process using port 3026
Write-Host "Checking for existing processes on port 3026..." -ForegroundColor Yellow

# Find process using port 3026
$connection = Get-NetTCPConnection -LocalPort 3026 -ErrorAction SilentlyContinue
if ($connection) {
    $processId = $connection.OwningProcess | Select-Object -Unique
    foreach ($pid in $processId) {
        $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
        if ($process) {
            Write-Host "Found process '$($process.ProcessName)' (PID: $pid) using port 3026" -ForegroundColor Yellow
            Write-Host "Stopping process..." -ForegroundColor Yellow
            Stop-Process -Id $pid -Force
            Write-Host "Process stopped successfully" -ForegroundColor Green
            Start-Sleep -Seconds 1
        }
    }
} else {
    Write-Host "No process found on port 3026" -ForegroundColor Green
}

# Start the VLM server
Write-Host "`nStarting VLM server on port 3026..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray

# Run uvicorn
python -m uvicorn app.main:app --host 0.0.0.0 --port 3026
