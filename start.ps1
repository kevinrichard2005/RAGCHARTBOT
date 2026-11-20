# =============================================================================
# RAG Chatbot — One-Command Startup Script (Windows PowerShell)
# Usage: .\start.ps1
# =============================================================================

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Ensure Node.js is on PATH if installed via Playwright/local AppData
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    $LocalNodeDir = "C:\Users\kevin richard D\AppData\Local\ms-playwright-go\1.57.0"
    if (Test-Path "$LocalNodeDir\node.exe") {
        $env:PATH = "$LocalNodeDir;$env:PATH"
        Write-Host "[INFO] Added Node.js ($LocalNodeDir) to PATH" -ForegroundColor Cyan
    }
}

$VenvPython = Join-Path $RootDir "apps\api\.venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    $VenvPython = "python"
}

Write-Host "Starting FastAPI Backend on http://localhost:8000 ..." -ForegroundColor Green
$ApiProc = Start-Process -FilePath $VenvPython `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload" `
    -WorkingDirectory (Join-Path $RootDir "apps\api") `
    -PassThru

Write-Host "Starting Next.js Frontend on http://localhost:3000 ..." -ForegroundColor Green
$NextBin = Join-Path $RootDir "apps\web\node_modules\next\dist\bin\next"
$WebProc = Start-Process -FilePath "node" `
    -ArgumentList $NextBin, "dev", "-p", "3000" `
    -WorkingDirectory (Join-Path $RootDir "apps\web") `
    -PassThru

Write-Host ""
Write-Host "RAG Chatbot is live!" -ForegroundColor Yellow
Write-Host "  -> Frontend UI: http://localhost:3000"
Write-Host "  -> Backend API: http://localhost:8000/docs"
Write-Host "  -> Health Info: http://localhost:8000/health"
Write-Host ""
Write-Host "Press Ctrl+C to stop both servers." -ForegroundColor DarkGray

try {
    Wait-Process -Id $ApiProc.Id, $WebProc.Id
} finally {
    if (-not $ApiProc.HasExited) { Stop-Process -Id $ApiProc.Id -Force -ErrorAction SilentlyContinue }
    if (-not $WebProc.HasExited) { Stop-Process -Id $WebProc.Id -Force -ErrorAction SilentlyContinue }
}
