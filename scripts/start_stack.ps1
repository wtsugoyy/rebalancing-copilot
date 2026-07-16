# Start the full offline Rebalancing Copilot stack.
#
#   Native (host): Ollama (GPU via ROCm) + Supermemory
#   Docker:        app + Langfuse (server + postgres)
#
# Ollama must run natively: Docker on Windows cannot pass through an AMD GPU.
# Supermemory must run in a Linux container: its installer does not support Windows.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Write-Host "==> 1/4 Ollama (native, GPU)" -ForegroundColor Cyan
$env:OLLAMA_HOST = "0.0.0.0:11434"   # bind all ifaces so containers can reach it
if (-not (Get-Process -Name "ollama" -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" `
                  -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 6
}
try { Invoke-WebRequest http://localhost:11434 -UseBasicParsing -TimeoutSec 5 | Out-Null
      Write-Host "    ok - ollama serving" -ForegroundColor Green }
catch { Write-Host "    FAILED to start ollama" -ForegroundColor Red; exit 1 }

Write-Host "==> 2/4 Supermemory (container, local embeddings, no egress)" -ForegroundColor Cyan
if (-not (docker ps -q -f name=supermemory)) {
    docker rm -f supermemory 2>$null | Out-Null
    docker run -d --name supermemory -p 8787:8787 `
        --add-host=host.docker.internal:host-gateway `
        -e OPENAI_BASE_URL=http://host.docker.internal:11434/v1 `
        -e OPENAI_API_KEY=ollama -e OPENAI_MODEL=qwen2.5:3b -e PORT=8787 `
        -v supermemory-data:/.supermemory `
        node:22-slim sh -c "apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null 2>&1 && npx -y supermemory local start --port 8787" | Out-Null
    Write-Host "    starting (first boot downloads the embedding model)..." -ForegroundColor Yellow
}
Write-Host "    ok - supermemory on :8787" -ForegroundColor Green

Write-Host "==> 3/4 Docker stack (app + langfuse + postgres)" -ForegroundColor Cyan
Push-Location $root
docker compose up -d --build
Pop-Location

Write-Host "==> 4/4 Ready" -ForegroundColor Cyan
Write-Host "    Dashboard : http://localhost:8501" -ForegroundColor Green
Write-Host "    Langfuse  : http://localhost:3000  (admin@copilot.local / copilot-demo-1234)" -ForegroundColor Green
Write-Host "    Supermemory: http://localhost:8787" -ForegroundColor Green
