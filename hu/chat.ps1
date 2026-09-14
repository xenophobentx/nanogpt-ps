# Szövegfolytató felület indítása egy betanított modellel.
#   .\chat.ps1 -MaxTokens 1
#   .\chat.ps1 -WeightsFile .\sajat-weights.json
# Szöveg beírása után a /step 1 megmutatja a következő karakter választását.
# /help: parancsok. /reset: kontextus törlése. /q: kilépés.
# A beírt szöveg a kontextushoz kerül; a modell súlyai nem változnak.
param(
    [string]$Model = 'shakespeare',
    [string]$WeightsFile = '',
    [double]$Temperature = 0.8,
    [int]$MaxTokens = 1000,
    [int]$TopK = 0,
    [ValidateSet('hu', 'en')][string]$Lang = 'hu'
)
if ($WeightsFile -eq '') { $WeightsFile = Join-Path $PSScriptRoot "nanogpt-$Model-weights.json" }
if (-not (Test-Path $WeightsFile)) {
    Write-Host "Nincs ilyen sulyfajl: $WeightsFile" -ForegroundColor Red
    Write-Host "Elerheto modellek:" -ForegroundColor Yellow
    Get-ChildItem (Join-Path $PSScriptRoot 'nanogpt-*-weights.json') | ForEach-Object { '  -Model ' + ($_.BaseName -replace '^nanogpt-', '' -replace '-weights$', '') }
    exit 1
}
& (Join-Path $PSScriptRoot 'nanogpt-ps.ps1') -Chat -WeightsFile $WeightsFile -Temperature $Temperature -MaxTokens $MaxTokens -TopK $TopK -Lang $Lang
