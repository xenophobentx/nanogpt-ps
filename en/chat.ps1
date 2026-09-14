# Open the English text continuation interface with a trained model.
# In the repository this launcher is chat-en.ps1; packages ship it as chat.ps1.
# Package examples:
#   .\chat.ps1 -MaxTokens 1
#   .\chat.ps1 -WeightsFile .\my-weights.json
# After entering text, use /step 1 to inspect the next character selection.
# /help lists commands. /reset clears context. /q exits.
# The model continues text; entering a line does not update its weights.
param(
    [string]$Model = 'shakespeare',
    [string]$WeightsFile = '',
    [double]$Temperature = 0.8,
    [int]$MaxTokens = 1000,
    [int]$TopK = 0
)
if ($WeightsFile -eq '') { $WeightsFile = Join-Path $PSScriptRoot "nanogpt-$Model-weights.json" }
if (-not (Test-Path $WeightsFile)) {
    Write-Host "No such weights file: $WeightsFile" -ForegroundColor Red
    Write-Host "Available models:" -ForegroundColor Yellow
    Get-ChildItem (Join-Path $PSScriptRoot 'nanogpt-*-weights.json') | ForEach-Object { '  -Model ' + ($_.BaseName -replace '^nanogpt-', '' -replace '-weights$', '') }
    exit 1
}
& (Join-Path $PSScriptRoot 'nanogpt-ps.ps1') -Chat -Lang en -WeightsFile $WeightsFile -Temperature $Temperature -MaxTokens $MaxTokens -TopK $TopK
