param([ValidateSet('All', 'GitHub', 'HuggingFace', 'Preview')][string]$Target = 'All',
      [ValidateSet('Direct', 'PullRequest')][string]$ModelPublishMode = 'PullRequest')
$ErrorActionPreference = 'Stop'
try {
    $main = Join-Path $PSScriptRoot 'publish_erm_ai.ps1'
    $tokens = $null
    $parseErrors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile($main, [ref]$tokens, [ref]$parseErrors)
    if ($parseErrors.Count -gt 0) {
        foreach ($problem in $parseErrors) { Write-Host $problem.ToString() -ForegroundColor Red }
        throw 'Publisher syntax check failed. No publishing commands were run.'
    }
    & $main -Target $Target -ModelPublishMode $ModelPublishMode
    exit 0
} catch {
    Write-Host ''
    Write-Host ('STOPPED: ' + $_.Exception.Message) -ForegroundColor Red
    Write-Host 'Any already completed remote uploads are kept. You can retry only the unfinished service.'
    exit 1
}
