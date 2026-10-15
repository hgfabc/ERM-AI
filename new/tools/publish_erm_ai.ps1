param([ValidateSet('All', 'GitHub', 'HuggingFace', 'Preview')][string]$Target = 'All',
      [ValidateSet('Direct', 'PullRequest')][string]$ModelPublishMode = 'PullRequest',
      [switch]$DefinitionsOnly)
# ASCII source intentionally supports Windows PowerShell 5.1 without an encoding dependency.
$ErrorActionPreference = 'Stop'
$env:GH_DEBUG = $null
$env:HF_HUB_DEBUG = $null
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$CodeRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$WorkspaceRoot = [IO.Path]::GetFullPath((Join-Path $CodeRoot '..'))
$WorkRoot = Join-Path $CodeRoot '.publish-work'
$SessionRoot = Join-Path $WorkRoot ((Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
$UpstreamRepo = 'ntuhophai/ERM-AI'
$ModelRepo = 'ntuhophai/ERM-AI'
$script:Gh = $null
$script:Python = $null
$script:PublishedCodeUrl = 'https://github.com/ntuhophai/ERM-AI'
$Utf8 = New-Object System.Text.UTF8Encoding($false)
New-Item -ItemType Directory -Path $SessionRoot -Force | Out-Null

function Save-Json([string]$Path, $Value) {
    [IO.File]::WriteAllText($Path, (ConvertTo-Json -InputObject $Value -Depth 100), $Utf8)
}

function Invoke-Native([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    $result = $LASTEXITCODE
    if ($result -ne 0) { throw ('Command failed with exit code ' + $result + ': ' + [IO.Path]::GetFileName($Exe)) }
}

function Test-Native([string]$Exe, [string[]]$Arguments) {
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Exe @Arguments | Out-Host
        $result = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previousPreference }
    return ($result -eq 0)
}

function Get-Gh {
    $existing = Get-Command gh -ErrorAction SilentlyContinue
    if ($existing) { return $existing.Source }
    Write-Host 'GitHub CLI is missing. Downloading its official portable Windows release into .publish-work/tools ...'
    $headers = @{ 'User-Agent' = 'ERM-AI-Publisher'; 'Accept' = 'application/vnd.github+json' }
    $release = Invoke-RestMethod -Uri 'https://api.github.com/repos/cli/cli/releases/latest' -Headers $headers
    if ($release.tag_name -notmatch '^v[0-9]+(\.[0-9]+){1,3}(-[A-Za-z0-9.-]+)?$') { throw 'Unexpected GitHub CLI release tag.' }
    $architecture = 'amd64'
    if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { $architecture = 'arm64' }
    elseif (-not [Environment]::Is64BitOperatingSystem) { $architecture = '386' }
    $pattern = '^gh_[0-9.]+_windows_' + $architecture + '\.zip$'
    $assets = @($release.assets | Where-Object { $_.name -match $pattern })
    if ($assets.Count -ne 1) { throw 'Cannot identify one official GitHub CLI Windows asset.' }
    $asset = $assets[0]
    if (-not $asset.browser_download_url.StartsWith('https://github.com/cli/cli/releases/')) { throw 'Unexpected CLI download source.' }
    $cache = Join-Path $WorkRoot ('tools/' + $release.tag_name)
    New-Item -ItemType Directory -Path $cache -Force | Out-Null
    $zip = Join-Path $cache $asset.name
    $expected = $null
    if ($asset.digest -match '^sha256:([a-fA-F0-9]{64})$') { $expected = $Matches[1] }
    else {
        $checksums = @($release.assets | Where-Object { $_.name -match '_checksums\.txt$' })
        if ($checksums.Count -ne 1 -or -not $checksums[0].browser_download_url.StartsWith('https://github.com/cli/cli/releases/')) {
            throw 'Official CLI checksum is unavailable; download was not executed.'
        }
        $checksumPath = Join-Path $cache 'checksums.txt'
        Invoke-WebRequest -UseBasicParsing -Uri $checksums[0].browser_download_url -OutFile $checksumPath
        foreach ($line in [IO.File]::ReadAllLines($checksumPath)) {
            if ($line -match '^([a-fA-F0-9]{64})\s+\*?(.+)$' -and $Matches[2].Trim() -eq $asset.name) { $expected = $Matches[1]; break }
        }
    }
    if (-not $expected) { throw 'GitHub CLI checksum is unavailable.' }
    $cachedValid = (Test-Path -LiteralPath $zip -PathType Leaf) -and (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash -eq $expected
    if (-not $cachedValid) { Invoke-WebRequest -UseBasicParsing -Uri $asset.browser_download_url -OutFile $zip }
    if ((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash -ne $expected) { throw 'GitHub CLI checksum verification failed.' }
    $expanded = Join-Path $cache ('verified-' + [guid]::NewGuid().ToString('N'))
    Expand-Archive -LiteralPath $zip -DestinationPath $expanded
    $binaries = @(Get-ChildItem -LiteralPath $expanded -Recurse -File -Filter 'gh.exe')
    if ($binaries.Count -ne 1) { throw 'Cannot find one gh.exe in the verified archive.' }
    return $binaries[0].FullName
}

function Get-Python {
    $local = Join-Path $CodeRoot '.venv/Scripts/python.exe'
    if (Test-Path -LiteralPath $local -PathType Leaf) { return $local }
    $found = Get-Command python -ErrorAction SilentlyContinue
    if (-not $found) { throw 'Python 3.10+ is required. Restore new/.venv or install Python before retrying.' }
    Invoke-Native $found.Source @('-c', "import sys; assert sys.version_info >= (3,10), 'Python 3.10+ required'") | Out-Host
    $runtime = Join-Path $WorkRoot 'runtime'
    $runtimePython = Join-Path $runtime 'Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $runtimePython -PathType Leaf)) { Invoke-Native $found.Source @('-m', 'venv', $runtime) | Out-Host }
    Write-Host 'Preparing an isolated runtime. The first setup can download PyTorch and take several minutes.'
    Invoke-Native $runtimePython @('-m', 'pip', 'install', '-r', (Join-Path $CodeRoot 'requirements-dev.txt'), 'huggingface_hub', 'safetensors') | Out-Host
    return $runtimePython
}

function Ensure-PythonPackages {
    $script:Python = Get-Python
    Invoke-Native $script:Python @('-c', "import sys; assert sys.version_info >= (3,10), 'Python 3.10+ required'")
    if (-not (Test-Native $script:Python @('-c', "import torch,timm,yaml,pytest,sklearn,safetensors,huggingface_hub,inspect; from huggingface_hub import HfApi; assert 'write' in inspect.signature(HfApi.auth_check).parameters, 'Update huggingface_hub for write-access checks'; assert tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]) >= (2,6), 'PyTorch 2.6+ required'"))) {
        Write-Host 'Installing required packages into the selected local virtual environment ...'
        Invoke-Native $script:Python @('-m', 'pip', 'install', '-r', (Join-Path $CodeRoot 'requirements-dev.txt'), 'huggingface_hub', 'safetensors')
    }
}

function Assert-PublicText([string]$Text, [string]$Name) {
    if ($Text -match '(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----)') {
        throw ('Credential-like content found in ' + $Name + '. Nothing from this package will be published.')
    }
    foreach ($privatePath in @($CodeRoot, $WorkspaceRoot, $env:USERPROFILE)) {
        if ($privatePath -and ($Text.Contains($privatePath) -or $Text.Contains($privatePath.Replace('\', '/')))) {
            throw ('An actual local path was found in ' + $Name + '. Remove it from the public file first.')
        }
    }
}

function Get-PublicFiles {
    $selected = New-Object System.Collections.Generic.List[object]
    $top = @('main.py', 'README.md', '.gitignore', 'requirements.txt', 'requirements-dev.txt', 'requirements-tested.txt',
             'PUBLISHING.md', 'INFERENCE.md', 'MODEL_CATALOG.md', 'publish_erm_ai.bat', 'publish_github.bat', 'publish_huggingface.bat', 'publish_preview.bat')
    foreach ($required in @('main.py', 'README.md', '.gitignore', 'requirements.txt', 'requirements-dev.txt', 'PUBLISHING.md')) {
        if (-not (Test-Path -LiteralPath (Join-Path $CodeRoot $required) -PathType Leaf)) { throw ('Required public source file is missing: ' + $required) }
    }
    foreach ($relative in $top) {
        $path = Join-Path $CodeRoot $relative
        if (Test-Path -LiteralPath $path -PathType Leaf) { $selected.Add((Get-Item -LiteralPath $path)) }
    }
    foreach ($directory in @('multistream', 'tests', 'configs', 'publish_profiles', 'tools')) {
        $path = Join-Path $CodeRoot $directory
        $folder = Get-Item -LiteralPath $path
        if ($folder.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw ('Linked source directory is not allowed: ' + $directory) }
        foreach ($file in Get-ChildItem -LiteralPath $path -File) {
            $extensions = if ($directory -in @('configs', 'publish_profiles')) { @('.yaml', '.yml') } elseif ($directory -eq 'tools') { @('.py', '.ps1') } else { @('.py') }
            if ($file.Extension.ToLowerInvariant() -in $extensions) { $selected.Add($file) }
        }
    }
    $result = New-Object System.Collections.Generic.List[object]
    foreach ($file in $selected) {
        if ($file.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Linked source files are not allowed.' }
        $full = [IO.Path]::GetFullPath($file.FullName)
        if (-not $full.StartsWith($CodeRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Source path escaped new/.' }
        $relative = $full.Substring($CodeRoot.Length + 1).Replace('\', '/')
        if ($relative -notmatch '^[A-Za-z0-9_./-]+$') { throw ('Rename the public file to an ASCII filename: ' + $relative) }
        if ($file.Length -gt 1MB) { throw ('Unexpectedly large source file: ' + $relative) }
        $text = [IO.File]::ReadAllText($full, [Text.Encoding]::UTF8)
        if ($relative -eq 'README.md') {
            $text = [regex]::Replace($text, '\[([^\]]+)\]\(\.\./old/[^)]+\)', '$1 (local archive; not uploaded)')
            $text = "# Public source snapshot`n`nLegacy files, datasets, private results and local environments are not included in this repository.`n`n" + $text
        }
        Assert-PublicText $text $relative
        $result.Add([pscustomobject]@{ path = ('new/' + $relative); bytes = $Utf8.GetBytes($text) })
    }
    $readme = @'
# ERM-AI

Single-stream and multistream OCT research code lives in [`new/`](new/).

- [Usage and experiment workflow](new/README.md)
- [Windows publishing launcher](new/PUBLISHING.md)
- [Published model download and inference](new/INFERENCE.md)
- [All stream/fold models and model architecture](new/MODEL_CATALOG.md)
- Model repository: https://huggingface.co/ntuhophai/ERM-AI

Windows model publishing defaults to one Hugging Face pull request for the complete collection.
The repository maintainer reviews and merges it; the launcher never merges the PR.

```powershell
cd new
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe main.py --help
```

Use Python 3.10+ and install a matching PyTorch/torchvision pair for your CPU or CUDA environment.
Real datasets, old checkpoints, patient manifests, predictions and local environments are excluded.
Synthetic smoke tests validate the software workflow and do not establish clinical model performance.
The original artifacts remain in the source computer's local `old/` archive.
'@
    $result.Add([pscustomobject]@{ path = 'README.md'; bytes = $Utf8.GetBytes($readme) })
    $ignore = @'
old/
audit/
archive_manifest.json
**/.venv/
**/.cache/
**/.publish-work/
**/__pycache__/
**/.pytest_cache/
**/runs/
**/artifacts/
**/smoke_output/
**/verification.json
*.pth
*.pt
*.safetensors
'@
    $result.Add([pscustomobject]@{ path = '.gitignore'; bytes = $Utf8.GetBytes($ignore) })
    $paths = @($result | ForEach-Object { $_.path })
    if (@($paths | Select-Object -Unique).Count -ne $paths.Count) { throw 'Duplicate public paths.' }
    return $result.ToArray()
}

function Invoke-GhJson([string]$Method, [string]$Endpoint, $Body = $null, [switch]$Allow404) {
    $arguments = @('api', '--hostname', 'github.com', '--method', $Method, $Endpoint)
    if ($null -ne $Body) {
        $inputPath = Join-Path $SessionRoot ([guid]::NewGuid().ToString('N') + '.json')
        Save-Json $inputPath $Body
        $arguments += @('--input', $inputPath)
    }
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $output = @(& $script:Gh @arguments 2>&1)
        $result = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previousPreference }
    $text = ($output | ForEach-Object { $_.ToString() }) -join "`n"
    if ($result -ne 0) {
        if ($Allow404 -and $text -match 'HTTP\s*404') { return $null }
        throw ('GitHub API failed: ' + $Method + ' ' + $Endpoint + "`n" + $text)
    }
    return ($text | ConvertFrom-Json)
}

function Assert-Fork($Fork, $User, $Upstream) {
    if (-not $Fork -or $Fork.owner.login -ne $User.login -or -not $Fork.fork -or
        $Fork.parent.full_name -ne $UpstreamRepo -or [string]$Fork.parent.id -ne [string]$Upstream.id) {
        throw 'The personal ERM-AI repository is not the expected fork. No source files were uploaded.'
    }
}

function Get-ReusableBranch($User, [string]$ForkRepo) {
    $receiptPath = Join-Path $WorkRoot 'last-github.json'
    if (-not (Test-Path -LiteralPath $receiptPath -PathType Leaf)) { return $null }
    try { $previous = [IO.File]::ReadAllText($receiptPath, [Text.Encoding]::UTF8) | ConvertFrom-Json }
    catch { return $null }
    if ($previous.repo -ne $ForkRepo -or $previous.branch -notmatch '^publish/new-code-[0-9]{8}-[0-9]{6}-[a-f0-9]{8}$' -or $previous.commit -notmatch '^[a-f0-9]{40}$') { return $null }
    $ref = Invoke-GhJson 'GET' ('repos/' + $ForkRepo + '/git/ref/heads/' + $previous.branch) -Allow404
    if (-not $ref -or $ref.object.sha -ne $previous.commit) {
        Write-Host 'Previous branch is missing or has edits outside this publisher. A new branch will preserve those edits.'
        return $null
    }
    $head = [Uri]::EscapeDataString($User.login + ':' + $previous.branch)
    $pulls = @(Invoke-GhJson 'GET' ('repos/' + $UpstreamRepo + '/pulls?state=all&head=' + $head))
    if (@($pulls | Where-Object { $_ -and $_.state -eq 'closed' }).Count -gt 0 -and @($pulls | Where-Object { $_ -and $_.state -eq 'open' }).Count -eq 0) {
        Write-Host 'Previous PR is closed or merged. A new publication branch will be created.'
        return $null
    }
    return $previous
}

function Publish-GitHub($Files) {
    Write-Host 'Running the local test suite before publishing source ...' -ForegroundColor Cyan
    $env:OMP_NUM_THREADS = '2'
    $env:MKL_NUM_THREADS = '2'
    Invoke-Native $script:Python @('-m', 'pytest', (Join-Path $CodeRoot 'tests'), '-q')
    $script:Gh = Get-Gh
    if (-not (Test-Native $script:Gh @('auth', 'status', '--hostname', 'github.com', '--active'))) {
        Invoke-Native $script:Gh @('auth', 'login', '--hostname', 'github.com', '--git-protocol', 'https', '--web')
    }
    $user = Invoke-GhJson 'GET' 'user'
    if (-not $user.login -or $user.login -notmatch '^[A-Za-z0-9-]+$') { throw 'Unexpected GitHub account name.' }
    if ($user.login -eq 'ntuhophai') { throw 'The active GitHub account owns the upstream. Sign in with your personal account to create a fork.' }
    $upstream = Invoke-GhJson 'GET' ('repos/' + $UpstreamRepo)
    $forkRepo = $user.login + '/ERM-AI'
    Write-Host ('Source target: ' + $forkRepo + ' (publication branch; no automatic PR or merge)')
    $fork = Invoke-GhJson 'GET' ('repos/' + $forkRepo) -Allow404
    if ($null -eq $fork) {
        Invoke-Native $script:Gh @('repo', 'fork', $UpstreamRepo, '--clone=false', '--remote=false')
        for ($attempt = 0; $attempt -lt 12; $attempt++) {
            $fork = Invoke-GhJson 'GET' ('repos/' + $forkRepo) -Allow404
            if ($fork) { break }
            Start-Sleep -Seconds 3
        }
    }
    Assert-Fork $fork $user $upstream
    if (-not $fork.permissions.push) { throw 'The local GitHub login cannot write to its fork. Check gh login scopes.' }
    $baseBranch = $upstream.default_branch
    $baseRef = Invoke-GhJson 'GET' ('repos/' + $UpstreamRepo + '/git/ref/heads/' + $baseBranch)
    $baseCommit = Invoke-GhJson 'GET' ('repos/' + $UpstreamRepo + '/git/commits/' + $baseRef.object.sha)
    $reusable = Get-ReusableBranch $user $forkRepo
    if ($reusable) {
        $baseCommit = Invoke-GhJson 'GET' ('repos/' + $forkRepo + '/git/commits/' + $reusable.commit)
        Write-Host ('Updating the existing publication branch: ' + $reusable.branch)
    }
    $entries = New-Object System.Collections.Generic.List[object]
    foreach ($file in $Files) {
        Write-Host ('Uploading source: ' + $file.path)
        $blob = Invoke-GhJson 'POST' ('repos/' + $forkRepo + '/git/blobs') @{ encoding = 'base64'; content = [Convert]::ToBase64String($file.bytes) }
        $entries.Add(@{ path = $file.path; mode = '100644'; type = 'blob'; sha = $blob.sha })
    }
    $tree = Invoke-GhJson 'POST' ('repos/' + $forkRepo + '/git/trees') @{ base_tree = $baseCommit.tree.sha; tree = $entries.ToArray() }
    $author = @{ name = $user.login; email = ([string]$user.id + '+' + $user.login + '@users.noreply.github.com') }
    $commit = Invoke-GhJson 'POST' ('repos/' + $forkRepo + '/git/commits') @{
        message = 'Add complete model collection publishing and inference documentation';
        tree = $tree.sha; parents = @($baseCommit.sha); author = $author; committer = $author
    }
    if ($reusable) {
        $branch = $reusable.branch
        [void](Invoke-GhJson 'PATCH' ('repos/' + $forkRepo + '/git/refs/heads/' + $branch) @{ sha = $commit.sha; force = $false })
    } else {
        $branch = 'publish/new-code-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
        [void](Invoke-GhJson 'POST' ('repos/' + $forkRepo + '/git/refs') @{ ref = ('refs/heads/' + $branch); sha = $commit.sha })
    }
    $compare = 'https://github.com/' + $UpstreamRepo + '/compare/' + [Uri]::EscapeDataString($baseBranch) + '...' + $user.login + ':' + $branch + '?expand=1'
    $script:PublishedCodeUrl = 'https://github.com/' + $forkRepo + '/tree/' + $commit.sha + '/new'
    Save-Json (Join-Path $WorkRoot 'last-github.json') @{ repo = $forkRepo; branch = $branch; commit = $commit.sha; source_url = $script:PublishedCodeUrl; compare_url = $compare }
    Write-Host ('GITHUB COMPLETE: ' + $script:PublishedCodeUrl) -ForegroundColor Green
    Write-Host ('Open this page to create your PR: ' + $compare)
    Start-Process -FilePath $compare | Out-Null
}

function Select-File([string]$Title, [string]$Filter, [string]$InitialDirectory) {
    Add-Type -AssemblyName System.Windows.Forms
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = $Title
    $dialog.Filter = $Filter
    $dialog.CheckFileExists = $true
    $dialog.Multiselect = $false
    if (Test-Path -LiteralPath $InitialDirectory -PathType Container) { $dialog.InitialDirectory = $InitialDirectory }
    try {
        if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { return $dialog.FileName }
        return $null
    } finally { $dialog.Dispose() }
}

function Publish-HuggingFace {
    $mode = if ($ModelPublishMode -eq 'PullRequest') { 'pr' } else { 'direct' }
    $targetDescription = if ($mode -eq 'pr') { 'all models in one pull request; maintainer merges' } else { 'all models, direct commits' }
    Write-Host ('Model target: https://huggingface.co/' + $ModelRepo + ' (' + $targetDescription + ')') -ForegroundColor Cyan
    $records = Join-Path $WorkspaceRoot 'old/records'
    $profiles = Join-Path $CodeRoot 'publish_profiles/legacy_collection.yaml'
    $exporter = Join-Path $PSScriptRoot 'export_model_collection.py'
    $uploader = Join-Path $PSScriptRoot 'upload_model_collection.py'
    Write-Host 'Checking all stream/fold sources. No checkpoint picker is needed.'
    Invoke-Native $script:Python @($exporter, '--records-root', $records, '--profiles', $profiles, '--plan-only')
    Write-Host 'Legacy profiles are explicit settings, not proof of historical stream semantics.'
    Write-Host 'Four-/six-stream runtime versions were not retained; review legacy_collection.yaml if their resize settings differ.'
    $confirmed = Read-Host 'Type YES to publish every listed legacy model using these declared profiles'
    if ($confirmed -cne 'YES') { Write-Host 'HUGGING FACE SKIPPED: collection profiles were not confirmed.'; return }
    $collection = Join-Path $WorkRoot 'model-collection'
    $stateName = if ($mode -eq 'pr') { 'upload-collection-pr-state.json' } else { 'upload-collection-state.json' }
    $uploadState = Join-Path $WorkRoot $stateName
    $accessDescription = if ($mode -eq 'pr') { 'Confirming login and repository read access before preparing the model PR ...' } else { 'Confirming write access before exporting the full collection ...' }
    Write-Host $accessDescription
    Invoke-Native $script:Python @($uploader, '--repo', $ModelRepo, '--mode', $mode, '--login-only')
    if ($mode -eq 'pr') {
        Write-Host 'Preparing or resuming one model PR before exporting the full collection ...'
        $prepareArguments = @($uploader, '--repo', $ModelRepo, '--mode', 'pr', '--prepare-pr', '--state-file', $uploadState)
        if ($script:PublishedCodeUrl -match '^https://github\.com/[A-Za-z0-9-]+/ERM-AI/tree/[a-fA-F0-9]{40}/new$') {
            $prepareArguments += @('--code-url', $script:PublishedCodeUrl)
        }
        Invoke-Native $script:Python $prepareArguments
    }
    Write-Host 'Exporting all models into stable per-condition/per-fold directories. Completed exports are reused on retry.'
    Invoke-Native $script:Python @($exporter, '--records-root', $records, '--profiles', $profiles, '--output', $collection, '--code-url', $script:PublishedCodeUrl)
    Write-Host 'Uploading models one at a time. Remote weights/configs are checked before skipping completed items.'
    Invoke-Native $script:Python @($uploader, '--folder', $collection, '--repo', $ModelRepo, '--mode', $mode, '--state-file', $uploadState)
    if ($mode -eq 'pr') {
        if (-not (Test-Path -LiteralPath $uploadState -PathType Leaf)) { throw 'The model PR upload completed without its local PR receipt. Check the uploader output.' }
        $receipt = [IO.File]::ReadAllText($uploadState, [Text.Encoding]::UTF8) | ConvertFrom-Json
        $expectedPrefix = 'https://huggingface.co/' + $ModelRepo + '/discussions/'
        if (-not $receipt.pr_url -or $receipt.repo -ne $ModelRepo -or $receipt.pr_status -ne 'open' -or
                $receipt.pr_url -notmatch ('^' + [regex]::Escape($expectedPrefix) + '[1-9][0-9]*$')) {
            throw 'The local model PR receipt did not identify the expected target repository.'
        }
        Write-Host ('HUGGING FACE PR READY: ' + $receipt.pr_url) -ForegroundColor Green
        Write-Host 'All uploaded models are on this open PR branch. The repository maintainer must review and merge it into the main branch.'
        Start-Process -FilePath $receipt.pr_url | Out-Null
    } else {
        Write-Host ('HUGGING FACE COMPLETE: https://huggingface.co/' + $ModelRepo) -ForegroundColor Green
    }
}
if ($DefinitionsOnly) { return }
Set-Location -LiteralPath $CodeRoot
Write-Host 'ERM-AI publisher' -ForegroundColor Cyan
Write-Host 'GitHub: personal fork and publication branch; you create or review the PR in the browser.'
if ($ModelPublishMode -eq 'PullRequest') {
    Write-Host 'Hugging Face: every stream/fold model and historical tensor archive in one resumable PR; the maintainer merges it.'
} else {
    Write-Host 'Hugging Face: every stream/fold model and historical tensor archive in separate directories, direct commits.'
}
Write-Host 'Passwords/tokens are entered only in the official local login flow. The original files are kept.'
$publicFiles = @(Get-PublicFiles)
Save-Json (Join-Path $SessionRoot 'public-source-manifest.json') @{
    paths = @($publicFiles | ForEach-Object { $_.path });
    excluded = @('old', 'datasets', 'runs', 'artifacts', '.venv', '.cache', '.publish-work', 'verification', 'patient manifests', 'private results')
}
Write-Host ('Selected public source files: ' + $publicFiles.Count)
if ($Target -eq 'Preview') {
    $publicFiles | ForEach-Object { Write-Host $_.path }
    Write-Host ('PREVIEW COMPLETE. No login, fork, source upload, or model upload was performed. Manifest: ' + $SessionRoot) -ForegroundColor Green
    return
}
Ensure-PythonPackages
if ($Target -in @('All', 'GitHub')) { Publish-GitHub $publicFiles }
if ($Target -eq 'HuggingFace') {
    $previous = Join-Path $WorkRoot 'last-github.json'
    if (Test-Path -LiteralPath $previous -PathType Leaf) {
        $state = [IO.File]::ReadAllText($previous, [Text.Encoding]::UTF8) | ConvertFrom-Json
        if ($state.source_url -match '^https://github\.com/[A-Za-z0-9-]+/ERM-AI/tree/[a-fA-F0-9]{40}/new$') { $script:PublishedCodeUrl = $state.source_url }
    }
}
if ($Target -in @('All', 'HuggingFace')) { Publish-HuggingFace }
Write-Host 'DONE. Local originals and any completed remote commits have been kept.' -ForegroundColor Green
