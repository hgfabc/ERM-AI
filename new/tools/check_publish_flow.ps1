param([Parameter(Mandatory=$true)][string]$Case, [Parameter(Mandatory=$true)][string]$Python)
# Offline checks only: GitHub, CLI and browser operations are replaced below.
$ErrorActionPreference = 'Stop'
$TestInterpreter = $Python
. (Join-Path $PSScriptRoot 'publish_erm_ai.ps1') -DefinitionsOnly
$ProductionWorkRoot = $WorkRoot
$WorkRoot = Join-Path $CodeRoot ('.publish-work/test-windows-' + [guid]::NewGuid().ToString('N'))
$SessionRoot = Join-Path $WorkRoot 'session'
New-Item -ItemType Directory -Path $SessionRoot -Force | Out-Null

if ($Case -eq 'native-exit') {
    $success = Test-Native $TestInterpreter @('-c', "print('visible progress')")
    if ($success -isnot [bool] -or -not $success) { throw 'Native success status was contaminated by stdout.' }
    $failure = Test-Native $TestInterpreter @('-c', "import sys; print('visible failure'); print('expected stderr',file=sys.stderr); sys.exit(7)")
    if ($failure -isnot [bool] -or $failure) { throw 'Native failure was incorrectly accepted.' }
    Write-Host 'Native status checks passed.'
    exit 0
}

if ($Case -eq 'source-list') {
    $files = @(Get-PublicFiles)
    if ($files.Count -lt 10) { throw 'Unexpectedly incomplete public source package.' }
    foreach ($file in $files) {
        if ($file.path -match '(^|/)(old|audit|runs|artifacts|smoke_output|\.venv|\.cache|\.publish-work)(/|$)|verification\.json|\.(pth|pt|safetensors|png|jpg|jpeg)$') {
            throw ('Private/generated path selected: ' + $file.path)
        }
    }
    if (-not @($files | Where-Object { $_.path -eq 'new/main.py' })) { throw 'Missing main entry point.' }
    $publicPaths = @($files | ForEach-Object { $_.path })
    foreach ($readmePath in @('README.md', 'new/README.md', 'new/README.zh-TW.md', 'new/README.ja.md')) {
        $readmeFile = @($files | Where-Object { $_.path -ceq $readmePath })
        if ($readmeFile.Count -ne 1) { throw ('Missing translated README: ' + $readmePath) }
        $readmeText = $Utf8.GetString($readmeFile[0].bytes)
        $navigation = ($readmeText -split '\r?\n', 2)[0]
        $links = [regex]::Matches($navigation, '\[[^\]]+\]\(([^)]+)\)')
        if ($links.Count -ne 3 -or -not $navigation.StartsWith('[English](README.md)')) {
            throw ('Missing top-level language selector: ' + $readmePath)
        }
        foreach ($link in $links) {
            $languagePath = if ($readmePath.StartsWith('new/')) { 'new/' + $link.Groups[1].Value } else { $link.Groups[1].Value }
            if ($languagePath -cnotin $publicPaths) { throw ('Unpublished language link: ' + $readmePath + ' -> ' + $languagePath) }
        }
        if ($readmeText -match '\]\(\.\./old/') { throw ('Local archive link in published README: ' + $readmePath) }
    }
    Write-Host ('Public source list passed: ' + $files.Count + ' files.')
    exit 0
}

if ($Case -in @('model-flow', 'model-direct', 'model-no-github')) {
    # Isolated local receipts only: no collection export, CLI login, network or browser.
    $ModelPublishMode = if ($Case -eq 'model-direct') { 'Direct' } else { 'PullRequest' }
    $isPr = ($ModelPublishMode -eq 'PullRequest')
    $script:Python = $TestInterpreter
    $script:PublishedCodeUrl = if ($Case -eq 'model-no-github') { 'https://github.com/ntuhophai/ERM-AI' } else { 'https://github.com/publisher-test/ERM-AI/tree/' + ('a' * 40) + '/new' }
    $script:ModelCalls = New-Object System.Collections.Generic.List[object]
    $script:MockConfirmation = 'YES'
    $script:FailModelLogin = $false
    $script:FailPreparePr = $false
    $script:BadPrReceipt = $false
    $script:MockPlan = @{ model_count = 167; inference_ready_count = 160; historical_count = 7; condition_count = 16; fold_count = 10 }
    $mockPrUrl = 'https://huggingface.co/ntuhophai/ERM-AI/discussions/42'
    function Select-File { throw 'The collection flow must not show a single-checkpoint picker.' }
    function Read-Host([string]$Prompt) {
        $script:ModelCalls.Add(@{ kind = 'confirmation'; prompt = $Prompt })
        return $script:MockConfirmation
    }
    function Start-Process([string]$FilePath) { $script:ModelCalls.Add(@{ kind = 'browser'; url = $FilePath }) }
    function Invoke-Native([string]$Exe, [string[]]$Arguments) {
        $script:ModelCalls.Add(@{ kind = 'native'; exe = $Exe; arguments = @($Arguments) })
        if ($Arguments -contains '--plan-only') { Write-Host (ConvertTo-Json $script:MockPlan -Compress) }
        if ($script:FailModelLogin -and $Arguments -contains '--login-only') { throw 'Mock model access denial.' }
        if ($script:FailPreparePr -and $Arguments -contains '--prepare-pr') { throw 'Mock PR preparation denial.' }
        if ($isPr -and ($Arguments -contains '--prepare-pr' -or $Arguments -contains '--folder')) {
            $stateIndex = [Array]::IndexOf($Arguments, '--state-file')
            if ($stateIndex -lt 0) { throw 'Missing PR state file.' }
            $url = if ($script:BadPrReceipt) { 'https://example.invalid/unsafe' } else { $mockPrUrl }
            $status = if ($Arguments -contains '--folder') { 'open' } else { 'draft' }
            Save-Json $Arguments[$stateIndex + 1] @{ repo = 'ntuhophai/ERM-AI'; pr_url = $url; pr_ref = 'refs/pr/42'; pr_status = $status }
        }
    }
    function Assert-ModelArgument($Call, [string]$Flag, [string]$Expected) {
        $position = [Array]::IndexOf([string[]]$Call.arguments, $Flag)
        if ($position -lt 0 -or $position + 1 -ge $Call.arguments.Count -or $Call.arguments[$position + 1] -cne $Expected) {
            throw ('Incorrect collection argument: ' + $Flag)
        }
    }
    function Assert-CompleteModelFlow {
        $events = $script:ModelCalls.ToArray()
        $expectedOrder = if ($isPr) { 'native,confirmation,native,native,native,native,browser' } else { 'native,confirmation,native,native,native' }
        if (($events.kind -join ',') -cne $expectedOrder) { throw 'Incorrect plan, confirmation, access, PR preparation, export or upload order.' }
        $native = @($events | Where-Object { $_.kind -eq 'native' })
        foreach ($call in $native) {
            if ($call.exe -cne $TestInterpreter) { throw 'The model flow used an unexpected interpreter.' }
            if (@($call.arguments | Where-Object { $_ -in @('--checkpoint', '--legacy-config', '--condition', '--fold') }).Count) {
                throw 'The all-model flow was reduced to a single checkpoint/condition/fold.'
            }
        }
        $exporter = Join-Path $PSScriptRoot 'export_model_collection.py'
        $uploader = Join-Path $PSScriptRoot 'upload_model_collection.py'
        $exportIndex = if ($isPr) { 3 } else { 2 }
        $uploadIndex = $exportIndex + 1
        $mode = if ($isPr) { 'pr' } else { 'direct' }
        if ($native[0].arguments[0] -cne $exporter -or -not ($native[0].arguments -contains '--plan-only')) { throw 'Missing collection plan.' }
        if ($native[1].arguments[0] -cne $uploader -or -not ($native[1].arguments -contains '--login-only') -or ($native[1].arguments -contains '--folder')) {
            throw 'Access must be checked before exporting, without requiring a collection folder.'
        }
        if ($native[$exportIndex].arguments[0] -cne $exporter -or ($native[$exportIndex].arguments -contains '--plan-only')) { throw 'Missing full collection export.' }
        if ($native[$uploadIndex].arguments[0] -cne $uploader -or ($native[$uploadIndex].arguments -contains '--login-only')) { throw 'Missing collection upload.' }
        $expectedCollection = Join-Path $WorkRoot 'model-collection'
        $stateName = if ($isPr) { 'upload-collection-pr-state.json' } else { 'upload-collection-state.json' }
        $expectedState = Join-Path $WorkRoot $stateName
        foreach ($call in @($native[0], $native[$exportIndex])) {
            Assert-ModelArgument $call '--records-root' (Join-Path $WorkspaceRoot 'old/records')
            Assert-ModelArgument $call '--profiles' (Join-Path $CodeRoot 'publish_profiles/legacy_collection.yaml')
        }
        foreach ($call in @($native[1], $native[$uploadIndex])) {
            Assert-ModelArgument $call '--repo' 'ntuhophai/ERM-AI'
            Assert-ModelArgument $call '--mode' $mode
        }
        Assert-ModelArgument $native[$exportIndex] '--output' $expectedCollection
        Assert-ModelArgument $native[$exportIndex] '--code-url' $script:PublishedCodeUrl
        Assert-ModelArgument $native[$uploadIndex] '--folder' $expectedCollection
        Assert-ModelArgument $native[$uploadIndex] '--state-file' $expectedState
        if ($isPr) {
            if ($native[2].arguments[0] -cne $uploader -or -not ($native[2].arguments -contains '--prepare-pr')) { throw 'PR preparation must precede export.' }
            Assert-ModelArgument $native[2] '--mode' 'pr'
            Assert-ModelArgument $native[2] '--repo' 'ntuhophai/ERM-AI'
            Assert-ModelArgument $native[2] '--state-file' $expectedState
            if ($Case -eq 'model-no-github') {
                if ($native[2].arguments -contains '--code-url') { throw 'PR preparation must omit a missing GitHub commit snapshot.' }
            } else {
                Assert-ModelArgument $native[2] '--code-url' $script:PublishedCodeUrl
            }
            if ($events[-1].url -cne $mockPrUrl) { throw 'The expected HF PR page was not opened.' }
        }
        if ([IO.Path]::GetDirectoryName($expectedState) -cne $WorkRoot -or $expectedState.StartsWith($expectedCollection + [IO.Path]::DirectorySeparatorChar)) {
            throw 'Upload state must remain outside the public collection.'
        }
        if ($script:MockPlan.model_count -ne 167 -or $script:MockPlan.inference_ready_count -ne 160 -or $script:MockPlan.historical_count -ne 7) {
            throw 'Unexpected complete-collection fixture.'
        }
        return ,$native
    }
    $SessionRoot = Join-Path $WorkRoot 'offline-model-session-one'
    Publish-HuggingFace
    $first = Assert-CompleteModelFlow
    $firstArguments = @($first | ForEach-Object { $_.arguments -join '|' })
    $script:ModelCalls.Clear()
    $SessionRoot = Join-Path $WorkRoot 'offline-model-session-two'
    Publish-HuggingFace
    $second = Assert-CompleteModelFlow
    $secondArguments = @($second | ForEach-Object { $_.arguments -join '|' })
    if (($firstArguments -join "`n") -cne ($secondArguments -join "`n")) { throw 'Retry paths changed with the per-run session.' }
    foreach ($answer in @('NO', 'yes', '', ' YES ')) {
        $script:ModelCalls.Clear()
        $script:MockConfirmation = $answer
        Publish-HuggingFace
        if ($script:ModelCalls.Count -ne 2 -or ($script:ModelCalls.kind -join ',') -ne 'native,confirmation' -or
                -not ($script:ModelCalls[0].arguments -contains '--plan-only')) {
            throw 'A non-YES confirmation continued to login, PR preparation, export or upload.'
        }
    }
    $script:ModelCalls.Clear()
    $script:MockConfirmation = 'YES'
    $script:FailModelLogin = $true
    $denied = $false
    try { Publish-HuggingFace } catch { $denied = $true }
    if (-not $denied -or $script:ModelCalls.Count -ne 3 -or -not ($script:ModelCalls[2].arguments -contains '--login-only')) {
        throw 'Access denial must stop before PR preparation and the large export/upload.'
    }
    if ($isPr) {
        $script:FailModelLogin = $false
        $script:FailPreparePr = $true
        $script:ModelCalls.Clear()
        $denied = $false
        try { Publish-HuggingFace } catch { $denied = $true }
        if (-not $denied -or $script:ModelCalls.Count -ne 4 -or -not ($script:ModelCalls[3].arguments -contains '--prepare-pr')) {
            throw 'PR preparation failure must stop before exporting or uploading models.'
        }
        $script:FailPreparePr = $false
        $script:BadPrReceipt = $true
        $script:ModelCalls.Clear()
        $denied = $false
        try { Publish-HuggingFace } catch { $denied = $true }
        if (-not $denied -or @($script:ModelCalls | Where-Object { $_.kind -eq 'browser' }).Count) {
            throw 'An invalid PR receipt must not open a browser page.'
        }
    }
    Write-Host 'Mock complete-model flow passed: 167 entries, stable collection/state, PR/direct modes, access/preparation failure and cancellation. No model commands were executed.'
    exit 0
}

$script:FakeCreated = ($Case -eq 'wrong-fork')
$script:ReuseBranch = 'publish/new-code-20261012-010203-1234abcd'
if ($Case -eq 'github-reuse') {
    $script:FakeCreated = $true
    Save-Json (Join-Path $WorkRoot 'last-github.json') @{ repo = 'publisher-test/ERM-AI'; branch = $script:ReuseBranch; commit = ('a' * 40) }
}
$script:Calls = New-Object System.Collections.Generic.List[object]
function Get-Gh { return 'mock-gh-never-executed' }
function Test-Native { return $true }
function Invoke-Native([string]$Exe, [string[]]$Arguments) {
    $script:Calls.Add(@{ kind = 'native'; arguments = $Arguments })
    if ($Arguments[0] -eq 'repo' -and $Arguments[1] -eq 'fork') { $script:FakeCreated = $true }
}
function Start-Process { $script:Calls.Add(@{ kind = 'browser' }); return $null }
function Invoke-GhJson([string]$Method, [string]$Endpoint, $Body = $null, [switch]$Allow404) {
    $script:Calls.Add(@{ kind = 'api'; method = $Method; endpoint = $Endpoint; body = $Body })
    switch ($Endpoint) {
        'user' { return [pscustomobject]@{ login = 'publisher-test'; id = 101; email = 'private@example.test' } }
        'repos/ntuhophai/ERM-AI' { return [pscustomobject]@{ id = 202; default_branch = 'main' } }
        'repos/publisher-test/ERM-AI' {
            if (-not $script:FakeCreated) { return $null }
            $parentId = if ($Case -eq 'wrong-fork') { 999 } else { 202 }
            return [pscustomobject]@{ owner = @{ login = 'publisher-test' }; fork = $true;
                parent = @{ id = $parentId; full_name = 'ntuhophai/ERM-AI' }; permissions = @{ push = $true } }
        }
        'repos/ntuhophai/ERM-AI/git/ref/heads/main' { return [pscustomobject]@{ object = @{ sha = ('a' * 40) } } }
        { $_ -eq ('repos/publisher-test/ERM-AI/git/ref/heads/' + $script:ReuseBranch) } { return [pscustomobject]@{ object = @{ sha = ('a' * 40) } } }
        { $_.StartsWith('repos/ntuhophai/ERM-AI/pulls?') } { return @() }
        { $_ -eq ('repos/ntuhophai/ERM-AI/git/commits/' + ('a' * 40)) } { return [pscustomobject]@{ sha = ('a' * 40); tree = @{ sha = ('b' * 40) } } }
        { $_ -eq ('repos/publisher-test/ERM-AI/git/commits/' + ('a' * 40)) } { return [pscustomobject]@{ sha = ('a' * 40); tree = @{ sha = ('b' * 40) } } }
        'repos/publisher-test/ERM-AI/git/blobs' { return [pscustomobject]@{ sha = ('c' * 40) } }
        'repos/publisher-test/ERM-AI/git/trees' { return [pscustomobject]@{ sha = ('d' * 40) } }
        'repos/publisher-test/ERM-AI/git/commits' { return [pscustomobject]@{ sha = ('e' * 40) } }
        'repos/publisher-test/ERM-AI/git/refs' { return [pscustomobject]@{ ref = $Body.ref } }
        { $_ -eq ('repos/publisher-test/ERM-AI/git/refs/heads/' + $script:ReuseBranch) } { return [pscustomobject]@{ ref = ('refs/heads/' + $script:ReuseBranch) } }
        default { throw ('Unmocked endpoint: ' + $Endpoint) }
    }
}
$payload = @([pscustomobject]@{ path = 'new/main.py'; bytes = $Utf8.GetBytes('print("public fixture")') })
if ($Case -eq 'wrong-fork') {
    $rejected = $false
    try { Publish-GitHub $payload } catch { $rejected = $true }
    if (-not $rejected) { throw 'An unrelated existing fork was accepted.' }
    if (@($script:Calls | Where-Object { $_.kind -eq 'api' -and $_.method -eq 'POST' }).Count) { throw 'Wrong fork received writes.' }
    Write-Host 'Wrong fork rejected before writes.'
    exit 0
}
if ($Case -notin @('github-flow', 'github-reuse')) { throw 'Unknown offline test case.' }
Publish-GitHub $payload
$posts = @($script:Calls | Where-Object { $_.kind -eq 'api' -and $_.method -in @('POST', 'PATCH') })
if ($posts.Count -ne 4) { throw 'Unexpected GitHub mutation sequence.' }
foreach ($call in $posts) {
    if (-not $call.endpoint.StartsWith('repos/publisher-test/ERM-AI/')) { throw 'A write targeted upstream.' }
}
$treeCall = $posts | Where-Object { $_.endpoint.EndsWith('/trees') }
if ($treeCall.body.base_tree -ne ('b' * 40) -or $treeCall.body.tree[0].path -ne 'new/main.py') { throw 'Source tree was not based on upstream.' }
$commitCall = $posts | Where-Object { $_.endpoint.EndsWith('/commits') }
if ($commitCall.body.author.email -ne '101+publisher-test@users.noreply.github.com') { throw 'Private author email was used.' }
$refCall = $posts | Where-Object { $_.endpoint.Contains('/git/refs') }
if ($Case -eq 'github-reuse') {
    if ($refCall.method -ne 'PATCH' -or $refCall.body.force -ne $false -or -not $refCall.endpoint.EndsWith($script:ReuseBranch)) {
        throw 'Existing publication branch was not updated safely.'
    }
} elseif ($refCall.body.ref -notmatch '^refs/heads/publish/new-code-') { throw 'Default branch was targeted.' }
if (-not (Test-Path -LiteralPath (Join-Path $WorkRoot 'last-github.json'))) { throw 'Missing completion receipt.' }
Write-Host 'Mock GitHub fork/source branch flow passed. No remote commands were executed.'
