$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$archive = Join-Path $projectRoot 'local-ner-bakeoff-submission.zip'
$required = @('.gitattributes', '.gitignore', 'README.md', 'NOTE.md', 'FULL_REPORT.md', 'FULL_REPORT.html', 'requirements.txt', 'requirements.lock.txt', 'models.lock.json', 'runtime.lock.json', 'ASSIGNMENT.md', 'score.py', 'data', 'src', 'scripts', 'tests', 'predictions', 'results', 'output')
foreach ($name in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $projectRoot $name))) {
        throw "Required deliverable missing: $name"
    }
}
$stage = Join-Path $projectRoot 'submission-stage'
if (Test-Path -LiteralPath $stage) { throw 'Archive staging directory already exists.' }
New-Item -ItemType Directory -Path $stage | Out-Null
foreach ($name in $required) {
    Copy-Item -LiteralPath (Join-Path $projectRoot $name) -Destination $stage -Recurse
}
$resolvedStage = (Resolve-Path -LiteralPath $stage).Path
if (-not $resolvedStage.StartsWith($projectRoot + [IO.Path]::DirectorySeparatorChar)) { throw 'Unexpected staging path' }
Get-ChildItem -LiteralPath $stage -Directory -Filter '__pycache__' -Recurse | Remove-Item -Recurse -Force
$payload = @(Get-ChildItem -LiteralPath $stage -File -Recurse -Force | Sort-Object FullName | ForEach-Object {
    [PSCustomObject]@{
        path = $_.FullName.Substring($resolvedStage.Length + 1).Replace('\', '/')
        bytes = $_.Length
        sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
})
$manifest = [PSCustomObject]@{
    description = 'SHA-256 manifest for final submission payload; excludes the manifest itself.'
    files = $payload
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $stage 'SUBMISSION_MANIFEST.json') -Encoding utf8
Copy-Item -LiteralPath (Join-Path $stage 'SUBMISSION_MANIFEST.json') -Destination (Join-Path $projectRoot 'SUBMISSION_MANIFEST.json') -Force
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $archive -Force
Remove-Item -LiteralPath $resolvedStage -Recurse -Force
Write-Output $archive
