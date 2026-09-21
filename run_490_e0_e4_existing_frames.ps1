param(
    [string]$Root = 'C:\Users\KUST\01QZF',
    [Parameter(Mandatory = $true)][string]$PreparedFramesDir,
    [string]$DetectorModel = '',
    [string]$SegmentationScript = '',
    [string]$SegformerModel = '',
    [string]$HybridnetsRoot = '',
    [string]$HybridnetsWeights = '',
    [string]$PoseWeights = '',
    [ValidateSet('cpu','cuda')][string]$Device = 'cuda',
    [int]$ExpectedEvents = 490,
    [switch]$AllowMissingEvents,
    [switch]$SkipPose,
    [switch]$SkipGnnLstm
)

$ErrorActionPreference = 'Stop'
$pipeline = Join-Path $Root 'vision_graph_pipeline'
$runner = Join-Path $pipeline 'run_e2_corresponding_joint_scene.ps1'
$manifestPath = Join-Path $PreparedFramesDir 'event_manifest.json'
$indexPath = Join-Path $PreparedFramesDir 'frame_index_all.csv'
if (-not (Test-Path -LiteralPath $indexPath)) {
    $indexPath = Join-Path $PreparedFramesDir 'frame_index.csv'
}

foreach ($required in @($runner, $manifestPath, $indexPath)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required path missing: $required"
    }
}

$manifestText = [System.IO.File]::ReadAllText($manifestPath, [System.Text.Encoding]::UTF8)
$manifest = @($manifestText | ConvertFrom-Json)
$indexEventCount = @(Import-Csv -LiteralPath $indexPath | Select-Object -ExpandProperty event_id -Unique).Count

Write-Host ("Prepared event manifest: {0} events" -f $manifest.Count)
Write-Host ("Frame index coverage: {0} events" -f $indexEventCount)

if (($manifest.Count -ne $ExpectedEvents -or $indexEventCount -ne $ExpectedEvents) -and -not $AllowMissingEvents) {
    $failurePath = Join-Path $PreparedFramesDir 'extraction_failures.csv'
    $messageTemplate = (
        "Expected {0} E0-E4 events, but manifest/index contain {1}/{2}. " +
        "Inspect {3}; repair the missing event or use -AllowMissingEvents for an explicitly incomplete run."
    )
    throw ($messageTemplate -f $ExpectedEvents, $manifest.Count, $indexEventCount, $failurePath)
}

if (-not $DetectorModel) { $DetectorModel = Join-Path $Root 'models\yolov8n.onnx' }
if (-not (Test-Path -LiteralPath $DetectorModel)) {
    throw "YOLO ONNX model missing: $DetectorModel"
}

$invoke = @{
    Root = $Root
    PreparedFramesDir = $PreparedFramesDir
    DetectorModel = $DetectorModel
    Device = $Device
    MaxFrames = 0
}
if ($SegmentationScript) { $invoke.SegmentationScript = $SegmentationScript }
if ($SegformerModel) { $invoke.SegformerModel = $SegformerModel }
if ($HybridnetsRoot) { $invoke.HybridnetsRoot = $HybridnetsRoot }
if ($HybridnetsWeights) { $invoke.HybridnetsWeights = $HybridnetsWeights }
if ($PoseWeights) { $invoke.PoseWeights = $PoseWeights }
if ($SkipPose) { $invoke.SkipPose = $true }
if ($SkipGnnLstm) { $invoke.SkipGnnLstm = $true }

Write-Host 'Running full per-frame inference for every prepared E0-E4 event.'
Write-Host 'The frame preparation stage is bypassed; existing frames and frame_index are reused.'
& $runner @invoke
if ($LASTEXITCODE -ne 0) {
    throw "Full E0-E4 pipeline failed with exit code $LASTEXITCODE"
}
