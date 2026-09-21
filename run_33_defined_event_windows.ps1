param(
    [string]$Root = 'C:\Users\KUST\01QZF',
    [Parameter(Mandatory = $true)][string]$TrackingDir,
    [double]$Fps = 25.145909137771923,
    [string]$FrameIndex = '',
    [string]$SemanticDir = '',
    [string[]]$FusionDir = @(),
    [string]$PoseWeights = '',
    [string]$OutputRoot = '',
    [ValidateSet('cpu','cuda')][string]$Device = 'cuda',
    [switch]$DisablePose,
    [switch]$SkipGnnLstm
)

$ErrorActionPreference = 'Stop'
$pipeline = Join-Path $Root 'vision_graph_pipeline'
$runner = Join-Path $pipeline 'run_joint_scene_test.ps1'
$artifacts = Join-Path $pipeline 'artifacts'
if (-not (Test-Path -LiteralPath $runner)) { throw "Joint runner missing: $runner" }

$invoke = @{
    Root = $Root
    TrackingDir = $TrackingDir
    Fps = $Fps
    EventId = '33_J1,33_P1'
    Device = $Device
    UseDefinedEventRanges = $true
    FrameIndex = $(if ($FrameIndex) { $FrameIndex } else { Join-Path $artifacts 'frame_index.csv' })
}
if ($SemanticDir) { $invoke.SemanticDir = $SemanticDir }
if ($FusionDir.Count -gt 0) { $invoke.FusionDir = $FusionDir }
if ($PoseWeights) { $invoke.PoseWeights = $PoseWeights }
if ($OutputRoot) { $invoke.OutputRoot = $OutputRoot }
if ($DisablePose) { $invoke.DisablePose = $true }
if ($SkipGnnLstm) { $invoke.SkipGnnLstm = $true }

Write-Host 'Running 33_J1 (1245-1834) and 33_P1 (1950-2599) directly.'
Write-Host 'No E2 correspondence table or frame-extraction step will run.'
& $runner @invoke
if ($LASTEXITCODE -ne 0) { throw "Joint scene test failed with exit code $LASTEXITCODE" }
