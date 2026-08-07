<#
.SYNOPSIS
    Pull results and job logs back from UCL Myriad.

.DESCRIPTION
    Bundles the remote results/ tree (and the SGE .o/.e logs) into one tarball,
    ships it back, and unpacks it under hpc_results\<timestamp>\ inside the
    project. Nothing local is overwritten -- every fetch lands in its own dated
    folder, so runs stay comparable.

.EXAMPLE
    .\fetch_results.ps1
    .\fetch_results.ps1 -IncludeLogs:$false
#>
[CmdletBinding()]
param(
    [string] $LocalRoot    = (Split-Path -Parent $PSScriptRoot),
    [string] $RemoteParent = "coupling0729",
    [string] $LoginNode    = "login13.myriad.rc.ucl.ac.uk",
    [string] $RemoteUser   = $env:USERNAME,
    [string] $KeyFile      = "$env:USERPROFILE\.ssh\id_ed25519",
    [bool]   $IncludeLogs  = $true
)

$ErrorActionPreference = "Stop"

$SSH     = "C:\Windows\System32\OpenSSH\ssh.exe"
$SCP     = "C:\Windows\System32\OpenSSH\scp.exe"
$SshOpts = @("-i", $KeyFile, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes")
$Remote  = "$RemoteUser@$LoginNode"
# .NET rather than Split-Path: in PowerShell 5.1 -LiteralPath and -Leaf sit in
# mutually exclusive parameter sets and cannot be combined.
$leaf    = [System.IO.Path]::GetFileName($LocalRoot.TrimEnd('\'))

if (-not (Test-Path -LiteralPath $LocalRoot)) { throw "LocalRoot not found: $LocalRoot" }

$stamp   = Get-Date -Format "yyyyMMdd_HHmmss"
$dest    = Join-Path $LocalRoot "hpc_results\$stamp"
$tbName  = "coupling_results_$stamp.tar.gz"
$memFile = "/tmp/members_$stamp.txt"
$wanted  = if ($IncludeLogs) { "$leaf/results logs" } else { "$leaf/results" }

# ls -d filters the list down to whatever actually exists, and tar -T reads the
# members from that file. Naming absent members on the tar line instead would
# make it exit 2 and bury the real outcome. Quote-free: Windows ssh.exe strips
# quotes out of the remote command.
$cmd = "cd ~/$RemoteParent && ls -d $wanted 2>/dev/null | tee $memFile; tar -czf /tmp/$tbName -T $memFile; echo BUNDLE_RC=`$?; rm -f $memFile"

Write-Host "bundling results on Myriad..." -ForegroundColor Cyan
& $SSH @SshOpts $Remote $cmd
if ($LASTEXITCODE -ne 0) { throw "remote bundling failed (exit $LASTEXITCODE) - do results exist yet?" }

$localTb = Join-Path $env:TEMP $tbName
Write-Host "downloading..." -ForegroundColor Cyan
& $SCP @SshOpts "${Remote}:/tmp/$tbName" $localTb
if ($LASTEXITCODE -ne 0) { throw "scp download failed with exit code $LASTEXITCODE" }

& $SSH @SshOpts $Remote "rm -f /tmp/$tbName" | Out-Null

New-Item -ItemType Directory -Force -Path $dest | Out-Null
& tar.exe -xzf $localTb -C $dest
if ($LASTEXITCODE -ne 0) { throw "local extract failed with exit code $LASTEXITCODE" }
Remove-Item -LiteralPath $localTb -Force

$files = Get-ChildItem -LiteralPath $dest -Recurse -File
Write-Host "fetched $($files.Count) file(s) -> $dest" -ForegroundColor Green
$files | Sort-Object LastWriteTime -Descending | Select-Object -First 15 |
    ForEach-Object { "  {0,10:N0}  {1}" -f $_.Length, $_.FullName.Substring($dest.Length + 1) }
