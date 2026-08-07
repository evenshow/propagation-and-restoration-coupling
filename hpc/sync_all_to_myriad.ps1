<#
.SYNOPSIS
    Push both coupling0729 and the legacy coupling_sim tree to UCL Myriad.

.DESCRIPTION
    Case 1 wraps the mature IEEE33 and Sioux Falls simulators, which live in
    coupling_sim and are located at import time by walking up the directory tree
    for a sibling `coupling_sim`. Syncing coupling0729 alone therefore produces
    a remote tree that imports fine until case 1 is actually constructed. Both
    leaves have to land under the same remote parent.

    Everything here is small (~1.7 MB), so both trees go in one tarball:
    /myriadfs writes small files at only ~37/s but takes a stream at 2.8 GB/s.

    Outputs (results/, hpc_results/) are excluded so a sync never clobbers
    results already on Myriad.

.EXAMPLE
    .\sync_all_to_myriad.ps1
    .\sync_all_to_myriad.ps1 -Clean
#>
[CmdletBinding()]
param(
    [string] $LocalParent  = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string[]] $Leaves     = @("coupling0729", "coupling_sim"),
    [string] $RemoteParent = "coupling0729",
    [string] $LoginNode    = "login13.myriad.rc.ucl.ac.uk",
    [string] $RemoteUser   = $env:USERNAME,
    [string] $KeyFile      = "$env:USERPROFILE\.ssh\id_ed25519",
    [switch] $Clean
)

$ErrorActionPreference = "Stop"

# Windows native OpenSSH only. Git Bash's OpenSSH 10.x fails authentication
# against Myriad even with a valid key.
$SSH     = "C:\Windows\System32\OpenSSH\ssh.exe"
$SCP     = "C:\Windows\System32\OpenSSH\scp.exe"
$SshOpts = @("-i", $KeyFile, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes")
$Remote  = "$RemoteUser@$LoginNode"

if (-not (Test-Path -LiteralPath $KeyFile)) { throw "SSH key not found: $KeyFile" }
foreach ($leaf in $Leaves) {
    $p = Join-Path $LocalParent $leaf
    if (-not (Test-Path -LiteralPath $p)) { throw "not found: $p" }
}

# OneDrive "files on demand" can leave a cloud placeholder where the real file
# should be; tar would pack the stub and the remote would get an empty shell.
Write-Host "checking OneDrive placeholders..." -ForegroundColor Cyan
$stubs = @()
foreach ($leaf in $Leaves) {
    $stubs += Get-ChildItem -LiteralPath (Join-Path $LocalParent $leaf) -Recurse -File -ErrorAction SilentlyContinue |
              Where-Object { $_.Attributes -match 'Offline' -or $_.Attributes -match 'RecallOnDataAccess' }
}
if ($stubs) {
    $stubs | Select-Object -First 10 | ForEach-Object { Write-Warning "  placeholder: $($_.FullName)" }
    throw "$($stubs.Count) file(s) are OneDrive placeholders. Right-click the folder -> 'Always keep on this device', wait, re-run."
}

# Shell scripts must arrive with LF and no BOM: a CRLF jobscript fails on Myriad
# as "$'\r': command not found", and a BOM makes the shebang unreadable.
Write-Host "normalising shell scripts..." -ForegroundColor Cyan
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$fixed = 0
foreach ($leaf in $Leaves) {
    Get-ChildItem -LiteralPath (Join-Path $LocalParent $leaf) -Recurse -File -Filter *.sh -ErrorAction SilentlyContinue |
        ForEach-Object {
            $raw  = [System.IO.File]::ReadAllText($_.FullName)
            $norm = $raw.TrimStart([char]0xFEFF) -replace "`r`n", "`n" -replace "`r", "`n"
            if ($norm -ne $raw) {
                [System.IO.File]::WriteAllText($_.FullName, $norm, $utf8NoBom)
                Write-Host "  normalised $($_.Name)" -ForegroundColor Yellow
                $fixed++
            }
        }
}
if ($fixed -eq 0) { Write-Host "  already clean" -ForegroundColor DarkGray }

$exclude = @(
    "--exclude=__pycache__", "--exclude=*.pyc", "--exclude=.pytest_cache",
    "--exclude=.git", "--exclude=.venv*",
    "--exclude=results", "--exclude=hpc_results", "--exclude=_tmp*"
)

$tarball = Join-Path $env:TEMP "coupling_sync_all.tar.gz"
if (Test-Path -LiteralPath $tarball) { Remove-Item -LiteralPath $tarball -Force }

# --format=ustar: this bsdtar has no 'gnu' format, and its default pax headers
# make GNU tar on the far end warn about unknown SCHILY.fflags keywords.
Write-Host "packing $($Leaves -join ', ') ..." -ForegroundColor Cyan
& tar.exe -czf $tarball --format=ustar -C $LocalParent @exclude @Leaves
if ($LASTEXITCODE -ne 0) { throw "tar failed with exit code $LASTEXITCODE" }
Write-Host "  $([math]::Round((Get-Item -LiteralPath $tarball).Length / 1KB, 1)) KB" -ForegroundColor DarkGray

Write-Host "uploading..." -ForegroundColor Cyan
& $SCP @SshOpts $tarball "${Remote}:$RemoteParent/"
if ($LASTEXITCODE -ne 0) { throw "scp failed with exit code $LASTEXITCODE" }

# Deliberately quote-free: Windows ssh.exe strips quotes from the remote command.
#
# bsdtar on Windows records every directory as 0555 because Windows has no
# directory write mode to map, so a straight extract creates read-only
# directories the extract itself cannot then write into.
# --delay-directory-restore defers the archived modes; the chmod restores write
# bits. The leading chmod must precede any purge, since an interrupted earlier
# sync can leave 0555 directories that rm cannot descend into.
$tbName = [System.IO.Path]::GetFileName($tarball)
$leafList = $Leaves -join " "
$purge  = if ($Clean) { "rm -rf $leafList;" } else { "" }
$cmd    = "cd ~/$RemoteParent && chmod -R u+rwX $leafList 2>/dev/null; $purge tar --delay-directory-restore -xzf $tbName && chmod -R u+rwX $leafList && rm -f $tbName && echo REMOTE_FILES=`$(find $leafList -type f | wc -l)"

Write-Host "unpacking on Myriad..." -ForegroundColor Cyan
& $SSH @SshOpts $Remote $cmd
if ($LASTEXITCODE -ne 0) { throw "remote unpack failed with exit code $LASTEXITCODE" }

Remove-Item -LiteralPath $tarball -Force
Write-Host "synced -> ${Remote}:~/$RemoteParent/{$($Leaves -join ',')}" -ForegroundColor Green
