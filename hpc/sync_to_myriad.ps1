<#
.SYNOPSIS
    Push the local coupling0729 tree to UCL Myriad.

.DESCRIPTION
    Packs the source tree into a single tarball and ships that, rather than
    scp-ing files one by one: /myriadfs writes small files at only ~37/s, but
    takes a sequential stream at 2.8 GB/s.

    Outputs (results/, hpc_results/) are excluded, so a sync never clobbers
    results already sitting on Myriad. Use fetch_results.ps1 to bring those back.

.PARAMETER Clean
    Delete the remote copy of the tree before unpacking. Without this, unpacking
    merges over what is there, so files deleted locally survive on the remote.

.EXAMPLE
    .\sync_to_myriad.ps1
    .\sync_to_myriad.ps1 -Clean
#>
[CmdletBinding()]
param(
    [string] $LocalRoot    = (Split-Path -Parent $PSScriptRoot),
    [string] $RemoteParent = "coupling0729",
    [string] $LoginNode    = "login13.myriad.rc.ucl.ac.uk",
    [string] $RemoteUser   = $env:USERNAME,
    [string] $KeyFile      = "$env:USERPROFILE\.ssh\id_ed25519",
    [switch] $Clean
)

$ErrorActionPreference = "Stop"

# Windows native OpenSSH only. The ssh that ships with Git Bash (OpenSSH 10.x)
# fails authentication against Myriad even with a valid key.
$SSH     = "C:\Windows\System32\OpenSSH\ssh.exe"
$SCP     = "C:\Windows\System32\OpenSSH\scp.exe"
$SshOpts = @("-i", $KeyFile, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes")
$Remote  = "$RemoteUser@$LoginNode"

if (-not (Test-Path -LiteralPath $LocalRoot)) { throw "LocalRoot not found: $LocalRoot" }
if (-not (Test-Path -LiteralPath $KeyFile))   { throw "SSH key not found: $KeyFile" }

# .NET rather than Split-Path: in PowerShell 5.1 -LiteralPath and -Parent/-Leaf
# sit in mutually exclusive parameter sets and cannot be combined.
$trimmed = $LocalRoot.TrimEnd('\')
$parent  = [System.IO.Path]::GetDirectoryName($trimmed)
$leaf    = [System.IO.Path]::GetFileName($trimmed)

# OneDrive "files on demand" can leave a cloud placeholder in place of the real
# file. tar would happily pack the stub and the remote would get an empty shell.
Write-Host "checking OneDrive placeholders..." -ForegroundColor Cyan
$stubs = Get-ChildItem -LiteralPath $LocalRoot -Recurse -File -ErrorAction SilentlyContinue |
         Where-Object { $_.Attributes -match 'Offline' -or $_.Attributes -match 'RecallOnDataAccess' }
if ($stubs) {
    $stubs | Select-Object -First 10 | ForEach-Object { Write-Warning "  placeholder: $($_.FullName)" }
    throw "$($stubs.Count) file(s) are OneDrive placeholders, not real content. Right-click the folder -> 'Always keep on this device', wait for sync, then re-run."
}

# Shell scripts have to arrive with LF endings and no BOM. A CRLF jobscript
# fails on Myriad as "$'\r': command not found", and a BOM makes the first line
# unreadable -- both opaque enough to be worth preventing at source.
Write-Host "normalising shell scripts..." -ForegroundColor Cyan
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$fixed = 0
Get-ChildItem -LiteralPath $LocalRoot -Recurse -File -Filter *.sh -ErrorAction SilentlyContinue |
    ForEach-Object {
        $raw  = [System.IO.File]::ReadAllText($_.FullName)
        $norm = $raw.TrimStart([char]0xFEFF) -replace "`r`n", "`n" -replace "`r", "`n"
        if ($norm -ne $raw) {
            [System.IO.File]::WriteAllText($_.FullName, $norm, $utf8NoBom)
            Write-Host "  normalised $($_.Name)" -ForegroundColor Yellow
            $fixed++
        }
    }
if ($fixed -eq 0) { Write-Host "  already clean" -ForegroundColor DarkGray }

$exclude = @(
    "--exclude=__pycache__", "--exclude=*.pyc", "--exclude=.pytest_cache",
    "--exclude=.git", "--exclude=.venv*",
    "--exclude=results", "--exclude=hpc_results"
)

$tarball = Join-Path $env:TEMP "coupling_sync.tar.gz"
if (Test-Path -LiteralPath $tarball) { Remove-Item -LiteralPath $tarball -Force }

# --format=ustar keeps bsdtar from emitting the pax extended headers that GNU
# tar on the far end reports as unknown SCHILY.fflags keywords. This bsdtar
# build has no 'gnu' format; ustar is universal and the paths here are well
# inside its 100/155 char name limits.
Write-Host "packing $leaf ..." -ForegroundColor Cyan
& tar.exe -czf $tarball --format=ustar -C $parent @exclude $leaf
if ($LASTEXITCODE -ne 0) { throw "tar failed with exit code $LASTEXITCODE" }
$kb = [math]::Round((Get-Item -LiteralPath $tarball).Length / 1KB, 1)
Write-Host "  $kb KB" -ForegroundColor DarkGray

Write-Host "uploading to ${Remote}:$RemoteParent/ ..." -ForegroundColor Cyan
& $SCP @SshOpts $tarball "${Remote}:$RemoteParent/"
if ($LASTEXITCODE -ne 0) { throw "scp failed with exit code $LASTEXITCODE" }

# Deliberately quote-free: Windows ssh.exe strips quotes out of the remote
# command, which silently turns quoted pipes and nested commands into garbage.
#
# bsdtar on Windows records every directory as 0555 -- no owner write bit --
# because Windows has no directory write mode for it to map. Extracting that
# straight gives read-only directories that the extraction itself then cannot
# write into. --delay-directory-restore holds off applying archived directory
# modes until the end, and the chmod afterwards puts the write bits back.
#
# The leading chmod must precede the -Clean purge: an earlier interrupted sync
# can leave 0555 directories behind that rm itself cannot descend into.
$tbName = [System.IO.Path]::GetFileName($tarball)
$purge  = if ($Clean) { "rm -rf $leaf;" } else { "" }
$cmd    = "cd ~/$RemoteParent && chmod -R u+rwX $leaf 2>/dev/null; $purge tar --delay-directory-restore -xzf $tbName && chmod -R u+rwX $leaf && rm -f $tbName && echo REMOTE_FILES=`$(find $leaf -type f | wc -l)"

Write-Host "unpacking on Myriad..." -ForegroundColor Cyan
& $SSH @SshOpts $Remote $cmd
if ($LASTEXITCODE -ne 0) { throw "remote unpack failed with exit code $LASTEXITCODE" }

Remove-Item -LiteralPath $tarball -Force
Write-Host "synced -> ${Remote}:~/$RemoteParent/$leaf" -ForegroundColor Green
