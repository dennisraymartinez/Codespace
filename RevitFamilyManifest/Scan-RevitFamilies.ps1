<#
.SYNOPSIS
    Scans local drives for Revit family files (.rfa) and writes a manifest CSV.

.DESCRIPTION
    Walks one or more roots (default C:\), collecting every .rfa it can reach.
    Prunes the folders that otherwise drown a families manifest: Windows,
    Program Files, recycle bins, Revit collaboration caches, temp folders, and
    Revit's own incremental family backups (Chair.0001.rfa).

    Output is a CSV whose first four columns match the existing Drive-based
    Revit Family Manifest sheet (Root / Name / Extension / Path) so it drops
    straight in, followed by the columns that make a local scan worth doing:
    folder, size, last modified, and the Revit release the family was saved in.

    Read-only. Nothing is written outside -OutDir.

.PARAMETER Roots
    Folders to scan. Default C:\. Scoping this (e.g. C:\Families, C:\Users\you\Documents)
    is dramatically faster and cleaner than a whole-volume sweep.

.PARAMETER ReadVersion
    Read each .rfa's BasicFileInfo stream to recover the Revit release it was
    saved in. Costs real time on a large library; worth it once.

.PARAMETER IncludeAutodeskLibraries
    Include the out-of-the-box Autodesk content under ProgramData\Autodesk.
    Off by default -- it adds thousands of families you did not author.

.PARAMETER IncludeBackups
    Include Revit's incremental family backups (Name.0001.rfa). Off by default.

.PARAMETER WebAppUrl
    Optional. Apps Script web app URL to POST results to, so the sheet updates
    without a manual import. Requires -Token.

.EXAMPLE
    .\Scan-RevitFamilies.ps1
    Scan C:\ with default exclusions, write CSV to the Desktop.

.EXAMPLE
    .\Scan-RevitFamilies.ps1 -Roots 'C:\.0REVIT FAMILIES' -ReadVersion -OutDir 'G:\My Drive\RevitManifest'
    Scan one library with version detection, land the CSV in Google Drive for Desktop.

.EXAMPLE
    .\Scan-RevitFamilies.ps1 -WebAppUrl 'https://script.google.com/macros/s/AKfy.../exec' -Token 'your-token'
    Scan and push straight into the sheet.
#>
[CmdletBinding()]
param(
    [string[]]$Roots = @('C:\'),
    [string]$OutDir,
    [string]$OutFile,
    [string[]]$ExcludeDir = @(),
    [string[]]$Extension = @('.rfa'),
    [switch]$IncludeTemplates,
    [switch]$IncludeBackups,
    [switch]$IncludeAutodeskLibraries,
    [switch]$ReadVersion,
    [string]$CatalogSuffix = '_cat',
    [int]$MaxReadMB = 8,
    [string]$WebAppUrl,
    [string]$Token,
    [int]$BatchSize = 2000
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# --------------------------------------------------------------------------
# Folder pruning. Matched as case-insensitive substrings of the full path.
# --------------------------------------------------------------------------
$defaultExcludes = @(
    '\$recycle.bin'
    '\system volume information'
    '\windows\'
    '\$winreagent'
    '\program files\'
    '\program files (x86)\'
    '\appdata\local\temp'
    '\appdata\local\packages'
    '\appdata\local\package cache'
    '\collaborationcache'          # Revit local cache of workshared central models
    '\appdata\local\autodesk\revit\packages'
    '\onedrivetemp'
    '\node_modules'
    '\.git\'
    '\revit_backups'
)
if (-not $IncludeAutodeskLibraries) {
    $defaultExcludes += '\programdata\autodesk'
}
$excludes = @($defaultExcludes + $ExcludeDir) | ForEach-Object { $_.ToLowerInvariant() }

if ($IncludeTemplates) { $Extension += '.rft' }
$extSet = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
foreach ($e in $Extension) { [void]$extSet.Add($e) }

# Revit incremental family backups: Chair.0001.rfa
$backupPattern = '\.\d{4}\.(rfa|rft)$'

function Test-Excluded {
    param([string]$Path)
    $p = $Path.ToLowerInvariant()
    if (-not $p.EndsWith('\')) { $p += '\' }
    foreach ($x in $excludes) { if ($p.Contains($x)) { return $true } }
    return $false
}

# --------------------------------------------------------------------------
# Recover the Revit release from the .rfa's BasicFileInfo stream.
# The stream is UTF-16LE in modern releases, ANSI in older ones, so try both.
# --------------------------------------------------------------------------
function Get-RevitRelease {
    param([string]$Path, [int]$MaxBytes)
    try {
        $fs = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open,
                                            [System.IO.FileAccess]::Read,
                                            [System.IO.FileShare]::ReadWrite)
        try {
            $len = [int][Math]::Min([int64]$fs.Length, [int64]$MaxBytes)
            if ($len -le 0) { return '' }
            $buf = New-Object byte[] $len
            $read = 0
            while ($read -lt $len) {
                $n = $fs.Read($buf, $read, $len - $read)
                if ($n -le 0) { break }
                $read += $n
            }
            foreach ($enc in @([Text.Encoding]::Unicode, [Text.Encoding]::ASCII)) {
                $text = $enc.GetString($buf, 0, $read)
                if ($text -match 'Autodesk Revit (\d{4})')  { return $Matches[1] }
                if ($text -match 'Format:\s*(\d{4})')       { return $Matches[1] }
            }
        } finally { $fs.Dispose() }
    } catch { }
    return ''
}

# --------------------------------------------------------------------------
# Type catalogs.
#
# A type catalog is a .txt sitting beside the family with the same base name;
# without it the family loads with only its default type and says nothing about
# why. The filesystem cannot tell us a family *expects* one -- catalogs are
# optional -- so a missing .txt is only reported as a problem when the family
# name follows the library's catalog convention (-CatalogSuffix, default _cat).
# --------------------------------------------------------------------------
function Get-TypeCatalog {
    param([string]$RfaPath)

    $dir  = [System.IO.Path]::GetDirectoryName($RfaPath)
    $base = [System.IO.Path]::GetFileNameWithoutExtension($RfaPath)
    $txt  = [System.IO.Path]::Combine($dir, $base + '.txt')

    if ([System.IO.File]::Exists($txt)) {
        $types = ''
        try {
            $reader = [System.IO.File]::OpenText($txt)
            try {
                $n = 0
                $first = $true
                while ($null -ne ($line = $reader.ReadLine())) {
                    if ($first) { $first = $false; continue }   # parameter header row
                    if (-not [string]::IsNullOrWhiteSpace($line)) { $n++ }
                }
                $types = $n
            } finally { $reader.Dispose() }
        } catch { }
        return [pscustomobject]@{ Status = 'Present'; Types = $types }
    }

    if ($CatalogSuffix -and $base.EndsWith($CatalogSuffix, [StringComparison]::OrdinalIgnoreCase)) {
        return [pscustomobject]@{ Status = 'MISSING'; Types = '' }
    }

    return [pscustomobject]@{ Status = ''; Types = '' }
}

# --------------------------------------------------------------------------
# Stack-based walk. Get-ChildItem -Recurse cannot prune and cannot survive
# the access-denied wall on a full C:\ sweep; this can do both.
# --------------------------------------------------------------------------
function Get-FamilyFiles {
    param([string]$Root, [string]$RootLabel)

    $found   = New-Object 'System.Collections.Generic.List[object]'
    $stack   = New-Object 'System.Collections.Generic.Stack[string]'
    $dirCount = 0
    $denied   = 0
    $stack.Push($Root)

    while ($stack.Count -gt 0) {
        $dir = $stack.Pop()
        $dirCount++

        if ($dirCount % 250 -eq 0) {
            Write-Progress -Activity "Scanning $RootLabel" `
                           -Status "$dirCount folders | $($found.Count) families | queue $($stack.Count)" `
                           -CurrentOperation $dir
        }

        try {
            foreach ($sub in [System.IO.Directory]::EnumerateDirectories($dir)) {
                if (-not (Test-Excluded $sub)) { $stack.Push($sub) }
            }
        } catch [System.UnauthorizedAccessException] { $denied++ }
          catch { $denied++ }

        try {
            foreach ($file in [System.IO.Directory]::EnumerateFiles($dir)) {
                $ext = [System.IO.Path]::GetExtension($file)
                if (-not $extSet.Contains($ext)) { continue }
                if (-not $IncludeBackups -and $file -match $backupPattern) { continue }

                try   { $info = New-Object System.IO.FileInfo $file }
                catch { continue }

                $catalog = Get-TypeCatalog -RfaPath $info.FullName

                $found.Add([pscustomobject]@{
                    Root          = $RootLabel
                    Name          = $info.Name
                    Extension     = $ext.TrimStart('.')
                    Path          = $info.FullName
                    Folder        = $info.DirectoryName
                    SizeKB        = [math]::Round($info.Length / 1KB, 1)
                    Modified      = $info.LastWriteTime.ToString('yyyy-MM-dd HH:mm')
                    RevitRelease  = ''
                    TypeCatalog   = $catalog.Status
                    CatalogTypes  = $catalog.Types
                    Copies        = 1
                })
            }
        } catch [System.UnauthorizedAccessException] { $denied++ }
          catch { $denied++ }
    }

    Write-Progress -Activity "Scanning $RootLabel" -Completed
    Write-Host ("  {0,-40} {1,6} families in {2} folders ({3} skipped, no access)" -f `
                $RootLabel, $found.Count, $dirCount, $denied)
    return $found
}

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
$started = Get-Date
Write-Host ""
Write-Host "Revit family manifest - local scan" -ForegroundColor Cyan
Write-Host ("Started {0}" -f $started.ToString('ddd MMM dd yyyy HH:mm:ss'))
Write-Host ("Extensions: {0}   Backups: {1}   OOTB Autodesk content: {2}" -f `
            ($Extension -join ', '),
            $(if ($IncludeBackups) { 'included' } else { 'excluded' }),
            $(if ($IncludeAutodeskLibraries) { 'included' } else { 'excluded' }))
Write-Host ""

$all = New-Object 'System.Collections.Generic.List[object]'
foreach ($root in $Roots) {
    if (-not (Test-Path -LiteralPath $root)) {
        Write-Warning "Root not found, skipping: $root"
        continue
    }
    $full  = (Resolve-Path -LiteralPath $root).Path
    $label = [System.IO.Path]::GetFileName($full.TrimEnd('\'))
    if ([string]::IsNullOrWhiteSpace($label)) { $label = $full.TrimEnd('\') }  # a bare drive: "C:"
    foreach ($item in @(Get-FamilyFiles -Root $full -RootLabel $label)) { $all.Add($item) }
}

# Drop duplicate paths a caller can create by passing overlapping roots.
$seen   = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
$unique = New-Object 'System.Collections.Generic.List[object]'
foreach ($item in $all) { if ($seen.Add($item.Path)) { $unique.Add($item) } }

# Flag families that exist under more than one path -- the real point of a manifest.
$byName = @{}
foreach ($item in $unique) {
    $k = $item.Name.ToLowerInvariant()
    if (-not $byName.ContainsKey($k)) { $byName[$k] = 0 }
    $byName[$k]++
}
foreach ($item in $unique) { $item.Copies = $byName[$item.Name.ToLowerInvariant()] }

if ($ReadVersion -and $unique.Count -gt 0) {
    Write-Host ""
    Write-Host "Reading Revit release from $($unique.Count) families..."
    $maxBytes = $MaxReadMB * 1MB
    $i = 0
    foreach ($item in $unique) {
        $i++
        if ($i % 50 -eq 0) {
            Write-Progress -Activity 'Reading family versions' `
                           -Status "$i of $($unique.Count)" `
                           -PercentComplete (100 * $i / $unique.Count)
        }
        $item.RevitRelease = Get-RevitRelease -Path $item.Path -MaxBytes $maxBytes
    }
    Write-Progress -Activity 'Reading family versions' -Completed
}

$sorted  = @($unique | Sort-Object Root, Path)
$elapsed = (Get-Date) - $started

Write-Host ""
Write-Host ("Total families: {0}   Duplicated names: {1}   Elapsed: {2:mm\:ss}" -f `
            $sorted.Count,
            (@($byName.GetEnumerator() | Where-Object { $_.Value -gt 1 }).Count),
            $elapsed) -ForegroundColor Green

$missingCat = @($sorted | Where-Object { $_.TypeCatalog -eq 'MISSING' })
if ($missingCat.Count -gt 0) {
    Write-Host ""
    Write-Host ("Type catalogs missing: {0}" -f $missingCat.Count) -ForegroundColor Yellow
    Write-Host "  These load with only their default type. Filter TypeCatalog = MISSING."
}

if ($ReadVersion) {
    Write-Host ""
    Write-Host "By Revit release:"
    $sorted | Group-Object RevitRelease | Sort-Object Name | ForEach-Object {
        $rel = if ([string]::IsNullOrWhiteSpace($_.Name)) { '(unreadable)' } else { $_.Name }
        Write-Host ("  {0,-14} {1,6}" -f $rel, $_.Count)
    }
}

# --------------------------------------------------------------------------
# Write CSV
# --------------------------------------------------------------------------
if (-not $OutFile) {
    if (-not $OutDir) { $OutDir = [Environment]::GetFolderPath('Desktop') }
    if (-not (Test-Path -LiteralPath $OutDir)) {
        New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
    }
    $stamp   = $started.ToString('yyyyMMdd_HHmmss')
    $OutFile = Join-Path $OutDir "RFA_Manifest_$stamp.csv"
}

$sorted | Select-Object Root, Name, Extension, Path, Folder, SizeKB, Modified, RevitRelease,
                        TypeCatalog, CatalogTypes, Copies |
    Export-Csv -LiteralPath $OutFile -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "CSV written: $OutFile" -ForegroundColor Green

# --------------------------------------------------------------------------
# Optional push to the Apps Script web app
# --------------------------------------------------------------------------
if ($WebAppUrl) {
    if (-not $Token) { throw "-WebAppUrl requires -Token (the value stored in the sheet's MANIFEST_TOKEN script property)." }

    Write-Host ""
    Write-Host "Pushing to the sheet..."
    $scannedAt = $started.ToString('o')

    function Send-Batch {
        param([string]$Mode, [array]$Rows)
        $payload = @{
            token     = $Token
            mode      = $Mode
            roots     = $Roots
            scannedAt = $scannedAt
            rows      = @($Rows)
        } | ConvertTo-Json -Depth 5 -Compress

        for ($attempt = 1; $attempt -le 4; $attempt++) {
            try {
                $r = Invoke-RestMethod -Uri $WebAppUrl -Method Post `
                                       -ContentType 'application/json' -Body $payload
                if ($r.ok -ne $true) { throw "Sheet rejected the batch: $($r.error)" }
                return
            } catch {
                if ($attempt -eq 4) { throw }
                $wait = [math]::Pow(2, $attempt)
                Write-Warning "  $Mode failed (attempt $attempt): $($_.Exception.Message). Retrying in ${wait}s..."
                Start-Sleep -Seconds $wait
            }
        }
    }

    Send-Batch -Mode 'start' -Rows @()
    $total = $sorted.Count
    for ($i = 0; $i -lt $total; $i += $BatchSize) {
        $slice = $sorted[$i..([math]::Min($i + $BatchSize - 1, $total - 1))]
        $rows  = $slice | ForEach-Object {
            ,@($_.Root, $_.Name, $_.Extension, $_.Path, $_.Folder, $_.SizeKB, $_.Modified, $_.RevitRelease,
              $_.TypeCatalog, $_.CatalogTypes, $_.Copies)
        }
        Send-Batch -Mode 'append' -Rows $rows
        Write-Host ("  sent {0} of {1}" -f [math]::Min($i + $BatchSize, $total), $total)
    }
    Send-Batch -Mode 'finish' -Rows @()
    Write-Host "Sheet updated." -ForegroundColor Green
}

Write-Host ""
