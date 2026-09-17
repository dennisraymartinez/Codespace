# Local Drive Family Manifest

Extends the Revit Family Manifest sheet to cover families on a local drive
(`C:\`, a mapped server share, anywhere on the workstation) alongside the
families it already indexes in Google Drive.

## Why this needs two pieces

Apps Script runs on Google's servers, inside a sandbox. It can walk Google
Drive — which is what the existing manifest does, the 382 folders and 1,836
files in row 1 — but it has no route to a local filesystem. No permission,
scope, or API changes that. So the scan happens on the workstation and the
results are handed to the sheet.

| | |
|---|---|
| `Run Family Scan.cmd` | Double-click this. Runs the scan with your paths baked in. |
| `Scan-RevitFamilies.ps1` | The scanner itself. Keep it beside the .cmd. |
| `LocalManifest.gs` | Runs in the sheet. Adds the button, imports the results. |

The first four CSV columns (`Root`, `Name`, `Extension`, `Path`) match the
existing manifest layout, so the two sheets read the same way.

## Install

**1. Sheet side.** Extensions › Apps Script, add a file named
`LocalManifest.gs`, paste in the contents, save.

If the project already has an `onOpen()` (it does — that is where the existing
**Revit Tools** menu comes from), delete the `onOpen()` at the top of
`LocalManifest.gs` and add one line inside the existing one instead:

```javascript
function onOpen() {
  // ...whatever is already here...
  addLocalManifestMenu_(SpreadsheetApp.getUi());
}
```

Reload the spreadsheet. **Revit Tools › Local C: Drive Manifest** appears.

**2. Workstation side.** Put `Run Family Scan.cmd` and `Scan-RevitFamilies.ps1`
in the same folder, anywhere convenient. Nothing to install — Windows
PowerShell 5.1 ships with Windows and is enough.

Then double-click **`Run Family Scan.cmd`**. It sets the execution policy for
its own run and clears the blocked-file mark itself, so there is no PowerShell
to type and nothing to remember between scans.

To point it somewhere else, right-click the .cmd and choose Edit. Only the
lines at the top, under the comment banner, are meant to be changed:

```bat
set "LIB1=D:\Dropbox\.0REVIT FAMILIES\Manufacturers"
set "LIB2=D:\Dropbox\.0REVIT FAMILIES\DOWNLOAD"
set "OUTDIR=%USERPROFILE%\Desktop"
```

Scanning the two libraries as separate roots is deliberate: the `Root` column
then reads `Manufacturers` or `DOWNLOAD`, so the download dump can be filtered
out of any search with one click.

The sections below are the manual route, for a one-off scan with different
switches.

## Getting results into the sheet

Two routes. Pick one.

### Route A — through Google Drive (recommended)

No deployment, nothing publicly reachable. Requires Google Drive for Desktop.

1. Make a folder in Drive called `RevitManifest`, let Drive for Desktop sync it.
2. Scan, writing the CSV into that folder:

```powershell
.\Scan-RevitFamilies.ps1 -Roots "C:\" -ReadVersion -OutDir "G:\My Drive\RevitManifest"
```

3. In the sheet: **Revit Tools › Local C: Drive Manifest › Import newest scan
   from Drive.**

### Route B — straight into the sheet

The scanner POSTs results in; no manual import step.

1. **Revit Tools › Local C: Drive Manifest › Settings...**, and type `NEW` at
   the token prompt to generate one. Copy it.
2. In the Apps Script editor: Deploy › New deployment › Web app. Execute as
   yourself, access **Anyone**. Copy the `/exec` URL.
3. Scan and push:

```powershell
.\Scan-RevitFamilies.ps1 -Roots "C:\" -ReadVersion `
    -WebAppUrl "https://script.google.com/macros/s/.../exec" -Token "your-token"
```

Anyone holding both that URL and the token can write to the sheet. Treat the
pair as a password; regenerate the token from Settings if it leaks.

**Revit Tools › Local C: Drive Manifest › Get the scanner command...** builds
both commands with your settings already filled in, ready to copy.

## What lands in the sheet

A tab named `LOCAL C DRIVE`, with a `Last built:` summary in row 1 matching the
existing manifest, headers in row 2, a filter, and duplicate family names
shaded red.

| Column | |
|---|---|
| Root | The scan root the family was found under |
| Name | File name |
| Extension | `rfa`, or `rft` with `-IncludeTemplates` |
| Path | Full path |
| Folder | Containing folder |
| Size KB | |
| Modified | Last write time |
| Revit Release | Release the family was saved in — only with `-ReadVersion` |
| Type Catalog | `Present`, `MISSING`, or blank — see below |
| Catalog Types | How many types the catalog defines |
| Copies | How many times this file name appears in the scan |

`Revit Release` is read out of the family's `BasicFileInfo` stream without
opening Revit. It is the column worth having: it tells you which families will
force an upgrade prompt before you load them, and which half of the library is
still back on an old release. Blank means the stream could not be read — some
very old or third-party-generated families do not carry it.

`Type Catalog` is the one that catches silent breakage. A type catalog is a
`.txt` sitting beside the family with the same base name, listing the model
range. If it goes missing, the family still loads — with only its default
type, and no error explaining why. Families get separated from their catalogs
constantly when content is downloaded, re-extracted, or copied between folders.

The filesystem cannot tell you a family *expects* a catalog, since catalogs are
optional. So the column reports what it can actually know:

| Value | Meaning |
|---|---|
| `Present` | A matching `.txt` is there. `Catalog Types` counts the types in it. |
| `MISSING` | No `.txt`, and the family name ends in the catalog suffix — so it was meant to have one. |
| blank | No `.txt` and no suffix. Almost certainly a single-type family, nothing wrong. |

`MISSING` depends on your naming convention. It defaults to `_cat`; change it
with `-CatalogSuffix`, or pass `-CatalogSuffix ''` to only ever report
`Present`.

`Copies` is the other one. The current Drive manifest already shows the same
family at several paths; on a local scan that number is usually worse, and it
is the list to work off when consolidating a library down to one authoritative
copy per family.

## Switches

| | |
|---|---|
| `-Roots "C:\Families","D:\Content"` | Scan specific libraries. Much faster than a whole volume, and far less noise. |
| `-ReadVersion` | Record the Revit release each family was saved in. Adds real time on a large library; worth running once. |
| `-IncludeAutodeskLibraries` | Include out-of-the-box Autodesk content under `ProgramData\Autodesk`. Off by default — it buries your own families under thousands you did not author. |
| `-IncludeBackups` | Include Revit's incremental family backups (`Hood.0001.rfa`). Off by default. |
| `-IncludeTemplates` | Include `.rft` family templates. |
| `-ExcludeDir` | Extra path fragments to prune, e.g. `-ExcludeDir '\archive\','\_superseded\'` |
| `-OutDir` / `-OutFile` | Where the CSV goes. Defaults to the Desktop. |

Pruned by default: `Windows`, `Program Files`, recycle bins, temp and package
folders, `ProgramData\Autodesk`, and Revit's `CollaborationCache` — the local
cache of workshared centrals, which otherwise contributes a large number of
families that are not really yours.

## Notes

- The scan is read-only. It opens families with `FileShare.ReadWrite`, so it
  is safe to run with Revit open.
- Folders the account cannot read are skipped and counted, not fatal — a full
  `C:\` sweep always hits some.
- A whole-volume scan takes a while, and much longer with `-ReadVersion`.
  Scoping `-Roots` to your actual library folders is almost always the better
  call.
- The scanner is Windows-only by design (`C:\`, `BasicFileInfo`), though the
  walk itself is plain .NET and runs under PowerShell 7 anywhere.

## Not covered

The manifest is a file-level inventory: what exists, where, how big, what
release. It does not read inside a family — no category, family/type
parameters, OmniClass, host behavior, or nested families, and nothing about
connected loads, clearances, or exhaust airflow. That needs the Revit API,
via pyRevit or a Dynamo graph, with Revit open. It is a reasonable next step
if you want the manifest to carry the equipment data rather than just the
file data.
