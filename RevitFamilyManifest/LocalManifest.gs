/**
 * Local C: drive family manifest -- sheet side.
 *
 * Apps Script runs on Google's servers and cannot read a local disk, so the
 * scanning is done by Scan-RevitFamilies.ps1 on the workstation. This file
 * provides the button, and two ways to get the scanner's results into the
 * sheet:
 *
 *   1. Drive import  -- the scanner writes its CSV into a Google Drive for
 *                       Desktop folder; this script picks up the newest one.
 *                       No deployment, nothing public. Recommended.
 *   2. Direct push   -- deploy this script as a web app; the scanner POSTs
 *                       results straight in. No manual step at all.
 *
 * To merge into an existing "Revit Tools" menu, delete the onOpen() below and
 * call addLocalManifestMenu_(ui) from the onOpen() you already have.
 */

var LOCAL_SHEET_NAME = 'LOCAL C DRIVE';
var LOCAL_HEADERS = ['Root', 'Name', 'Extension', 'Path', 'Folder',
                     'Size KB', 'Modified', 'Revit Release', 'Copies'];
var DEFAULT_DRIVE_FOLDER = 'RevitManifest';
var CSV_NAME_PATTERN = /^RFA_Manifest.*\.csv$/i;

function onOpen() {
  addLocalManifestMenu_(SpreadsheetApp.getUi());
}

function addLocalManifestMenu_(ui) {
  ui.createMenu('Revit Tools')
    .addSubMenu(ui.createMenu('Local C: Drive Manifest')
      .addItem('Get the scanner command...', 'showLocalScannerDialog')
      .addSeparator()
      .addItem('Import newest scan from Drive', 'importLocalManifestFromDrive')
      .addSeparator()
      .addItem('Settings...', 'configureLocalManifest'))
    .addToUi();
}

/* -------------------------------------------------------------------------
 * Settings
 * ---------------------------------------------------------------------- */

function props_() {
  return PropertiesService.getScriptProperties();
}

function configureLocalManifest() {
  var ui = SpreadsheetApp.getUi();
  var p = props_();

  var folder = ui.prompt(
    'Local manifest settings (1 of 2)',
    'Google Drive folder the scanner writes its CSV into.\n' +
    'On the workstation this is the Drive for Desktop path, e.g.\n' +
    'G:\\My Drive\\' + DEFAULT_DRIVE_FOLDER + '\n\n' +
    'Folder name (blank = ' + DEFAULT_DRIVE_FOLDER + '):',
    ui.ButtonSet.OK_CANCEL);
  if (folder.getSelectedButton() !== ui.Button.OK) return;
  p.setProperty('LOCAL_MANIFEST_FOLDER',
                folder.getResponseText().trim() || DEFAULT_DRIVE_FOLDER);

  var token = ui.prompt(
    'Local manifest settings (2 of 2)',
    'Shared token for the direct-push web app. Leave blank if you are only\n' +
    'using the Drive import. Anyone holding both the web app URL and this\n' +
    'token can write to this sheet, so treat it like a password.\n\n' +
    'Token (type NEW to generate one):',
    ui.ButtonSet.OK_CANCEL);
  if (token.getSelectedButton() !== ui.Button.OK) return;

  var t = token.getResponseText().trim();
  if (t.toUpperCase() === 'NEW') { t = Utilities.getUuid(); }
  if (t) {
    p.setProperty('MANIFEST_TOKEN', t);
    ui.alert('Token set', 'Token:\n\n' + t + '\n\nPass this to the scanner as -Token.',
             ui.ButtonSet.OK);
  } else {
    p.deleteProperty('MANIFEST_TOKEN');
  }
}

/* -------------------------------------------------------------------------
 * The button: hand the user a ready-to-run scanner command
 * ---------------------------------------------------------------------- */

function showLocalScannerDialog() {
  var p = props_();
  var folder = p.getProperty('LOCAL_MANIFEST_FOLDER') || DEFAULT_DRIVE_FOLDER;
  var token = p.getProperty('MANIFEST_TOKEN');
  var url = ScriptApp.getService().getUrl();

  var driveCmd =
    '.\\Scan-RevitFamilies.ps1 -Roots "C:\\" -ReadVersion ' +
    '-OutDir "G:\\My Drive\\' + folder + '"';

  var pushCmd = (token && url)
    ? '.\\Scan-RevitFamilies.ps1 -Roots "C:\\" -ReadVersion ' +
      '-WebAppUrl "' + url + '" -Token "' + token + '"'
    : null;

  var html =
    '<style>' +
      'body{font:13px/1.5 Roboto,Arial,sans-serif;margin:16px;color:#202124}' +
      'h3{margin:18px 0 6px;font-size:13px;text-transform:uppercase;' +
        'letter-spacing:.06em;color:#5f6368}' +
      'h3:first-child{margin-top:0}' +
      'pre{background:#f1f3f4;border:1px solid #dadce0;border-radius:4px;' +
        'padding:10px;white-space:pre-wrap;word-break:break-all;font-size:12px;margin:0}' +
      'button{margin-top:6px;padding:6px 14px;border:1px solid #dadce0;' +
        'background:#fff;border-radius:4px;cursor:pointer;font:inherit}' +
      'button:hover{background:#f8f9fa}' +
      'p{margin:6px 0}.note{color:#5f6368;font-size:12px}' +
    '</style>' +
    '<p>Apps Script cannot read <code>C:\\</code>. Run this on the workstation ' +
    'instead, then the sheet picks the results up.</p>' +
    '<h3>Scan and drop the CSV in Drive</h3>' +
    '<pre id="a">' + escapeHtml_(driveCmd) + '</pre>' +
    '<button onclick="copy(\'a\')">Copy</button>' +
    '<p class="note">Then: <b>Revit Tools &rsaquo; Local C: Drive Manifest &rsaquo; ' +
    'Import newest scan from Drive</b>.</p>';

  if (pushCmd) {
    html +=
      '<h3>Or push straight into this sheet</h3>' +
      '<pre id="b">' + escapeHtml_(pushCmd) + '</pre>' +
      '<button onclick="copy(\'b\')">Copy</button>' +
      '<p class="note">Contains the shared token. Do not paste it anywhere public.</p>';
  } else {
    html +=
      '<h3>Or push straight into this sheet</h3>' +
      '<p class="note">Not configured. Set a token under <b>Settings...</b>, then ' +
      'deploy this script as a web app (Deploy &rsaquo; New deployment &rsaquo; Web app, ' +
      'execute as you, access "Anyone").</p>';
  }

  html +=
    '<h3>Useful switches</h3>' +
    '<p class="note">' +
    '<code>-Roots "C:\\.0REVIT FAMILIES","D:\\Content"</code> scan specific libraries ' +
    '(much faster than a whole volume)<br>' +
    '<code>-ReadVersion</code> record the Revit release each family was saved in<br>' +
    '<code>-IncludeAutodeskLibraries</code> add the out-of-the-box Autodesk content<br>' +
    '<code>-IncludeBackups</code> add Revit\'s <code>Name.0001.rfa</code> backups<br>' +
    '<code>-IncludeTemplates</code> add <code>.rft</code> family templates' +
    '</p>' +
    '<script>function copy(id){' +
      'var r=document.createRange();r.selectNode(document.getElementById(id));' +
      'var s=getSelection();s.removeAllRanges();s.addRange(r);' +
      'document.execCommand("copy");s.removeAllRanges();}</script>';

  SpreadsheetApp.getUi().showModalDialog(
    HtmlService.createHtmlOutput(html).setWidth(560).setHeight(560),
    'Build manifest from a local drive');
}

function escapeHtml_(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/* -------------------------------------------------------------------------
 * Path 1: import the newest scan from Drive
 * ---------------------------------------------------------------------- */

function importLocalManifestFromDrive() {
  var ui = SpreadsheetApp.getUi();
  var folderName = props_().getProperty('LOCAL_MANIFEST_FOLDER') || DEFAULT_DRIVE_FOLDER;

  var folders = DriveApp.getFoldersByName(folderName);
  if (!folders.hasNext()) {
    ui.alert('Folder not found',
             'No Drive folder named "' + folderName + '".\n\n' +
             'Create it in Drive, let Drive for Desktop sync it, point the scanner ' +
             'at it with -OutDir, or change the name under Settings...',
             ui.ButtonSet.OK);
    return;
  }

  var newest = null;
  while (folders.hasNext()) {
    var files = folders.next().getFiles();
    while (files.hasNext()) {
      var f = files.next();
      if (!CSV_NAME_PATTERN.test(f.getName())) continue;
      if (!newest || f.getLastUpdated() > newest.getLastUpdated()) newest = f;
    }
  }

  if (!newest) {
    ui.alert('No scan found',
             'Nothing matching RFA_Manifest*.csv in "' + folderName + '".\n\n' +
             'Run the scanner first -- Revit Tools > Local C: Drive Manifest > ' +
             'Get the scanner command...',
             ui.ButtonSet.OK);
    return;
  }

  var rows = Utilities.parseCsv(newest.getBlob().getDataAsString('UTF-8'));
  if (rows.length < 2) {
    ui.alert('Empty scan', newest.getName() + ' has no data rows.', ui.ButtonSet.OK);
    return;
  }

  var body = rows.slice(1);  // drop the scanner's header
  writeLocalManifest_(body, {
    source: newest.getName(),
    scannedAt: newest.getLastUpdated()
  });

  ui.alert('Manifest imported',
           body.length + ' families from ' + newest.getName() + '\n\n' +
           'Sheet: ' + LOCAL_SHEET_NAME,
           ui.ButtonSet.OK);
}

/* -------------------------------------------------------------------------
 * Path 2: direct push from the scanner
 * ---------------------------------------------------------------------- */

function doPost(e) {
  var lock = LockService.getScriptLock();
  try {
    lock.waitLock(30000);

    var body = JSON.parse(e.postData.contents);
    var expected = props_().getProperty('MANIFEST_TOKEN');
    if (!expected || body.token !== expected) {
      return json_({ ok: false, error: 'bad token' });
    }

    var sheet = getLocalSheet_();
    var cache = CacheService.getScriptCache();

    if (body.mode === 'start') {
      resetLocalSheet_(sheet);
      sheet.getRange(1, 1).setValue('Scan in progress, started ' + new Date());
      sheet.getRange(2, 1, 1, LOCAL_HEADERS.length)
           .setValues([LOCAL_HEADERS]).setFontWeight('bold');
      sheet.setFrozenRows(2);
      cache.put('manifest_rows', '0', 3600);
      cache.put('manifest_roots', JSON.stringify(body.roots || []), 3600);
      return json_({ ok: true, mode: 'start' });
    }

    if (body.mode === 'append') {
      var rows = body.rows || [];
      if (rows.length) {
        sheet.getRange(sheet.getLastRow() + 1, 1, rows.length, LOCAL_HEADERS.length)
             .setValues(rows.map(padRow_));
      }
      var n = parseInt(cache.get('manifest_rows') || '0', 10) + rows.length;
      cache.put('manifest_rows', String(n), 3600);
      return json_({ ok: true, mode: 'append', total: n });
    }

    if (body.mode === 'finish') {
      var total = parseInt(cache.get('manifest_rows') || '0', 10);
      var roots = JSON.parse(cache.get('manifest_roots') || '[]');
      sheet.getRange(1, 1).setValue(summaryLine_({
        count: total,
        scannedAt: body.scannedAt ? new Date(body.scannedAt) : new Date(),
        source: 'direct scan of ' + (roots.join(', ') || 'local drive')
      }));
      finishFormatting_(sheet);
      return json_({ ok: true, mode: 'finish', total: total });
    }

    return json_({ ok: false, error: 'unknown mode: ' + body.mode });

  } catch (err) {
    return json_({ ok: false, error: String(err) });
  } finally {
    try { lock.releaseLock(); } catch (ignored) {}
  }
}

function json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj))
                       .setMimeType(ContentService.MimeType.JSON);
}

/* -------------------------------------------------------------------------
 * Shared writing
 * ---------------------------------------------------------------------- */

function getLocalSheet_() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  return ss.getSheetByName(LOCAL_SHEET_NAME) || ss.insertSheet(LOCAL_SHEET_NAME);
}

/** sheet.clear() leaves the filter behind, and a stale one makes createFilter() throw. */
function resetLocalSheet_(sheet) {
  var existing = sheet.getFilter();
  if (existing) existing.remove();
  sheet.setConditionalFormatRules([]);
  sheet.clear();
}

function padRow_(row) {
  var out = row.slice(0, LOCAL_HEADERS.length);
  while (out.length < LOCAL_HEADERS.length) out.push('');
  return out;
}

function summaryLine_(info) {
  var when = Utilities.formatDate(info.scannedAt,
                                  Session.getScriptTimeZone(),
                                  'EEE MMM dd yyyy HH:mm:ss zzz');
  return 'Last built: ' + when +
         ' | Source: ' + info.source +
         ' | Families found: ' + info.count;
}

function writeLocalManifest_(rows, info) {
  var sheet = getLocalSheet_();
  resetLocalSheet_(sheet);

  sheet.getRange(2, 1, 1, LOCAL_HEADERS.length)
       .setValues([LOCAL_HEADERS]).setFontWeight('bold');
  if (rows.length) {
    sheet.getRange(3, 1, rows.length, LOCAL_HEADERS.length)
         .setValues(rows.map(padRow_));
  }
  sheet.getRange(1, 1).setValue(summaryLine_({
    count: rows.length,
    scannedAt: info.scannedAt || new Date(),
    source: info.source
  }));

  sheet.setFrozenRows(2);
  finishFormatting_(sheet);
}

function finishFormatting_(sheet) {
  var last = sheet.getLastRow();
  if (last > 2) {
    if (!sheet.getFilter()) {
      sheet.getRange(2, 1, last - 1, LOCAL_HEADERS.length).createFilter();
    }

    // Flag every family whose name appears more than once in the scan.
    var copies = sheet.getRange(3, LOCAL_HEADERS.indexOf('Copies') + 1, last - 2, 1);
    var rule = SpreadsheetApp.newConditionalFormatRule()
                 .whenNumberGreaterThan(1)
                 .setBackground('#fce8e6')
                 .setRanges([copies])
                 .build();
    sheet.setConditionalFormatRules([rule]);
  }
  sheet.autoResizeColumns(1, 3);
}
