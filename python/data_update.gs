/**
 * ┌───────────────────────────────────────────────────────────────────┐
 * │ SARDINE: Trigger GitHub Actions from Google Sheets                │
 * └───────────────────────────────────────────────────────────────────┘
 * Adds a menu "🐟 SARDINE" with:
 *   - "Update website…"                  sheet -> GitHub (refresh-data.yml)
 *   - "Update spreadsheet from GitHub…"  GitHub -> sheet (sheet-sync.yml, mode pull)
 *   - "Check sync status"                refreshes the status line on the README tab
 * Each one dispatches a GitHub workflow_dispatch; the workflow does the work.
 *
 * Buttons on the README tab are drawings with one of these functions assigned
 * (right-click the drawing → ⋮ → Assign script): updateWebsite, updateSheetFromGitHub.
 *
 * Required Script Property:
 *   GITHUB_TOKEN  — a fine-scoped PAT with repo + workflow scopes
 *                   (Settings → Script properties → Script properties)
 */

const CFG = {
  owner: 'sardine-lab',
  repo: 'sardine-lab.github.io',
  workflow: 'refresh-data.yml',         // .github/workflows/<this file>
  syncWorkflow: 'sheet-sync.yml',       // GitHub -> sheet, and the status line
  ref: 'main'                           // branch to run on
};

// Tabs whose rows end up in data/*.js, and where the status line lives
const CONTENT_TABS = ['Publications', 'News', 'Projects', 'Team', 'Streams', 'GroupPhotos'];
const STATUS_TAB = 'README';
const STATUS_PREFIX = 'Sync status:';

function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu('🐟 SARDINE')
    .addItem('Update website…', 'updateWebsite')
    .addItem('Update spreadsheet from GitHub…', 'updateSheetFromGitHub')
    .addItem('Check sync status', 'checkSyncStatus')
    .addToUi();
}

/**
 * Marks the status line as soon as a content tab is edited by hand.
 * The workflows replace it with the real comparison when they run.
 */
function onEdit(e) {
  try {
    if (!e || CONTENT_TABS.indexOf(e.range.getSheet().getName()) < 0) return;
    setStatus_('🟡 Edited since the last check. Click UPDATE THE WEBSITE to publish.');
  } catch (err) {
    // the status line is cosmetic
  }
}

function updateWebsite() {
  const ui = SpreadsheetApp.getUi();
  const sheetId = SpreadsheetApp.getActive().getId();

  // Optional: ask for a commit message
  const resp = ui.prompt(
    'Update website',
    'Optional commit message (leave blank for default):',
    ui.ButtonSet.OK_CANCEL
  );
  if (resp.getSelectedButton() !== ui.Button.OK) return;

  const commitMessage = resp.getResponseText().trim() ||
    `chore(data): refresh from Google Sheets (${new Date().toISOString()})`;

  try {
    dispatchWorkflow_({
      commit_message: commitMessage,
      sheet_id: sheetId
    });
    const actionsUrl = `https://github.com/${CFG.owner}/${CFG.repo}/actions`;
    SpreadsheetApp.getActive().toast('Workflow dispatched. Opening Actions…', 'SARDINE', 5);
    const html = HtmlService.createHtmlOutput(
      `<div style="font:14px/1.4 system-ui">
         <p>✅ Dispatched! GitHub Actions will pull the Sheet, update <code>data/*.js</code>, commit, and deploy.</p>
         <p>If GitHub has edits this sheet does not have yet, nothing is published and the status line on the README tab says so.</p>
         <p><a href="${actionsUrl}" target="_blank">Open Actions</a></p>
       </div>`
    ).setWidth(420).setHeight(190);
    ui.showModelessDialog(html, 'SARDINE — Update launched');
  } catch (e) {
    ui.alert('Failed to dispatch workflow:\n' + (e && e.message ? e.message : e));
  }
}

/**
 * GitHub -> sheet: copies rows edited in data/*.js into this spreadsheet.
 * Tabs with changes that were not published yet are left alone.
 */
function updateSheetFromGitHub() {
  const ui = SpreadsheetApp.getUi();
  const resp = ui.alert(
    'Update spreadsheet from GitHub',
    'Rows that were edited in the repository (data/*.js) will be copied into this spreadsheet.\n' +
    'Tabs with changes that are not published yet are left alone.\n\nContinue?',
    ui.ButtonSet.OK_CANCEL
  );
  if (resp !== ui.Button.OK) return;

  try {
    dispatchWorkflow_({ mode: 'pull', sheet_id: SpreadsheetApp.getActive().getId() }, CFG.syncWorkflow);
    setStatus_('🔄 Updating this spreadsheet from GitHub… (about a minute)');
    SpreadsheetApp.getActive().toast('Started. The status line on the README tab changes when it is done.', 'SARDINE', 8);
  } catch (e) {
    ui.alert('Failed to dispatch workflow:\n' + (e && e.message ? e.message : e));
  }
}

function checkSyncStatus() {
  try {
    dispatchWorkflow_({ mode: 'status', sheet_id: SpreadsheetApp.getActive().getId() }, CFG.syncWorkflow);
    setStatus_('🔄 Checking… (about a minute)');
  } catch (e) {
    SpreadsheetApp.getUi().alert('Failed to dispatch workflow:\n' + (e && e.message ? e.message : e));
  }
}

/**
 * Writes the status line: the README cell in column A that starts with "Sync status:",
 * or A2 if there is none yet and A2 is empty.
 */
function setStatus_(text) {
  const sheet = SpreadsheetApp.getActive().getSheetByName(STATUS_TAB);
  if (!sheet) return;
  const values = sheet.getRange(1, 1, 40, 1).getValues();
  let row = -1;
  for (let i = 0; i < values.length; i++) {
    if (String(values[i][0]).indexOf(STATUS_PREFIX) === 0) { row = i + 1; break; }
  }
  if (row < 0) {
    if (String(values[1][0]).trim() !== '') return;
    row = 2;
  }
  sheet.getRange(row, 1).setValue(STATUS_PREFIX + ' ' + text);
}

/**
 * Calls GitHub "workflow_dispatch" API.
 */
function dispatchWorkflow_(inputs, workflow) {
  const token = PropertiesService.getScriptProperties().getProperty('GITHUB_TOKEN');
  if (!token) {
    throw new Error('Missing Script Property: GITHUB_TOKEN. Set it in Apps Script → Project Settings → Script properties.');
  }
  const url = `https://api.github.com/repos/${CFG.owner}/${CFG.repo}/actions/workflows/${encodeURIComponent(workflow || CFG.workflow)}/dispatches`;
  const payload = {
    ref: CFG.ref,
    inputs: inputs || {}
  };
  const res = UrlFetchApp.fetch(url, {
    method: 'post',
    contentType: 'application/json',
    payload: JSON.stringify(payload),
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28'
    },
    muteHttpExceptions: true
  });
  const code = res.getResponseCode();
  if (code !== 204) {
    throw new Error(`GitHub API ${code}: ${res.getContentText()}`);
  }
}