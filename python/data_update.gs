/**
 * ┌───────────────────────────────────────────────────────────────────┐
 * │ SARDINE: Trigger GitHub Actions from Google Sheets                │
 * └───────────────────────────────────────────────────────────────────┘
 * Adds a menu "🐟 SARDINE" → "Update website…" that dispatches a GitHub
 * workflow_dispatch. Optionally collects a commit message.
 *
 * Required Script Property:
 *   GITHUB_TOKEN  — a fine-scoped PAT with repo + workflow scopes
 *                   (Settings → Script properties → Script properties)
 */

const CFG = {
  owner: 'deep-spin',
  repo: 'sardine-website',
  workflow: 'refresh-data.yml',         // .github/workflows/<this file>
  ref: 'main'                           // branch to run on
};

function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu('🐟 SARDINE')
    .addItem('Update website…', 'updateWebsite')
    .addToUi();
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
         <p><a href="${actionsUrl}" target="_blank">Open Actions</a></p>
       </div>`
    ).setWidth(420).setHeight(140);
    ui.showModelessDialog(html, 'SARDINE — Update launched');
  } catch (e) {
    ui.alert('Failed to dispatch workflow:\n' + (e && e.message ? e.message : e));
  }
}

/**
 * Calls GitHub "workflow_dispatch" API.
 */
function dispatchWorkflow_(inputs) {
  const token = PropertiesService.getScriptProperties().getProperty('GITHUB_TOKEN');
  if (!token) {
    throw new Error('Missing Script Property: GITHUB_TOKEN. Set it in Apps Script → Project Settings → Script properties.');
  }
  const url = `https://api.github.com/repos/${CFG.owner}/${CFG.repo}/actions/workflows/${encodeURIComponent(CFG.workflow)}/dispatches`;
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