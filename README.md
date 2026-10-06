# 🐟 SARDINE Lab Website

A modern, static website for the SARDINE Lab — powered by clean HTML/CSS/JS **and** a private Google Sheet for content. Edit the sheet, run a tiny updater, ship 🚀

Link: https://sardine-lab.github.io/

---

## ✨ What’s inside

```
.
├── index.html                # Home
├── news.html                 # News
├── publications.html         # Publications
├── projects.html             # Projects
├── assets/                   # App logic & styles (hand-written)
│   ├── main.js               # Site chrome, home widgets
│   ├── metadata.js           # Site metadata (uses data/streams.js)
│   ├── news.js               # News manager
│   ├── projects.js           # Projects manager
│   ├── publications.js       # Publications manager
│   ├── pagination.js         # Pagination widget
│   ├── team.js               # Team manager
│   └── style.css             # Tailwind + custom styles
├── data/                     # Auto-generated content (do not hand-edit)
│   ├── countries.js
│   ├── news.js
│   ├── photos.js
│   ├── projects.js
│   ├── publications.js
│   ├── streams.js
│   └── team.js
├── documentation/
│   ├── ADDING_CONTENT.md
│   └── STYLING.md
└── python/                   # Updater & helpers
    ├── data_update.py        # ← run this to refresh data/*.js
    ├── sheet_sync.py         # compares sheet and data/*.js; copies code edits back to the sheet
    ├── sync_state.json       # hashes from the last time both sides matched (written by the bot)
    ├── test_sheet_sync.py    # offline checks for sheet_sync.py
    ├── data_update.gs        # Apps Script bound to the sheet (menu + buttons)
    ├── data_update.sh
    ├── jsdata_to_tsv.py
    ├── build_publications.py
    ├── sardine-website-*.json  # service account key (private)
    └── data/                   # scratch / intermediate
```

---

## 🧭 How the data flows

```
Google Sheet (private)
   └── Tabs: Publications, News, Projects, Team, Streams, GroupPhotos
        ↓ (python/data_update.py)
data/*.js (pure JSON-ish, with backticks for rich fields)
        ↓
assets/*.js managers render HTML into the pages
```

* **data/\*.js** is **auto-generated**. Keep your **helper methods** in `assets/*.js`.
* Rich fields (`abstract`, `content`, `description`) accept **Markdown** (converted to HTML) or **raw HTML** and are emitted as **template literals** (`` `...` ``), so multiline content is painless.

---

## 🔁 Editing data in code

You can also edit `data/*.js` directly and push. The sheet then needs those edits, or the next "Update the website" would overwrite them:

1. Edit `data/*.js`, commit, push to `main`.
2. Within a minute the status line on the sheet's README tab turns 🔵 "GitHub has changes that are not in the sheet".
3. Click **UPDATE SPREADSHEET FROM GITHUB** (or 🐟 SARDINE → Update spreadsheet from GitHub…). Only the cells that differ are written; rows nobody changed keep their Markdown.

| Status line | Meaning | What to do |
|---|---|---|
| 🟢 in sync | sheet and `data/*.js` hold the same content | nothing |
| 🟡 sheet has changes | edited in the sheet, not published | Update the website |
| 🔵 GitHub has changes | `data/*.js` edited in code | Update spreadsheet from GitHub |
| 🔴 both changed | the same tab changed on both sides | decide which side wins (see `--force` below) |

"Update the website" refuses to publish while a tab is 🔵 or 🔴, so code edits are never overwritten silently.

From a terminal, in `python/` (same credentials as `data_update.py`):

```
python sheet_sync.py --sheet SHEET_ID --sa sa.json status         # which side changed, per tab
python sheet_sync.py --sheet SHEET_ID --sa sa.json pull           # dry run: what would change in the sheet
python sheet_sync.py --sheet SHEET_ID --sa sa.json pull --apply   # write it (add --force to overwrite sheet edits)
python test_sheet_sync.py                                         # offline checks
```

When writing entries by hand, give rich fields as HTML (`<p>…</p>`) and keep publication ids consecutive. Plain text or gaps in ids still reach the sheet, but the next publish rewrites them in the sheet's form (`<p>` added, ids renumbered). A field the sheet has no column for blocks the update of that tab until a column exists.





Made with 🐟 in Lisbon.
