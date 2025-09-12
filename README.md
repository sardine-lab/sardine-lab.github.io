# 🐟 SARDINE Lab Website

A modern, static website for the SARDINE Lab — powered by clean HTML/CSS/JS **and** a private Google Sheet for content. Edit the sheet, run a tiny updater, ship 🚀

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

## 🗂️ Google Sheet — how to edit

Link: https://docs.google.com/spreadsheets/d/196rbIKmYXG5nb5-5jDpFLOOxKgNSFwdcYD5ZDjLOO6E/edit?usp=sharing

> **Rule of thumb:** one row = one item. Don’t rename tabs or headers. Use commas to separate lists.


### Authoring tips

* **Lists:** separate with commas (e.g., `multimodal, efficiency, attention`).
* **Markdown is welcome** in `abstract`, `content`, `description`. Examples:

  * `**bold**`, `_italics_`, `[link](https://example.com)`, lists, code blocks.
* **HTML also works.** If a cell already contains HTML, it’s passed through unchanged.
* **Dates:** prefer `dd/mm/yyyy` (e.g., `15/11/2024`). ISO `yyyy-mm-dd` is fine too.

---

## 🔐 Private Sheet access (service account)

1. Create a **Service Account** in Google Cloud, enable **Google Sheets API**.
2. Download the **JSON key** (it lives at `python/sardine-website-*.json` in this repo).
3. Share the Google Sheet with the service account email **as Viewer**.


---

## 🔄 Refreshing the site data

From the repo root (or `cd python/`):

```bash
# (first time)
python3 -m venv python/env
pip3 install gspread google-auth markdown

# run updater
python3 python/data_update.py --sheet "196rbIKmYXG5nb5-5jDpFLOOxKgNSFwdcYD5ZDjLOO6E" --sa "sardine-website-42bf7a19e1ff.json"
```

What it does:

* Reads all tabs.
* Converts Markdown → HTML (or passes HTML through).
* Writes:

  * `data/publications.js` → `const publicationsData = [ ... ]`
  * `data/news.js` → `const newsData = [ ... ]`
  * `data/projects.js` → `const projectsData = { current: [...], past: [...] }`
  * `data/team.js` → grouped object (`faculties`, `postdocs`, `phds`, `researchers`, `mscs`)
  * `data/streams.js` → `const streamsData = { [keyword]: {...} }`
  * `data/photos.js` → `const GROUP_PHOTOS = [ ... ]`

Rich fields are emitted as **template literals** (`` `...` ``), so multiline HTML is preserved.

---

## 🧪 Local preview

Just open `index.html` in your browser (no server required).
If you use a local server, any static server works (e.g., `python -m http.server`).

---

## 🧩 Frontend notes

* Load order: **`data/streams.js` before `assets/metadata.js`** (metadata reads streams).

---

## 🧑‍💻 Content shapes (for devs)

```js
// data/publications.js
const publicationsData = [{
  id, type, title, authors, venue, year, award,
  abstract: `...HTML from Markdown...`,
  streams: ["multimodal", "efficiency"],
  links: { paper, code, demo, bibtex }
}];

// data/news.js
const newsData = [{
  date: "15/11/2024", // original string preserved
  type, title,
  content: `...HTML from Markdown...`,
  tags: ["eurohpc", "multilingual"]
}];

// data/projects.js
const projectsData = {
  current: [{ name, title, status, ..., description: `...` }],
  past:    [{ ... }]
};

// data/team.js
const teamData = {
  faculties: [{ ... }],
  postdocs:  [{ ... }],
  phds:      [{ ... }],
  researchers:[{ ... }],
  mscs:      [{ ... }]
};

// data/streams.js
const streamsData = {
  multilingual: { keyword, name, color, icon, description },
  ...
};

// data/photos.js
const GROUP_PHOTOS = [{ year: "2024", description, filename }, ...];
```

---

## 🧭 Conventions

* **IDs for publications:** larger = newer.
* **Dates:** `dd/mm/yyyy` preferred and ISO accepted.
* **Lists in cells:** separate with commas.
* **Streams:** keys must match `Streams.keyword`.

---

Made with 🐟 in Lisbon.
