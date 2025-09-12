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





Made with 🐟 in Lisbon.
