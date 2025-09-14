#!/usr/bin/env python3
"""
data_update.py — Update ONLY the data literals in data/*.js while keeping helper functions.

Features:
- Tabs: Publications, News, Projects, Team, Streams, GroupPhotos
- Markdown → HTML for 'abstract' (publications), 'content' (news), 'description' (projects)
- Emits those fields as JS template literals (`...`) instead of quoted strings
- If a cell already looks like HTML, we pass it through (no markdown conversion)
- In-place replacement: only the right-hand literal of 'const <var> = ...' is changed

Auth:
  - Either pass --sa path/to/service_account.json
  - Or set env GOOGLE_APPLICATION_CREDENTIALS=/abs/path/to/service_account.json

Run:
  pip install gspread google-auth markdown
  python data_update.py --sheet YOUR_SHEET_ID --sa path/to/sa.json
"""

import argparse, json, os, re, sys
from typing import Any, Dict, List, Optional, Tuple

# =========================
# CONFIG — edit paths/names
# =========================
CONFIG = {
    "publications": {
        "worksheet": "Publications",
        "file_path": "../data/publications.js",
        "var_name": "publicationsData",   # Array
    },
    "news": {
        "worksheet": "News",
        "file_path": "../data/news.js",
        "var_name": "newsData",           # Array
    },
    "projects": {
        "worksheet": "Projects",
        "file_path": "../data/projects.js",
        "var_name": "projectsData",       # Array
    },
    "team": {
        "worksheet": "Team",
        "file_path": "../data/team.js",
        "var_name": "teamData",           # Object grouped by role
    },
    "streams": {
        "worksheet": "Streams",
        "file_path": "../data/streams.js",
        "var_name": "RESEARCH_STREAMS",        # Object keyed by keyword
    },
    "photos": {
        "worksheet": "GroupPhotos",
        "file_path": "../data/photos.js",
        "var_name": "GROUP_PHOTOS",       # Array
    },
}


# Fields that can contain HTML/Markdown and should be emitted with backticks
RICH_TEXT_KEYS = {"abstract", "content", "description", "bibtex"}

# =====================
# Google Sheets helpers
# =====================
def read_sheet(spreadsheet_id: str, worksheet: str, sa_path: Optional[str]) -> List[Dict[str, Any]]:
    import gspread
    from google.oauth2.service_account import Credentials
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    key_path = sa_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not key_path or not os.path.exists(key_path):
        sys.exit("Set --sa /path/to/service_account.json or env GOOGLE_APPLICATION_CREDENTIALS.")
    creds = Credentials.from_service_account_file(key_path, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(spreadsheet_id)
    ws = sh.worksheet(worksheet)
    rows = ws.get_all_records()
    print(f"[read] {worksheet}: {len(rows)} rows", file=sys.stderr)
    return rows

# ===========
# Transformers
# ===========
def to_int(x, default=None):
    try: return int(str(x).strip())
    except: return default

def split_list(x) -> List[str]:
    if x is None: return []
    return [t.strip() for t in str(x).replace("|", ",").replace(";", ",").split(",") if t.strip()]

def nonempty(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if v is None: continue
        if isinstance(v, (list, dict)) and not v: continue
        if v == "": continue
        out[k] = v
    return out

# ----- Markdown / HTML helpers -----
def looks_like_html(text: str) -> bool:
    if not text: return False
    # crude but effective: contains a tag-like pattern
    return bool(re.search(r"<[a-zA-Z][\s\S]*?>", text))

def as_html(text: str) -> str:
    """Return HTML string: pass through if looks like HTML, else convert Markdown → HTML."""
    if not text:
        return ""
    if looks_like_html(text):
        return text
    # Markdown → HTML
    import markdown as md
    return md.markdown(text, extensions=["extra", "sane_lists"])

# ----- Publications -----
def map_publication_row(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    title = r.get("title") or r.get("Title")
    if not title: return None
    return nonempty({
        "id": to_int(r.get("id") or r.get("ID")),
        "title": title,
        "authors": r.get("authors") or r.get("Authors") or "",
        "venue": r.get("venue") or r.get("Venue") or "",
        "year": to_int(r.get("year") or r.get("Year")),
        "type": (r.get("type") or r.get("Type") or "preprint").strip().lower(),
        "award": r.get("award") or r.get("Award") or "",
        "abstract": r.get("abstract") or r.get("Abstract") or "",
        "streams": split_list(r.get("streams") or r.get("Streams")),
        "links": nonempty({
            "paper": (r.get("link") or r.get("Link") or "").strip() or None,
            "code":  (r.get("code") or r.get("Code") or "").strip() or None,
            "demo":  (r.get("demo") or r.get("Demo") or "").strip() or None,
            "bibtex":(r.get("bibtex") or r.get("Bibtex") or "").strip() or None,
        }),
    })

# ----- News -----
def parse_news_date_for_sort(date_str: str) -> str:
    # Accepts 'dd/mm/yyyy' or 'yyyy-mm-dd'. Returns ISO 'yyyy-mm-dd' for sorting.
    s = (date_str or "").strip()
    if not s: return ""
    if "/" in s:
        try:
            d, m, y = s.split("/")
            return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
        except Exception:
            return s
    return s

def map_news_row(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    title = r.get("title") or r.get("Title")
    if not title: return None
    date_str = r.get("date") or r.get("Date") or ""
    return nonempty({
        "date": date_str.strip(),  # keep as entered for display
        "type": (r.get("type") or r.get("Type") or "news").strip().lower(),
        "title": title,
        "content": r.get("content") or r.get("Content") or "",
        "tags": split_list(r.get("tags") or r.get("Tags")),
        "_sort_key": parse_news_date_for_sort(date_str),
    })

# ----- Projects (array) -----
def map_project_row(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    name = r.get("name") or r.get("Name")
    if not name: return None
    return nonempty({
        "name": name,
        "title": r.get("title") or r.get("Title") or "",
        "status": (r.get("status") or r.get("Status") or "current").strip().lower(),
        "funding": r.get("funding") or r.get("Funding") or "",
        "pi": r.get("pi") or r.get("PI") or "",
        "period": r.get("period") or r.get("Period") or "",
        "team_members": split_list(r.get("team_members") or r.get("Team Members") or r.get("TeamMembers")),
        "collaborators": split_list(r.get("collaborators") or r.get("Collaborators")),
        "keywords": split_list(r.get("keywords") or r.get("Keywords")),
        "website": (r.get("website") or r.get("Website") or "").strip(),
        "figure": (r.get("figure") or r.get("Figure") or "").strip(),
        "publications": split_list(r.get("publications") or r.get("Publications")),
        "description": r.get("description") or r.get("Description") or "",
    })

# ----- Team (grouped object) -----
ROLE_MAP = {
    "faculty": "faculties", "professor": "faculties",
    "postdoc": "postdocs",
    "phd": "phds", "phd student": "phds",
    "researcher": "researchers",
    "msc": "mscs", "master": "mscs", "masters": "mscs", "ms student": "mscs",
}

def map_team_row(r: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    name = r.get("name") or r.get("Name")
    role_raw = (r.get("role") or r.get("Role") or "").strip().lower()
    if not name or not role_raw: return None
    key = ROLE_MAP.get(role_raw, role_raw + "s")
    grad = (r.get("graduation_year") or r.get("Graduation Year") or r.get("Graduation_Year") or "current")
    member = nonempty({
        "name": name,
        "role": r.get("role") or r.get("Role") or "",
        "position": r.get("position") or r.get("Position") or "",
        "image": r.get("image") or r.get("Image") or "",
        "advisor": r.get("advisor") or r.get("Advisor") or "",
        "co_advisor": r.get("co_advisor") or r.get("Co_Advisor") or r.get("Co-Advisor") or r.get("Co Advisor") or "",
        "start_year": r.get("start_year") or r.get("Start Year") or r.get("Start_Year") or "",
        "graduation_year": grad,
        "research_interests": split_list(r.get("research_interests") or r.get("Research Interests")),
        "previous_position": (r.get("previous_position") or r.get("Previous Position") or
                             ((r.get("position") or r.get("Position")) if str(grad).lower() != "current" else "")) or "",
        "links": nonempty({
            "website": (r.get("website") or r.get("Website") or "").strip() or None,
            "github": (r.get("github") or r.get("GitHub") or "").strip() or None,
            "linkedin": (r.get("linkedin")or r.get("LinkedIn") or "").strip() or None,
            "scholar": (r.get("scholar") or r.get("Scholar") or "").strip() or None,
        }),
        "country": r.get("country") or r.get("Country") or "",
        "city": r.get("city") or r.get("City") or "",
    })
    return key, member

# ----- Streams (object keyed by keyword) -----
def map_stream_row(r: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    key = (r.get("keyword") or r.get("Keyword") or "").strip()
    if not key: return None
    obj = nonempty({
        "keyword": key,
        "name": r.get("name") or r.get("Name") or key,
        "color": r.get("color") or r.get("Color") or "",
        "icon": r.get("icon") or r.get("Icon") or "",
        "description": r.get("description") or r.get("Description") or "",
    })
    return key, obj

# ----- Group Photos (array) -----
def map_photo_row(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    filename = (r.get("filename") or r.get("Filename") or "").strip()
    if not filename: return None
    year = r.get("year") or r.get("Year") or ""
    desc = r.get("description") or r.get("Description") or ""
    return {"year": str(year).strip(), "description": str(desc).strip(), "filename": filename}

# ===============================
# JS emitter with template fields
# ===============================
def js_template_escape(s: str) -> str:
    # Escape backticks and ${ to prevent accidental template interpolation
    return s.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

def js_dump(value: Any, indent: int = 2, key_stack: Tuple[str, ...] = ()) -> str:
    """
    Dump Python object to JS with:
      - standard JSON for most scalars/strings
      - template literals for strings when the current key is in RICH_TEXT_KEYS
    """
    sp = " " * indent
    if isinstance(value, dict):
        items = []
        for k, v in value.items():
            ks = key_stack + (k,)
            key_js = json.dumps(k)
            items.append(f"{key_js}: {js_dump(v, indent, ks)}")
        inner = (",\n" + " " * indent).join(items)
        return "{\n" + (" " * indent) + inner + "\n}"
    elif isinstance(value, list):
        parts = [js_dump(v, indent, key_stack) for v in value]
        if not parts:
            return "[]"
        inner = (",\n" + " " * indent).join(parts)
        return "[\n" + (" " * indent) + inner + "\n]"
    elif isinstance(value, str):
        # If last key is a rich text field, emit as template literal
        if key_stack and key_stack[-1] in RICH_TEXT_KEYS:
            return f"`{js_template_escape(value)}`"
        return json.dumps(value)
    elif value is True:
        return "true"
    elif value is False:
        return "false"
    elif value is None:
        return "null"
    else:
        # numbers
        return str(value)

# ===============================
# JS literal replace (safe-ish)
# ===============================
def replace_js_const_literal(source: str, var_name: str, new_literal: str) -> str:
    m = re.search(rf'\bconst\s+{re.escape(var_name)}\s*=\s*', source)
    if not m:
        raise ValueError(f"Could not find const {var_name} = in file.")
    i = m.end()

    def skip_ws_comments(s, pos):
        n = len(s)
        while pos < n:
            if s[pos] in ' \t\r\n':
                pos += 1
            elif s.startswith('//', pos):
                pos = s.find('\n', pos)
                if pos == -1: return n
            elif s.startswith('/*', pos):
                endc = s.find('*/', pos+2)
                if endc == -1: raise ValueError("Unterminated block comment")
                pos = endc + 2
            else:
                break
        return pos

    i = skip_ws_comments(source, i)
    if i >= len(source) or source[i] not in '[{`':
        # We always write arrays/objects; template literals only appear inside them.
        # Keep the same check as before for bracket start.
        if source[i] not in '[{':
            raise ValueError(f"Expected '[' or '{{' after const {var_name} =")

    open_char = source[i]
    close_char = ']' if open_char == '[' else '}'
    start = i
    depth = 0
    j = i
    n = len(source)
    in_string = False
    string_quote = ''
    escape = False
    in_line_comment = False
    in_block_comment = False

    while j < n:
        ch = source[j]
        nxt = source[j+1] if j+1 < n else ''

        if in_line_comment:
            if ch == '\n': in_line_comment = False
            j += 1; continue
        if in_block_comment:
            if ch == '*' and nxt == '/':
                in_block_comment = False; j += 2
            else:
                j += 1
            continue
        if in_string:
            if escape: escape = False
            elif ch == '\\': escape = True
            elif ch == string_quote: in_string = False
            j += 1; continue

        if ch == '/' and nxt == '/': in_line_comment = True; j += 2; continue
        if ch == '/' and nxt == '*': in_block_comment = True; j += 2; continue
        if ch in ('"', "'", '`'): in_string = True; string_quote = ch; j += 1; continue
        if ch == open_char: depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                end = j + 1
                break
        j += 1
    else:
        raise ValueError("Could not find matching closing bracket for data literal.")

    before = source[:start]
    after = source[end:]
    return before + new_literal + after

def write_fresh_file(path: str, var_name: str, literal: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = (
        "// Auto-generated — DO NOT EDIT\n"
        f"const {var_name} = {literal};\n"
        f"if (typeof module !== 'undefined') module.exports = {{ {var_name} }};\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"[ok] Created {path} with {var_name}", file=sys.stderr)

def inject_literal(path: str, var_name: str, literal: str):
    if not os.path.exists(path):
        write_fresh_file(path, var_name, literal)
        return
    src = open(path, "r", encoding="utf-8").read()
    try:
        new_src = replace_js_const_literal(src, var_name, literal)
        open(path, "w", encoding="utf-8").write(new_src)
        print(f"[ok] Updated {path} ({var_name})", file=sys.stderr)
    except ValueError:
        write_fresh_file(path, var_name, literal)

# =====
# Main
# =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True, help="Google Sheet ID")
    ap.add_argument("--sa", help="Path to service_account.json (optional if env set)")
    args = ap.parse_args()

    # Publications (array) — convert abstract to HTML, emit with backticks
    rows = read_sheet(args.sheet, CONFIG["publications"]["worksheet"], args.sa)
    pubs = [x for x in (map_publication_row(r) for r in rows) if x]
    for p in pubs:
        if "abstract" in p:
            p["abstract"] = as_html(p["abstract"])
    pubs.sort(key=lambda d: (d.get("year") or 0, d.get("id") or 0), reverse=True)
    inject_literal(CONFIG["publications"]["file_path"], CONFIG["publications"]["var_name"],
                   js_dump(pubs, indent=2))

    # News (array) — convert content to HTML, keep original date string for display
    rows = read_sheet(args.sheet, CONFIG["news"]["worksheet"], args.sa)
    news = [x for x in (map_news_row(r) for r in rows) if x]
    for n in news:
        if "content" in n:
            n["content"] = as_html(n["content"])
    news.sort(key=lambda d: d.get("_sort_key",""), reverse=True)
    for n in news: n.pop("_sort_key", None)
    inject_literal(CONFIG["news"]["file_path"], CONFIG["news"]["var_name"],
                   js_dump(news, indent=2))

    # Projects (grouped object) — convert description to HTML, group by status
    rows = read_sheet(args.sheet, CONFIG["projects"]["worksheet"], args.sa)
    projs = [x for x in (map_project_row(r) for r in rows) if x]
    for p in projs:
        if "description" in p:
            p["description"] = as_html(p["description"])

    grouped = {"current": [], "past": []}
    for p in projs:
        status = (p.get("status") or "current").lower()
        bucket = "current" if status == "current" else "past"  # treat 'completed' as past
        grouped[bucket].append(p)

    inject_literal(CONFIG["projects"]["file_path"], CONFIG["projects"]["var_name"],
                   js_dump(grouped, indent=2))

    # Team (grouped object)
    rows = read_sheet(args.sheet, CONFIG["team"]["worksheet"], args.sa)
    grouped_team: Dict[str, List[Dict[str, Any]]] = {}
    for m in (map_team_row(r) for r in rows):
        if not m: continue
        key, member = m
        grouped_team.setdefault(key, []).append(member)
    for k in list(grouped_team.keys()):
        grouped_team[k] = sorted(grouped_team[k], key=lambda m: m.get("name",""))
    inject_literal(CONFIG["team"]["file_path"], CONFIG["team"]["var_name"],
                   js_dump(grouped_team, indent=2))

    # Streams (object keyed by keyword)
    rows = read_sheet(args.sheet, CONFIG["streams"]["worksheet"], args.sa)
    streams: Dict[str, Dict[str, Any]] = {}
    for m in (map_stream_row(r) for r in rows):
        if not m: continue
        key, obj = m
        streams[key] = obj
    inject_literal(CONFIG["streams"]["file_path"], CONFIG["streams"]["var_name"],
                   js_dump(streams, indent=2))

    # Group Photos (array) — newest year first
    rows = read_sheet(args.sheet, CONFIG["photos"]["worksheet"], args.sa)
    photos = [x for x in (map_photo_row(r) for r in rows) if x]
    def year_key(p):
        try: return int(p.get("year") or 0)
        except: return 0
    photos.sort(key=year_key, reverse=True)
    inject_literal(CONFIG["photos"]["file_path"], CONFIG["photos"]["var_name"],
                   js_dump(photos, indent=2))

if __name__ == "__main__":
    main()


