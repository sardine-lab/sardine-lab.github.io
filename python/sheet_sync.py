#!/usr/bin/env python3
"""
sheet_sync.py — the other half of data_update.py.

data_update.py goes sheet -> data/*.js. This script compares the two sides and goes
data/*.js -> sheet, so data files edited in code can be carried back to the sheet.

  python sheet_sync.py status              which side changed, per tab
  python sheet_sync.py pull                what would change in the sheet (dry run)
  python sheet_sync.py pull --apply        make the sheet match data/*.js
  python sheet_sync.py published           record that data/*.js was just generated from the sheet

Both sides are compared as the objects data_update.py would emit, so Markdown in the
sheet and the HTML it produces count as equal. A pull only touches the cells that
differ: unchanged rows keep their Markdown and formatting.

sync_state.json holds a hash per tab from the last time both sides matched. It is what
tells "the sheet changed" from "GitHub changed" once they differ.

Auth and sheet id work as in data_update.py (--sheet/--sa, or env SHEET_ID and
GOOGLE_APPLICATION_CREDENTIALS). Needs node on PATH to read data/*.js.
"""

import argparse, datetime, hashlib, json, os, re, subprocess, sys
from typing import Any, Dict, List, Optional, Tuple

import data_update as du

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(HERE, "sync_state.json")
README_TAB = "README"
BADGE_PREFIX = "Sync status:"

IN_SYNC, SHEET_AHEAD, GITHUB_AHEAD, DIVERGED, UNKNOWN = "in_sync", "sheet_ahead", "github_ahead", "diverged", "unknown"

# Field that identifies a row, per tab
KEY = {"publications": "title", "news": "title", "projects": "name",
       "team": "name", "streams": "keyword", "photos": "filename"}

# ==========================
# Canonical objects + hashes
# ==========================
def canon(value: Any) -> Any:
    return json.loads(json.dumps(value))

def digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def flatten(tab: str, value: Any) -> List[Dict[str, Any]]:
    """The items of one tab as a flat list, in the order data_update.py emits them."""
    if tab == "projects":
        return list(value.get("current", [])) + list(value.get("past", []))
    if tab == "team":
        return [m for members in value.values() for m in members]
    if tab == "streams":
        return list(value.values())
    return list(value)

# =================
# Repo side (node)
# =================
NODE_READER = r"""
const fs = require('fs'), vm = require('vm');
const out = {};
for (const [key, file, name] of JSON.parse(process.argv[1])) {
  const ctx = vm.createContext({});
  vm.runInContext(fs.readFileSync(file, 'utf8') + '\n;globalThis.__value = ' + name + ';', ctx, { filename: file });
  out[key] = ctx.__value;
}
process.stdout.write(JSON.stringify(out));
"""

def load_repo(data_dir: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate data/*.js and return the data literals, keyed like du.CONFIG."""
    specs = []
    for key, cfg in du.CONFIG.items():
        path = os.path.join(HERE, cfg["file_path"])
        if data_dir:
            path = os.path.join(data_dir, os.path.basename(cfg["file_path"]))
        specs.append([key, path, cfg["var_name"]])
    res = subprocess.run(["node", "-e", NODE_READER, json.dumps(specs)], capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit("Could not read data/*.js with node:\n" + res.stderr.strip())
    return json.loads(res.stdout)

# ===========================
# Sheet side (objects per tab)
# ===========================
def sheet_data(records: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """What data_update.py would write, given every tab's rows (keyed by worksheet name)."""
    return canon(du.build_data(lambda worksheet: records.get(worksheet, [])))

def tab_data(tab: str, rows: List[Dict[str, Any]]) -> Any:
    worksheet = du.CONFIG[tab]["worksheet"]
    return canon(du.build_data(lambda name: rows if name == worksheet else [])[tab])

def row_object(tab: str, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The object one sheet row turns into, or None if data_update.py would skip it."""
    items = flatten(tab, tab_data(tab, [row]))
    return items[0] if items else None

# ==========================================
# Objects -> sheet cells (inverse of map_*_row)
# ==========================================
class DateCell(str):
    """A dd/mm/yyyy string that should be stored as a real date."""

def _join(items) -> str:
    return ", ".join(items or [])

def _num(value):
    return int(value) if isinstance(value, str) and value.isdigit() else value

def cells_for(tab: str, o: Dict[str, Any]) -> Dict[str, Any]:
    """Cell values for one object, keyed by normalised column name."""
    g = lambda k: o.get(k, "")
    links = o.get("links") or {}
    if tab == "publications":
        return {"id": g("id"), "type": g("type"), "title": g("title"), "authors": g("authors"),
                "venue": g("venue"), "year": g("year"), "award": g("award"),
                "link": links.get("paper", ""), "abstract": g("abstract"),
                "code": links.get("code", ""), "demo": links.get("demo", ""),
                "bibtex": links.get("bibtex", ""), "streams": _join(o.get("streams"))}
    if tab == "news":
        return {"date": DateCell(g("date")), "type": g("type"), "title": g("title"),
                "tags": _join(o.get("tags")), "content": g("content")}
    if tab == "projects":
        return {"name": g("name"), "title": g("title"), "status": g("status"), "pi": g("pi"),
                "funding": g("funding"), "period": g("period"),
                "team_members": _join(o.get("team_members")),
                "collaborators": _join(o.get("collaborators")), "keywords": _join(o.get("keywords")),
                "website": g("website"), "figure": g("figure"),
                "publications": _join(o.get("publications")), "description": g("description")}
    if tab == "team":
        grad = g("graduation_year")
        return {"role": g("role"), "name": g("name"), "image": g("image"), "position": g("position"),
                "previous_position": g("previous_position"), "advisor": g("advisor"),
                "co_advisor": g("co_advisor"), "start_year": g("start_year"),
                "graduation_year": "" if str(grad).lower() == "current" else grad,
                "research_interests": _join(o.get("research_interests")),
                "country": g("country"), "city": g("city"),
                "website": links.get("website", ""), "github": links.get("github", ""),
                "linkedin": links.get("linkedin", ""), "scholar": links.get("scholar", "")}
    if tab == "streams":
        return {"keyword": g("keyword"), "name": g("name"), "color": g("color"),
                "icon": g("icon"), "description": g("description")}
    if tab == "photos":
        return {"filename": g("filename"), "year": _num(g("year")), "description": g("description")}
    raise KeyError(tab)

def norm_header(h: str) -> str:
    return re.sub(r"[\s\-]+", "_", str(h).strip().lower())

def numericise(value):
    """Same conversion gspread applies in get_all_records()."""
    if isinstance(value, str) and "_" not in value:
        try: return int(value)
        except ValueError:
            try: return float(value)
            except ValueError: return value
    return value

def shown(value) -> Any:
    """How a written cell value comes back from get_all_records()."""
    if value is None or value == "":
        return ""
    return str(value) if isinstance(value, DateCell) else numericise(value)

# =============
# Tab snapshot
# =============
class Tab:
    """One worksheet as read from the sheet: headers, rows, and which columns hold formulas."""
    def __init__(self, key: str, title: str, gid: int, headers: List[str],
                 rows: List[Dict[str, Any]], formulas: Dict[str, str]):
        self.key, self.title, self.gid = key, title, gid
        self.headers, self.rows, self.formulas = headers, rows, formulas
        self.col = {norm_header(h): i for i, h in enumerate(headers) if str(h).strip()}
        self.header_of = {norm_header(h): h for h in headers if str(h).strip()}

    def positional_ids(self) -> bool:
        """True if the id column is the sheet's 'newest row on top' counting formula."""
        f = re.sub(r"\s+", "", self.formulas.get("id", "")).upper()
        return self.key == "publications" and bool(re.fullmatch(r"=COUNTA\([A-Z]+:[A-Z]+\)-ROW\(\)\+1", f))

    def refresh(self, rows: List[Dict[str, Any]]) -> None:
        """Recompute formula columns after rows were added, removed or moved."""
        if self.positional_ids():
            title, ident = self.header_of["title"], self.header_of["id"]
            n = sum(1 for r in rows if str(r.get(title, "")).strip())
            for i, r in enumerate(rows):
                r[ident] = n - i

# ========
# Planning
# ========
class Plan:
    def __init__(self, tab: Tab):
        self.tab = tab
        self.ops: List[tuple] = []          # ("set", row, col, value) | ("delete", row) | ("insert", row, cells) | ("move", src, dst)
        self.lines: List[str] = []          # human-readable summary
        self.added = self.changed = self.removed = self.moved = 0
        self.residual: List[str] = []       # differences in form that remain after the plan (ids, ordering, ...)
        self.lost: List[str] = []           # GitHub fields with no sheet column
        self.exact = True

def _fields(o: Dict[str, Any]) -> set:
    """Names of the non-empty fields of an object, one level into nested objects."""
    out = set()
    for k, v in o.items():
        if isinstance(v, dict):
            out |= {f"{k}.{kk}" for kk, vv in v.items() if vv not in ("", None, [], {})}
        elif v not in ("", None, [], {}):
            out.add(k)
    return out

def unrepresentable(tab: Tab, des: List[dict]) -> List[str]:
    """Fields on GitHub that no sheet column holds. Publishing from the sheet would drop them."""
    out = []
    for o in des:
        row = {h: "" for h in tab.headers if str(h).strip()}
        for c, v in cells_for(tab.key, o).items():
            if c in tab.header_of:
                row[tab.header_of[c]] = shown(v)
        lost = sorted(_fields(o) - _fields(row_object(tab.key, row) or {}))
        if lost:
            out.append(f'"{str(o.get(KEY[tab.key], ""))[:60]}": {", ".join(lost)}')
    return out

def _match(cur: List[Optional[dict]], des: List[dict], tab: str) -> Dict[int, int]:
    """Pair desired objects with current rows: by key first, then by overall similarity.

    A loose pairing is harmless: a paired row is edited in place instead of being
    removed and re-added, which keeps the cells that did not change.
    """
    key = KEY[tab]
    by_key: Dict[Any, List[int]] = {}
    for i, o in enumerate(cur):
        if o is not None:
            by_key.setdefault(o.get(key), []).append(i)
    pairs: Dict[int, int] = {}
    for j, o in enumerate(des):
        candidates = by_key.get(o.get(key))
        if candidates:
            pairs[j] = candidates.pop(0)
    left_cur = [i for i, o in enumerate(cur) if o is not None and i not in set(pairs.values())]
    for j, o in enumerate(des):
        if j in pairs or not left_cur:
            continue
        want = cells_for(tab, o)
        filled = [c for c, v in want.items() if v not in ("", None) and c != "id"]
        scored = [(sum(1 for c in filled if cells_for(tab, cur[i]).get(c) == want[c]), -i) for i in left_cur]
        score, neg_i = max(scored)
        if score >= 2:                                  # a renamed row, not a new one
            pairs[j] = -neg_i
            left_cur.remove(-neg_i)
    return pairs

def _simulate(tab: Tab, ops: List[tuple]) -> List[Dict[str, Any]]:
    """Apply ops to a copy of the tab's rows, as get_all_records() would then return them."""
    rows = [dict(r) for r in tab.rows]
    for op in ops:
        if op[0] == "set":
            rows[op[1]][tab.header_of[op[2]]] = shown(op[3])
        elif op[0] == "delete":
            del rows[op[1]]
        elif op[0] == "insert":
            row = {h: "" for h in tab.headers if str(h).strip()}
            for c, v in op[2].items():
                if c in tab.header_of and c not in tab.formulas:
                    row[tab.header_of[c]] = shown(v)
            rows.insert(op[1], row)
        elif op[0] == "move":
            rows.insert(op[2], rows.pop(op[1]))
    tab.refresh(rows)
    return rows

def _diff_items(tab: str, got: Any, want: Any) -> List[str]:
    key = KEY[tab]
    a, b = flatten(tab, got), flatten(tab, want)
    out: List[str] = []
    bk = {o.get(key): o for o in b}
    ak = {o.get(key): o for o in a}
    for k, o in bk.items():
        if k not in ak:
            out.append(f'"{k}" is on GitHub but would still be missing from the sheet')
        elif ak[k] != o:
            fields = sorted(f for f in set(o) | set(ak[k]) if o.get(f) != ak[k].get(f))
            out.append(f'"{k}": {", ".join(fields)} would still differ')
    out += [f'"{k}" would still be in the sheet but is not on GitHub' for k in ak if k not in bk]
    if not out and got != want:
        out.append("same items, different order or grouping")
    return out

def plan_tab(tab: Tab, want: Any, reorder: bool = False) -> Plan:
    """Work out the cell edits, inserts, deletes and moves that make this tab produce `want`."""
    name = tab.key
    plan = Plan(tab)
    des = flatten(name, want)
    plan.lost = unrepresentable(tab, des)
    cur = [row_object(name, r) for r in tab.rows]
    pairs = _match(cur, des, name)
    cur_of = pairs
    des_of = {i: j for j, i in pairs.items()}
    label = lambda o: str(o.get(KEY[name], ""))[:70]

    # 1. cell edits on rows that exist on both sides
    for i, o in enumerate(cur):
        if i not in des_of or o == des[des_of[i]]:
            continue
        have, need = cells_for(name, o), cells_for(name, des[des_of[i]])
        changed = [c for c in need if need[c] != have.get(c) and c in tab.col and c not in tab.formulas]
        for c in changed:
            plan.ops.append(("set", i, c, need[c]))
        skipped = [c for c in need if need[c] != have.get(c) and c not in changed and not (c == "id" and tab.positional_ids())]
        if changed:
            plan.changed += 1
            plan.lines.append(f'  ~ row {i + 2} "{label(o)}": {", ".join(changed)}')
        for c in skipped:
            plan.lines.append(f'  ! row {i + 2} "{label(o)}": cannot write {c} (no such column, or it is a formula)')

    # 2. rows that are no longer on GitHub. Rows data_update.py ignores (no title/name) are left alone.
    doomed = [i for i, o in enumerate(cur) if o is not None and i not in des_of]
    for i in sorted(doomed, reverse=True):
        plan.ops.append(("delete", i))
        plan.removed += 1
        plan.lines.append(f'  - row {i + 2} "{label(cur[i])}"')
    kept = [i for i in range(len(cur)) if i not in doomed]

    # 3. final row order: existing rows stay put and each new row goes just above the row
    #    that follows it on GitHub. With reorder=True the rows are put in GitHub's order.
    order = list(range(len(des)))
    if name == "publications":
        order.sort(key=lambda j: -(des[j].get("id") or 0))
    if reorder:
        seq = [("cur", cur_of[j]) if j in cur_of else ("new", j) for j in order]
        seq += [("cur", i) for i in kept if i not in des_of]
    else:
        seq = [("cur", i) for i in kept]
        nxt = None
        for j in reversed(order):
            if j in cur_of:
                nxt = ("cur", cur_of[j])
            else:
                seq.insert(seq.index(nxt) if nxt is not None else len(seq), ("new", j))
                nxt = ("new", j)

    live = [("cur", i) for i in kept]
    for p, tok in enumerate(seq):
        if tok[0] == "new":
            plan.ops.append(("insert", p, cells_for(name, des[tok[1]])))
            live.insert(p, tok)
            plan.added += 1
            plan.lines.append(f'  + row {p + 2} "{label(des[tok[1]])}"')
        else:
            q = live.index(tok)
            if q != p:
                plan.ops.append(("move", q, p))
                live.insert(p, live.pop(q))
                plan.moved += 1

    # 4. check the result before anything is written
    predicted = tab_data(name, _simulate(tab, plan.ops))
    if predicted != want:
        if not reorder and sorted(map(digest, flatten(name, predicted))) == sorted(map(digest, flatten(name, want))):
            return plan_tab(tab, want, reorder=True)
        plan.exact = False
        plan.residual = _diff_items(name, predicted, want)
    if plan.moved:
        plan.lines.append(f"  > {plan.moved} row(s) moved to match GitHub's order")
    return plan

# ==========================
# Plans -> Sheets API requests
# ==========================
EPOCH = datetime.date(1899, 12, 30)

def _cell(value, formula: bool = False) -> Dict[str, Any]:
    if value is None or value == "":
        return {}
    if formula:
        return {"userEnteredValue": {"formulaValue": value}}
    if isinstance(value, DateCell):
        m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", value.strip())
        if m:
            day = datetime.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            return {"userEnteredValue": {"numberValue": (day - EPOCH).days},
                    "userEnteredFormat": {"numberFormat": {"type": "DATE", "pattern": "dd/MM/yyyy"}}}
        return {"userEnteredValue": {"stringValue": str(value)}}
    if isinstance(value, bool):
        return {"userEnteredValue": {"boolValue": value}}
    if isinstance(value, (int, float)):
        return {"userEnteredValue": {"numberValue": value}}
    return {"userEnteredValue": {"stringValue": str(value)}}

def _update(gid: int, row: int, col: int, cells: List[Dict[str, Any]], fields: str) -> Dict[str, Any]:
    return {"updateCells": {"range": {"sheetId": gid, "startRowIndex": row, "endRowIndex": row + 1,
                                      "startColumnIndex": col, "endColumnIndex": col + len(cells)},
                            "rows": [{"values": cells}], "fields": fields}}

def _rows(gid: int, start: int, end: int) -> Dict[str, Any]:
    return {"sheetId": gid, "dimension": "ROWS", "startIndex": start, "endIndex": end}

def requests_for(plan: Plan) -> List[Dict[str, Any]]:
    """Translate a plan into spreadsheets.batchUpdate requests. Data row p is grid row p + 1."""
    tab, gid, reqs = plan.tab, plan.tab.gid, []
    n = len(tab.rows)
    for op in plan.ops:
        if op[0] == "set":
            cell = _cell(op[3])
            fields = "userEnteredValue" + (",userEnteredFormat.numberFormat" if "userEnteredFormat" in cell else "")
            reqs.append(_update(gid, op[1] + 1, tab.col[op[2]], [cell], fields))
        elif op[0] == "delete":
            reqs.append({"deleteDimension": {"range": _rows(gid, op[1] + 1, op[1] + 2)}})
            n -= 1
        elif op[0] == "move":
            reqs.append({"moveDimension": {"source": _rows(gid, op[1] + 1, op[1] + 2), "destinationIndex": op[2] + 1}})
        elif op[0] == "insert":
            p = op[1]
            if p == n and n > 0:
                # Appending: add the row above the last one and swap them, so it lands inside the table.
                reqs.append({"insertDimension": {"range": _rows(gid, p, p + 1), "inheritFromBefore": p > 1}})
                reqs.append({"moveDimension": {"source": _rows(gid, p + 1, p + 2), "destinationIndex": p}})
            else:
                reqs.append({"insertDimension": {"range": _rows(gid, p + 1, p + 2), "inheritFromBefore": p > 0}})
            n += 1
            width = max(tab.col.values()) + 1
            cells: List[Dict[str, Any]] = [{} for _ in range(width)]
            for c, i in tab.col.items():
                # formula columns reuse the first data row's formula as is (fine while it has no row-relative references)
                cells[i] = _cell(tab.formulas[c], formula=True) if c in tab.formulas else _cell(op[2].get(c, ""))
            reqs.append(_update(gid, p + 1, 0, [{k: v for k, v in c.items() if k == "userEnteredValue"} for c in cells],
                                "userEnteredValue"))
            for i, c in enumerate(cells):
                if "userEnteredFormat" in c:
                    reqs.append(_update(gid, p + 1, i, [{"userEnteredFormat": c["userEnteredFormat"]}],
                                        "userEnteredFormat.numberFormat"))
    return reqs

# ===========
# Live sheet
# ===========
class LiveSheet:
    """The Google Sheet, through gspread. Tests swap this for an in-memory stand-in."""
    def __init__(self, sheet_id: str, sa_path: Optional[str], write: bool = False):
        import gspread
        from google.oauth2.service_account import Credentials
        scope = "https://www.googleapis.com/auth/spreadsheets" + ("" if write else ".readonly")
        key_path = sa_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not key_path or not os.path.exists(key_path):
            sys.exit("Set --sa /path/to/service_account.json or env GOOGLE_APPLICATION_CREDENTIALS.")
        creds = Credentials.from_service_account_file(key_path, scopes=[scope])
        self.sh = gspread.authorize(creds).open_by_key(sheet_id)
        self._gids: Optional[Dict[str, int]] = None

    def gid(self, title: str) -> int:
        if self._gids is None:
            sheets = self.sh.fetch_sheet_metadata()["sheets"]
            self._gids = {s["properties"]["title"]: s["properties"]["sheetId"] for s in sheets}
        return self._gids[title]

    def read_tabs(self) -> Dict[str, Tab]:
        """Every tab in two requests. The Sheets API allows 60 reads a minute, and a
        publish runs this several times, so reading tab by tab runs out of quota."""
        titles = [cfg["worksheet"] for cfg in du.CONFIG.values()]
        shown_values = self.sh.values_batch_get([f"'{t}'" for t in titles])["valueRanges"]
        top_rows = self.sh.values_batch_get([f"'{t}'!1:2" for t in titles],
                                            params={"valueRenderOption": "FORMULA"})["valueRanges"]
        tabs = {}
        for key, title, grid, top in zip(du.CONFIG, titles, shown_values, top_rows):
            headers, rows = records(grid.get("values", []))
            first = (top.get("values", []) + [[], []])[1]
            formulas = {norm_header(h): v for h, v in zip(headers, first)
                        if str(h).strip() and isinstance(v, str) and v.startswith("=")}
            tabs[key] = Tab(key, title, self.gid(title), headers, rows, formulas)
        return tabs

    def batch_update(self, requests: List[Dict[str, Any]]) -> None:
        self.sh.batch_update({"requests": requests})

    def write_badge(self, text: str, color: Dict[str, float]) -> None:
        col_a = [r[0] if r else "" for r in self.sh.values_get(f"'{README_TAB}'!A1:A40").get("values", [])]
        row = next((i for i, v in enumerate(col_a) if str(v).startswith(BADGE_PREFIX)), None)
        if row is None:
            if len(col_a) > 1 and str(col_a[1]).strip():
                raise RuntimeError(f"{README_TAB}!A2 is not empty; put '{BADGE_PREFIX}' in the cell that should hold the badge.")
            row = 1
        cell = {"userEnteredValue": {"stringValue": text},
                "userEnteredFormat": {"backgroundColor": color, "textFormat": {"bold": True}}}
        self.sh.batch_update({"requests": [_update(self.gid(README_TAB), row, 0, [cell],
            "userEnteredValue,userEnteredFormat.backgroundColor,userEnteredFormat.textFormat.bold")]})

def records(values: List[List[Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Header list and row dicts from a tab's values, as gspread's get_all_records() builds them."""
    if not values:
        return [], []
    headers = [str(h) for h in values[0]]
    rows = []
    for raw in values[1:]:
        cells = list(raw) + [""] * (len(headers) - len(raw))
        rows.append({h: numericise(v) for h, v in zip(headers, cells)})
    return headers, rows

def read_tabs(sheet) -> Dict[str, Tab]:
    return sheet.read_tabs()

# ===============
# State + status
# ===============
def load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"tabs": {}}

def save_state(tabs: Dict[str, str], direction: str) -> bool:
    """Record hashes for tabs known to match on both sides. Returns True if the file changed."""
    state = load_state()
    merged = dict(state.get("tabs", {}), **tabs)
    if merged == state.get("tabs"):
        return False
    state = {"_": "Written by sheet_sync.py: content hash per tab from the last time the sheet and data/*.js matched.",
             "synced_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "direction": direction, "tabs": merged}
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.write("\n")
    return True

def compare(sheet_side: Dict[str, Any], repo_side: Dict[str, Any], base: Dict[str, str]) -> Dict[str, str]:
    status = {}
    for tab in du.CONFIG:
        s, g, b = digest(sheet_side[tab]), digest(canon(repo_side[tab])), base.get(tab)
        status[tab] = (IN_SYNC if s == g else UNKNOWN if b is None else
                       SHEET_AHEAD if g == b else GITHUB_AHEAD if s == b else DIVERGED)
    return status

WORDS = {IN_SYNC: "in sync", SHEET_AHEAD: "sheet has changes that are not on GitHub",
         GITHUB_AHEAD: "GitHub has changes that are not in the sheet",
         DIVERGED: "changed on both sides", UNKNOWN: "different (no record of when they last matched)"}

def badge(status: Dict[str, str]) -> Tuple[str, Dict[str, float]]:
    names = lambda kind: ", ".join(du.CONFIG[t]["worksheet"] for t, s in status.items() if s == kind)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    parts = []
    if names(DIVERGED):
        parts.append(f"🔴 Sheet and GitHub both changed ({names(DIVERGED)}). Ask before overwriting either side.")
    if names(GITHUB_AHEAD):
        parts.append(f"🔵 GitHub has changes that are not in the sheet ({names(GITHUB_AHEAD)}). Click UPDATE SPREADSHEET FROM GITHUB.")
    if names(SHEET_AHEAD):
        parts.append(f"🟡 The sheet has changes that are not on the website yet ({names(SHEET_AHEAD)}). Click UPDATE THE WEBSITE.")
    if names(UNKNOWN):
        parts.append(f"⚪ Sheet and GitHub differ ({names(UNKNOWN)}).")
    if not parts:
        parts.append("🟢 Sheet and GitHub are in sync.")
    worst = next(k for k in (DIVERGED, GITHUB_AHEAD, SHEET_AHEAD, UNKNOWN, IN_SYNC) if k in status.values())
    rgb = {DIVERGED: (0.96, 0.80, 0.80), GITHUB_AHEAD: (0.81, 0.89, 0.95), SHEET_AHEAD: (1.0, 0.95, 0.80),
           UNKNOWN: (0.94, 0.94, 0.94), IN_SYNC: (0.85, 0.92, 0.83)}[worst]
    return f"{BADGE_PREFIX} " + " ".join(parts) + f" (checked {stamp})", dict(zip(("red", "green", "blue"), rgb))

def print_status(status: Dict[str, str]) -> None:
    for tab, s in status.items():
        print(f"{du.CONFIG[tab]['worksheet']:14} {WORDS[s]}")

def try_badge(sheet, status: Dict[str, str]) -> None:
    try:
        sheet.write_badge(*badge(status))
        print("[ok] Badge updated", file=sys.stderr)
    except Exception as e:      # the badge is cosmetic: never fail a sync over it
        print(f"[warn] Could not write the badge: {e}", file=sys.stderr)

# =========
# Commands
# =========
def cmd_status(sheet, args) -> int:
    tabs = read_tabs(sheet)
    repo = load_repo()
    status = compare(sheet_data({t.title: t.rows for t in tabs.values()}), repo, load_state().get("tabs", {}))
    print_status(status)
    if args.record:
        save_state({t: digest(canon(repo[t])) for t, s in status.items() if s == IN_SYNC}, "check")
    if args.badge:
        try_badge(sheet, status)
    if args.guard_publish and any(s in (GITHUB_AHEAD, DIVERGED) for s in status.values()):
        print("\nNot publishing: GitHub has changes the sheet does not have, and publishing would overwrite them.\n"
              "Run 'Update spreadsheet from GitHub' first.", file=sys.stderr)
        return 3
    return 0

def cmd_pull(sheet, args) -> int:
    tabs = read_tabs(sheet)
    repo = canon(load_repo())
    status = compare(sheet_data({t.title: t.rows for t in tabs.values()}), repo, load_state().get("tabs", {}))
    requests, applied, blocked = [], [], []
    for key, tab in tabs.items():
        title, s = tab.title, status[key]
        if s == IN_SYNC:
            print(f"{title}: in sync")
            continue
        if s != GITHUB_AHEAD and not args.force:
            print(f"{title}: skipped, {WORDS[s]} (use --force to overwrite the sheet)")
            continue
        plan = plan_tab(tab, repo[key])
        if plan.lost:
            print(f"{title}: not updated. GitHub has fields the sheet has no column for, "
                  "and publishing from the sheet would drop them:")
            for line in plan.lost:
                print(f"  ! {line}")
            blocked.append(title)
            continue
        print(f"{title}: {plan.added} row(s) to add, {plan.changed} to change, {plan.removed} to remove")
        for line in plan.lines:
            print(line)
        for line in plan.residual:
            print(f"  ! {line}")
        requests += requests_for(plan)
        applied.append(key)
    if not args.apply:
        print("\nDry run: nothing was written. Add --apply to update the sheet.")
        return 4 if blocked else 0
    if requests:
        sheet.batch_update(requests)        # one batch: either every edit lands or none does
        print(f"[ok] Sheet updated ({len(requests)} edits)", file=sys.stderr)
    # GitHub's content is now in the sheet for these tabs, so GitHub's hash becomes the last synced point.
    save_state({t: digest(repo[t]) for t in applied}, "github->sheet")
    after = read_tabs(sheet)
    status = compare(sheet_data({t.title: t.rows for t in after.values()}), repo, load_state().get("tabs", {}))
    left = [du.CONFIG[t]["worksheet"] for t in applied if status[t] != IN_SYNC]
    if left:
        print(f"[note] {', '.join(left)}: the sheet now holds GitHub's content but writes it in a slightly "
              "different form (see the '!' lines above). The next publish from the sheet rewrites data/*.js "
              "that way.", file=sys.stderr)
    if args.badge:
        try_badge(sheet, status)
    return 4 if blocked else 0

def cmd_published(args) -> int:
    repo = load_repo()
    save_state({t: digest(canon(repo[t])) for t in du.CONFIG}, "sheet->github")
    return 0

def main():
    ap = argparse.ArgumentParser(description="Compare the Google Sheet with data/*.js and sync GitHub -> sheet.")
    ap.add_argument("--sheet", default=os.environ.get("SHEET_ID"), help="Google Sheet ID (or env SHEET_ID)")
    ap.add_argument("--sa", help="Path to service_account.json (optional if env set)")
    sub = ap.add_subparsers(dest="command", required=True)
    st = sub.add_parser("status", help="say which side changed, per tab")
    st.add_argument("--badge", action="store_true", help="write the result in the README tab")
    st.add_argument("--record", action="store_true", help="remember tabs that match as the last synced point")
    st.add_argument("--guard-publish", action="store_true", help="exit 3 if publishing would overwrite GitHub edits")
    pl = sub.add_parser("pull", help="make the sheet match data/*.js")
    pl.add_argument("--apply", action="store_true", help="write to the sheet (default is a dry run)")
    pl.add_argument("--force", action="store_true", help="also overwrite tabs that changed in the sheet")
    pl.add_argument("--badge", action="store_true", help="write the result in the README tab")
    sub.add_parser("published", help="record that data/*.js was just generated from the sheet")
    args = ap.parse_args()

    if args.command == "published":
        sys.exit(cmd_published(args))
    if not args.sheet:
        sys.exit("Set --sheet SHEET_ID or env SHEET_ID.")
    write = args.badge or getattr(args, "apply", False)
    sheet = LiveSheet(args.sheet, args.sa, write=write)
    sys.exit(cmd_status(sheet, args) if args.command == "status" else cmd_pull(sheet, args))

if __name__ == "__main__":
    main()
