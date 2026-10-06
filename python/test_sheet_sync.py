#!/usr/bin/env python3
"""
Offline checks for sheet_sync.py. No network: the Google Sheet is replaced by an
in-memory grid that understands the handful of Sheets API requests sheet_sync sends.

  python test_sheet_sync.py
"""

import copy, datetime, sys

import data_update as du
import sheet_sync as ss

HEADERS = {
    "publications": ["id", "type", "title", "authors", "venue", "year", "award", "link",
                     "abstract", "code", "demo", "bibtex", "streams"],
    "news": ["date", "type", "title", "tags", "content"],
    "projects": ["name", "title", "status", "pi", "funding", "period", "team_members", "collaborators",
                 "keywords", "website", "figure", "publications", "description"],
    "team": ["role", "name", "image", "position", "previous_position", "advisor", "co_advisor", "start_year",
             "graduation_year", "research_interests", "country", "city", "website", "github", "linkedin", "scholar"],
    "streams": ["keyword", "name", "color", "icon", "description"],
    "photos": ["filename", "year", "description"],
}
ID_FORMULA = "=COUNTA(C:C) - ROW() + 1"

class FakeSheet:
    """Grid per tab. Row 0 is the header; a cell is {"v": value, "f": formula, "date": bool}."""
    def __init__(self, data):
        self.grids, self.writes = {}, []
        for gid, key in enumerate(du.CONFIG):
            items = ss.flatten(key, data[key])
            if key == "publications":
                items = sorted(items, key=lambda o: -o["id"])
            rows = [[{"v": h, "f": None, "date": False} for h in HEADERS[key]]]
            for o in items:
                cells = ss.cells_for(key, o)
                rows.append([self._seed(key, h, cells[h]) for h in HEADERS[key]])
            self.grids[gid] = rows
        self.gid = {key: gid for gid, key in enumerate(du.CONFIG)}

    @staticmethod
    def _seed(key, header, value):
        if key == "publications" and header == "id":
            return {"v": None, "f": ID_FORMULA, "date": False}
        cell = ss._cell(value)
        entered = cell.get("userEnteredValue", {})
        return {"v": next(iter(entered.values()), None), "f": None, "date": "userEnteredFormat" in cell}

    def _shown(self, key, rows, r, c):
        cell = rows[r][c]
        if cell["f"]:
            titles = sum(1 for row in rows if row[HEADERS[key].index("title")]["v"] not in (None, ""))
            return titles - (r + 1) + 1          # COUNTA(title column) - ROW() + 1
        if cell["v"] in (None, ""):
            return ""
        if cell["date"] and isinstance(cell["v"], (int, float)):
            return (ss.EPOCH + datetime.timedelta(days=cell["v"])).strftime("%d/%m/%Y")
        return ss.numericise(cell["v"])

    def read_tab(self, key):
        rows = self.grids[self.gid[key]]
        records = [{h: self._shown(key, rows, r, c) for c, h in enumerate(HEADERS[key])} for r in range(1, len(rows))]
        formulas = {h: cell["f"] for h, cell in zip(HEADERS[key], rows[1])} if len(rows) > 1 else {}
        return ss.Tab(key, du.CONFIG[key]["worksheet"], self.gid[key], list(HEADERS[key]), records,
                      {h: f for h, f in formulas.items() if f})

    def read_tabs(self):
        return {key: self.read_tab(key) for key in du.CONFIG}

    def batch_update(self, requests):
        for req in requests:
            (kind, body), = req.items()
            if kind == "updateCells":
                rng, rows = body["range"], self.grids[body["range"]["sheetId"]]
                assert rng["endRowIndex"] == rng["startRowIndex"] + 1 and rng["startRowIndex"] >= 1
                for k, data in enumerate(body["rows"][0]["values"]):
                    cell = rows[rng["startRowIndex"]][rng["startColumnIndex"] + k]
                    if "userEnteredValue" in body["fields"]:
                        entered = data.get("userEnteredValue", {})
                        cell["f"] = entered.get("formulaValue")
                        cell["v"] = None if cell["f"] else next(iter(entered.values()), None)
                    if "numberFormat" in body["fields"]:
                        cell["date"] = data.get("userEnteredFormat", {}).get("numberFormat", {}).get("type") == "DATE"
                    self.writes.append((rng["sheetId"], rng["startRowIndex"], rng["startColumnIndex"] + k))
                continue
            rng = body.get("range") or body.get("source")
            rows, start = self.grids[rng["sheetId"]], rng["startIndex"]
            assert rng["dimension"] == "ROWS" and rng["endIndex"] == start + 1 and start >= 1
            if kind == "insertDimension":
                like = rows[start - 1] if body["inheritFromBefore"] else (rows[start] if start < len(rows) else rows[-1])
                assert not (body["inheritFromBefore"] and start == 1), "would inherit the header's formatting"
                rows.insert(start, [{"v": None, "f": None, "date": c["date"]} for c in like])
            elif kind == "deleteDimension":
                del rows[start]
            elif kind == "moveDimension":
                dest = body["destinationIndex"]
                rows.insert(dest if dest < start else dest - 1, rows.pop(start))
            else:
                raise AssertionError(kind)

def produced(sheet):
    tabs = ss.read_tabs(sheet)
    return ss.sheet_data({t.title: t.rows for t in tabs.values()})

def pull(sheet, want):
    plans = {key: ss.plan_tab(tab, want[key]) for key, tab in ss.read_tabs(sheet).items()}
    sheet.batch_update([r for p in plans.values() for r in ss.requests_for(p)])
    return plans

def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{' — ' + detail if detail and not ok else ''}")
    return ok

def main():
    repo = ss.canon(ss.load_repo())
    results = []

    # 1. A sheet built from data/*.js must regenerate data/*.js exactly.
    sheet = FakeSheet(repo)
    got = produced(sheet)
    bad = [k for k in du.CONFIG if got[k] != repo[k]]
    results.append(check("sheet built from data/*.js regenerates the same data", not bad, str(bad)))

    # 2. Nothing to do when both sides already match.
    plans = {key: ss.plan_tab(tab, repo[key]) for key, tab in ss.read_tabs(sheet).items()}
    results.append(check("no edits planned when in sync", all(not p.ops and p.exact for p in plans.values())))

    # 3. Edits made in code land in the sheet, touching only what changed.
    want = copy.deepcopy(repo)
    pubs = want["publications"]
    newest = max(pubs, key=lambda o: o["id"])
    pubs.remove(newest)                                                 # removed in code
    pubs.insert(0, {"id": newest["id"], "title": "A Brand New Paper", "authors": "A. Sardine, B. Sardine",
                    "venue": "NeurIPS", "year": 2026, "type": "conference",
                    "abstract": "<p>New abstract with <em>markup</em>.</p>", "streams": ["attention", "theory"],
                    "links": {"paper": "https://arxiv.org/abs/0000.00000",
                              "bibtex": "@inproceedings{x,\n  title={A Brand New Paper}\n}"}})
    pubs[3]["venue"], pubs[3]["type"] = "ICML", "conference"            # edited in code
    pubs[5]["title"] = pubs[5]["title"] + " (Extended Version)"         # renamed in code
    news = want["news"]
    news.insert(1, {"date": news[0]["date"], "type": "event", "title": "Second item on the same day",
                    "content": "<p>Hello from code.</p>", "tags": ["test"]})
    news[4]["content"] = "<p>Rewritten in code.</p>"
    del news[7]
    team = want["team"]
    role = next(iter(team))
    team[role].append({"name": "Zz New Member", "role": team[role][0]["role"], "position": "Visiting Researcher",
                       "start_year": 2026, "graduation_year": "current", "country": "Portugal", "city": "Lisbon"})
    team[role][0]["position"] = "Somewhere Else"
    del team[role][1]
    team[role].sort(key=lambda m: m.get("name", ""))
    want["projects"]["current"][0]["description"] = "<p>Updated description.</p>"
    want["projects"]["current"].append({"name": "NEWPROJ", "title": "A New Project", "status": "current",
                                        "pi": "André Martins", "period": "2027-2029",
                                        "description": "<p>Fresh.</p>", "keywords": ["one", "two"]})
    want["streams"]["new-stream"] = {"keyword": "new-stream", "name": "New Stream", "color": "#000000",
                                     "icon": "fas fa-star", "description": "A stream added in code."}
    want["photos"].append({"year": "1999", "description": "An old photo", "filename": "group0.jpg"})
    del want["photos"][0]

    sheet = FakeSheet(repo)
    cells_before = sum(len(r) for g in sheet.grids.values() for r in g)
    plans = pull(sheet, want)
    got = produced(sheet)
    bad = [k for k in du.CONFIG if got[k] != want[k]]
    results.append(check("code edits (add, change, rename, remove) reach the sheet", not bad, str(bad)))
    results.append(check("every plan predicted its own outcome", all(p.exact for p in plans.values()),
                         str({k: p.residual for k, p in plans.items() if not p.exact})))
    results.append(check("only changed cells and new rows were written", len(sheet.writes) < cells_before / 20,
                         f"{len(sheet.writes)} cell writes for {cells_before} cells"))
    results.append(check("no rows were moved", not any(p.moved for p in plans.values())))

    # 4. Markdown typed in the sheet survives a pull that changes another cell of the same row.
    sheet = FakeSheet(repo)
    grid = sheet.grids[sheet.gid["news"]]
    grid[2][HEADERS["news"].index("content")]["v"] = "Typed **in Markdown** by a human."
    want = ss.canon(produced(sheet))
    want["news"][1]["title"] = want["news"][1]["title"] + " (edited in code)"
    pull(sheet, want)
    kept = grid[2][HEADERS["news"].index("content")]["v"] == "Typed **in Markdown** by a human."
    results.append(check("Markdown cells are left alone when their HTML is unchanged", kept and produced(sheet) == want))

    # 5. A different order on GitHub is reproduced by moving rows.
    sheet = FakeSheet(repo)
    want = copy.deepcopy(repo)
    current = want["projects"]["current"]
    current[0], current[-1] = current[-1], current[0]
    plans = pull(sheet, want)
    results.append(check("row order follows GitHub when it matters",
                         produced(sheet)["projects"] == want["projects"] and plans["projects"].moved > 0))

    # 6. What the sheet cannot hold is reported instead of being lost on the next publish.
    sheet = FakeSheet(repo)
    want = copy.deepcopy(repo)
    want["news"][0]["featured"] = True
    plan = ss.plan_tab(sheet.read_tab("news"), want["news"])
    results.append(check("fields without a sheet column are reported", bool(plan.lost) and "featured" in plan.lost[0]))
    want = copy.deepcopy(repo)
    want["news"][0]["content"] = "Plain text typed in code, no tags."
    plan = ss.plan_tab(sheet.read_tab("news"), want["news"])
    results.append(check("differences in form only are flagged as such",
                         not plan.lost and not plan.exact and bool(plan.residual), str(plan.residual)))

    # 7. Which side changed.
    base = {k: ss.digest(repo[k]) for k in du.CONFIG}
    edited = copy.deepcopy(repo)
    edited["news"][0]["title"] += "!"
    other = copy.deepcopy(repo)
    other["news"][0]["title"] += "?"
    results.append(check("status tells the sides apart", (
        ss.compare(repo, repo, base)["news"] == ss.IN_SYNC and
        ss.compare(edited, repo, base)["news"] == ss.SHEET_AHEAD and
        ss.compare(repo, edited, base)["news"] == ss.GITHUB_AHEAD and
        ss.compare(edited, other, base)["news"] == ss.DIVERGED and
        ss.compare(edited, repo, {})["news"] == ss.UNKNOWN and
        ss.compare(edited, edited, {})["news"] == ss.IN_SYNC)))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)

if __name__ == "__main__":
    main()
