#!/usr/bin/env python3
"""
structured_sqlite_to_schema.py

Reads structured.sqlite and writes:
 - schema.dbml
 - schema_inferred.dbml  (adds heuristic FK refs based on column name patterns)
 - schema.dot
 - tables_csv/ (one CSV per table)

Outputs are written next to the SQLite DB.
"""

import sqlite3
import csv
import os
import re
import sys
from pathlib import Path

# -------------------------------------------------
# Resolve paths safely (independent of CWD)
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "structured_data" / "structured.sqlite"
OUTPUT_DIR = DB_PATH.parent  # write outputs next to DB

# -------------------------------------------------
# SQLite helpers
# -------------------------------------------------
def get_tables(conn):
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    )
    return [r[0] for r in cur.fetchall()]

def get_columns(conn, table):
    cur = conn.execute(f"PRAGMA table_info('{table}')")
    return [
        {
            "cid": row[0],
            "name": row[1],
            "type": row[2] or "TEXT",
            "notnull": bool(row[3]),
            "dflt": row[4],
            "pk": bool(row[5]),
        }
        for row in cur.fetchall()
    ]

def get_foreign_keys(conn, table):
    cur = conn.execute(f"PRAGMA foreign_key_list('{table}')")
    return [
        {
            "id": r[0],
            "seq": r[1],
            "table": r[2],
            "from_col": r[3],
            "to_col": r[4],
        }
        for r in cur.fetchall()
    ]

# -------------------------------------------------
# Writers
# -------------------------------------------------
def write_dbml(tables_info, outpath, refs_extra=None):
    lines = []

    for tname, info in tables_info.items():
        lines.append(f"Table {tname} {{")
        for col in info["columns"]:
            flags = []
            if col["pk"]:
                flags.append("pk")
            if col["notnull"] and not col["pk"]:
                flags.append("not null")

            flagstr = f" [{', '.join(flags)}]" if flags else ""
            ctype = col["type"] or "text"
            lines.append(f"  {col['name']} {ctype}{flagstr}")
        lines.append("}\n")

    for tname, info in tables_info.items():
        for fk in info["fks"]:
            lines.append(
                f"Ref: {tname}.{fk['from_col']} > {fk['table']}.{fk['to_col']}"
            )

    if refs_extra:
        for ref in refs_extra:
            lines.append(
                f"Ref: {ref['from_table']}.{ref['from_col']} > "
                f"{ref['to_table']}.{ref['to_col']}"
            )

    outpath.write_text("\n".join(lines), encoding="utf-8")
    print(f"✔ DBML written → {outpath}")

def write_dot(tables_info, outpath):
    lines = [
        "digraph G {",
        "  graph [rankdir=LR];",
        "  node [shape=plaintext];",
        "",
    ]

    for tname, info in tables_info.items():
        tbl = [
            f'<table border="0" cellborder="1" cellspacing="0">'
            f'<tr><td bgcolor="lightgray" colspan="2"><b>{tname}</b></td></tr>'
        ]
        for col in info["columns"]:
            pk = "<b>PK</b> " if col["pk"] else ""
            tbl.append(
                f'<tr><td align="left">{pk}{col["name"]}</td>'
                f'<td align="left">{col["type"]}</td></tr>'
            )
        tbl.append("</table>")
        label = "".join(tbl).replace('"', '\\"')
        lines.append(f'  "{tname}" [label=<{label}>];')

    lines.append("")
    for tname, info in tables_info.items():
        for fk in info["fks"]:
            lines.append(
                f'  "{tname}" -> "{fk["table"]}" '
                f'[label="{fk["from_col"]}→{fk["to_col"]}"];'
            )

    lines.append("}")
    outpath.write_text("\n".join(lines), encoding="utf-8")
    print(f"✔ DOT written → {outpath}")

# -------------------------------------------------
# FK inference
# -------------------------------------------------
def infer_fks_by_name(tables_info):
    refs = []
    table_names = set(tables_info.keys())

    for tname, info in tables_info.items():
        for col in info["columns"]:
            cname = col["name"]
            m = re.match(r"^(.+?)_id$", cname, re.I) or re.match(r"^(.+?)id$", cname, re.I)
            if not m:
                continue

            base = m.group(1)
            candidates = [
                base,
                base.lower(),
                base.capitalize(),
                base + "s",
                base.rstrip("s"),
            ]

            target = next((c for c in candidates if c in table_names), None)
            if target:
                refs.append(
                    {
                        "from_table": tname,
                        "from_col": cname,
                        "to_table": target,
                        "to_col": "id",
                    }
                )

    uniq = {(r["from_table"], r["from_col"], r["to_table"], r["to_col"]): r for r in refs}
    print(f"✔ Inferred {len(uniq)} FK(s)")
    return list(uniq.values())

# -------------------------------------------------
# CSV export
# -------------------------------------------------
def export_csvs(conn, tables, outdir):
    outdir.mkdir(exist_ok=True)
    for t in tables:
        cur = conn.execute(f'SELECT * FROM "{t}"')
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()

        path = outdir / f"{t}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            for r in rows:
                writer.writerow([None if v is None else str(v) for v in r])

        print(f"✔ CSV written → {path}")

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    if not DB_PATH.exists():
        print(f"❌ Database not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    tables = get_tables(conn)
    tables_info = {
        t: {
            "columns": get_columns(conn, t),
            "fks": get_foreign_keys(conn, t),
        }
        for t in tables
    }

    write_dbml(tables_info, OUTPUT_DIR / "schema.dbml")
    write_dot(tables_info, OUTPUT_DIR / "schema.dot")

    export_csvs(conn, tables, OUTPUT_DIR / "tables_csv")

    inferred = infer_fks_by_name(tables_info)
    write_dbml(
        tables_info,
        OUTPUT_DIR / "schema_inferred.dbml",
        refs_extra=inferred,
    )

    conn.close()
    print("\n✅ Schema generation complete")

if __name__ == "__main__":
    main()
