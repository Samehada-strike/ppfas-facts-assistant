import json
import sqlite3
from pathlib import Path

# ========== CONFIG ==========
PROJECT_ROOT = Path(r"E:\ML Projects\RAG_Chatbot_PP_MF")

STRUCTURED_JSON_PATH = PROJECT_ROOT / "project_files" / "data" / "json_files" / "parag_parikh_structured_db.json"
SQLITE_PATH = PROJECT_ROOT / "project_files" / "data" /"structured_data" / "structured.sqlite"

SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)

# ========== LOAD JSON ==========
def load_structured_json():
    if not STRUCTURED_JSON_PATH.exists():
        raise FileNotFoundError(f"JSON file not found: {STRUCTURED_JSON_PATH}")
    with STRUCTURED_JSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# ========== BUILD SQLITE DB ==========
def create_tables(conn: sqlite3.Connection):
    cur = conn.cursor()

    # Drop existing tables (rebuild)
    cur.executescript(
        """
        PRAGMA foreign_keys = OFF;
        DROP TABLE IF EXISTS return_values;
        DROP TABLE IF EXISTS return_periods;
        DROP TABLE IF EXISTS returns_meta;
        DROP TABLE IF EXISTS expense_exit_tax;
        DROP TABLE IF EXISTS fund_managers;
        DROP TABLE IF EXISTS holdings;
        DROP TABLE IF EXISTS returns_summary;
        DROP TABLE IF EXISTS advanced_ratios;
        DROP TABLE IF EXISTS minimum_investment;
        DROP TABLE IF EXISTS nav_summary;
        DROP TABLE IF EXISTS schemes;
        PRAGMA foreign_keys = ON;
        """
    )

    # schemes table (scheme_id primary key)
    cur.execute(
        """
        CREATE TABLE schemes (
            scheme_id TEXT PRIMARY KEY,
            scheme_name TEXT,
            category TEXT,
            sub_category TEXT,
            risk TEXT,
            amc_name TEXT
        );
        """
    )

    # nav_summary - PRIMARY KEY on scheme_id (one row per scheme)
    cur.execute(
        """
        CREATE TABLE nav_summary (
            scheme_id TEXT PRIMARY KEY,
            nav_value_raw TEXT,
            nav_as_on TEXT,
            FOREIGN KEY (scheme_id) REFERENCES schemes (scheme_id)
        );
        """
    )

    # minimum_investment - primary key scheme_id
    cur.execute(
        """
        CREATE TABLE minimum_investment (
            scheme_id TEXT PRIMARY KEY,
            min_first_investment TEXT,
            min_second_investment TEXT,
            min_sip TEXT,
            FOREIGN KEY (scheme_id) REFERENCES schemes (scheme_id)
        );
        """
    )

    # advanced_ratios - primary key scheme_id
    cur.execute(
        """
        CREATE TABLE advanced_ratios (
            scheme_id TEXT PRIMARY KEY,
            sharpe_ratio REAL,
            sortino_ratio REAL,
            alpha REAL,
            beta REAL,
            standard_deviation REAL,
            information_ratio REAL,
            risk_rating INTEGER,
            risk TEXT,
            FOREIGN KEY (scheme_id) REFERENCES schemes (scheme_id)
        );
        """
    )

    # expense_exit_tax - primary key scheme_id
    cur.execute(
        """
        CREATE TABLE expense_exit_tax (
            scheme_id TEXT PRIMARY KEY,
            expense_ratio TEXT,
            exit_load TEXT,
            stamp_duty TEXT,
            tax_implication TEXT,
            FOREIGN KEY (scheme_id) REFERENCES schemes (scheme_id)
        );
        """
    )

    # returns_meta - one row per scheme (meta info)
    cur.execute(
        """
        CREATE TABLE returns_meta (
            scheme_id TEXT PRIMARY KEY,
            category TEXT,
            active_type TEXT,
            FOREIGN KEY (scheme_id) REFERENCES schemes (scheme_id)
        );
        """
    )

    # return_periods - one row per period per scheme (order preserved by ordinal)
    cur.execute(
        """
        CREATE TABLE return_periods (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scheme_id TEXT,
            period_label TEXT,
            ord INTEGER,
            FOREIGN KEY (scheme_id) REFERENCES schemes (scheme_id)
        );
        """
    )

    # return_values - one value per scheme x metric_label x period_label
    cur.execute(
        """
        CREATE TABLE return_values (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scheme_id TEXT,
            metric_label TEXT,
            period_label TEXT,
            value TEXT,
            FOREIGN KEY (scheme_id) REFERENCES schemes (scheme_id)
        );
        """
    )

    # holdings (many rows per scheme)
    cur.execute(
        """
        CREATE TABLE holdings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scheme_id TEXT,
            name TEXT,
            sector TEXT,
            instrument TEXT,
            assets TEXT,
            FOREIGN KEY (scheme_id) REFERENCES schemes (scheme_id)
        );
        """
    )

    # fund_managers (many rows per scheme)
    cur.execute(
        """
        CREATE TABLE fund_managers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scheme_id TEXT,
            manager_name TEXT,
            tenure TEXT,
            education TEXT,
            experience TEXT,
            FOREIGN KEY (scheme_id) REFERENCES schemes (scheme_id)
        );
        """
    )

    conn.commit()


def populate_tables(conn: sqlite3.Connection, data: list[dict]):
    cur = conn.cursor()

    for row in data:
        scheme_id = row.get("scheme_id")
        overview = row.get("overview") or {}
        nav = overview.get("nav") or {}
        min_inv = row.get("minimum_investment") or {}
        adv = row.get("advanced_ratios") or {}
        rr = row.get("returns_and_rankings") or {}
        holdings = (row.get("holdings") or {}).get("items") or []
        fm = (row.get("fund_management") or {}).get("managers") or []
        eet = row.get("expense_exit_tax") or {}

        # Insert into schemes
        cur.execute(
            """
            INSERT OR REPLACE INTO schemes (scheme_id, scheme_name, category, sub_category, risk, amc_name)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                scheme_id,
                overview.get("scheme_name"),
                overview.get("category"),
                overview.get("sub_category"),
                overview.get("risk"),
                "PPFAS Mutual Fund",
            ),
        )

        # nav_summary
        cur.execute(
            """
            INSERT OR REPLACE INTO nav_summary (scheme_id, nav_value_raw, nav_as_on)
            VALUES (?, ?, ?)
            """,
            (
                scheme_id,
                nav.get("value_raw"),
                nav.get("as_on"),
            ),
        )

        # minimum_investment
        cur.execute(
            """
            INSERT OR REPLACE INTO minimum_investment (
                scheme_id, min_first_investment, min_second_investment, min_sip
            ) VALUES (?, ?, ?, ?)
            """,
            (
                scheme_id,
                min_inv.get("min_first_investment"),
                min_inv.get("min_second_investment"),
                min_inv.get("min_sip"),
            ),
        )

        # advanced_ratios
        cur.execute(
            """
            INSERT OR REPLACE INTO advanced_ratios (
                scheme_id, sharpe_ratio, sortino_ratio, alpha, beta,
                standard_deviation, information_ratio, risk_rating, risk
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scheme_id,
                _safe_float(adv.get("sharpe_ratio")),
                _safe_float(adv.get("sortino_ratio")),
                _safe_float(adv.get("alpha")),
                _safe_float(adv.get("beta")),
                _safe_float(adv.get("standard_deviation")),
                _safe_float(adv.get("information_ratio")),
                _safe_int(adv.get("risk_rating")),
                adv.get("risk"),
            ),
        )

        # expense_exit_tax
        cur.execute(
            """
            INSERT OR REPLACE INTO expense_exit_tax (
                scheme_id, expense_ratio, exit_load, stamp_duty, tax_implication
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                scheme_id,
                eet.get("expense_ratio"),
                eet.get("exit_load"),
                eet.get("stamp_duty"),
                eet.get("tax_implication"),
            ),
        )

        # returns: rr may follow new schema. If present, populate meta + periods + values
        if rr:
            # returns_meta
            cur.execute(
                """
                INSERT OR REPLACE INTO returns_meta (scheme_id, category, active_type)
                VALUES (?, ?, ?)
                """,
                (
                    scheme_id,
                    rr.get("category"),
                    rr.get("active_type") or (rr.get("available_types")[0] if rr.get("available_types") else None),
                ),
            )

            # Clear any existing periods/values for this scheme to avoid duplicates on rebuild
            cur.execute("DELETE FROM return_periods WHERE scheme_id = ?", (scheme_id,))
            cur.execute("DELETE FROM return_values WHERE scheme_id = ?", (scheme_id,))

            table = rr.get("table") or {}
            columns = table.get("columns") or []
            rows_list = table.get("rows") or []

            # Insert periods preserving order; columns are period labels
            for idx, col in enumerate(columns):
                cur.execute(
                    """
                    INSERT INTO return_periods (scheme_id, period_label, ord)
                    VALUES (?, ?, ?)
                    """,
                    (scheme_id, col, idx),
                )

            # Insert values: for each metric row and for each period, insert a value row
            for r in rows_list:
                label = r.get("label")
                values = r.get("values") or []
                # values align to columns by index
                for j, val in enumerate(values):
                    period_label = columns[j] if j < len(columns) else None
                    cur.execute(
                        """
                        INSERT INTO return_values (scheme_id, metric_label, period_label, value)
                        VALUES (?, ?, ?, ?)
                        """,
                        (scheme_id, label, period_label, val),
                    )

        # holdings
        for h in holdings:
            cur.execute(
                """
                INSERT INTO holdings (scheme_id, name, sector, instrument, assets)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    scheme_id,
                    h.get("name"),
                    h.get("sector"),
                    h.get("instrument"),
                    h.get("assets"),
                ),
            )

        # fund_managers
        for m in fm:
            cur.execute(
                """
                INSERT INTO fund_managers (scheme_id, manager_name, tenure, education, experience)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    scheme_id,
                    m.get("name"),
                    m.get("tenure"),
                    m.get("education"),
                    m.get("experience"),
                ),
            )

    conn.commit()


def _safe_float(v):
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _safe_int(v):
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def build_sqlite_db(data: list[dict]):
    print(f"Creating SQLite DB at: {SQLITE_PATH}")
    conn = sqlite3.connect(SQLITE_PATH)
    try:
        create_tables(conn)
        populate_tables(conn, data)
    finally:
        conn.close()
    print("✅ SQLite DB created and populated.")


# ========== MAIN ==========
def main():
    print("Loading structured JSON...")
    data = load_structured_json()
    print(f"Loaded {len(data)} scheme records.")

    build_sqlite_db(data)

    print("\nAll done.")


if __name__ == "__main__":
    main()