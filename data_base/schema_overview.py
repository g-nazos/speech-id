import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg import sql
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DEFAULT_OUTPUT_FILE = ROOT_DIR / "results" / "database_schema_overview.md"


@dataclass(slots=True)
class TableOverview:
    name: str
    row_count: int
    columns: list[dict[str, str]]
    constraints: list[dict[str, str]]
    indexes: list[dict[str, str]]


def load_environment() -> dict[str, str]:
    dotenv_path = BASE_DIR / ".env"
    if not dotenv_path.exists():
        raise FileNotFoundError(
            f"Could not find .env file at {dotenv_path}. Make sure the script is run from data_base/."
        )

    load_dotenv(dotenv_path)

    env = {
        "DB_USER": os.getenv("DB_USER"),
        "DB_PASSWORD": os.getenv("DB_PASSWORD"),
        "DB_NAME": os.getenv("DB_NAME"),
        "DB_HOST": os.getenv("DB_HOST"),
        "DB_PORT": os.getenv("DB_PORT"),
    }

    missing = [key for key, value in env.items() if not value]
    if missing:
        raise EnvironmentError(
            f"Missing required DB environment variables in {dotenv_path}: {', '.join(missing)}"
        )

    return {key: value for key, value in env.items() if value is not None}


def build_connection_string(env: dict[str, str]) -> str:
    return (
        f"host={env['DB_HOST']} port={env['DB_PORT']} dbname={env['DB_NAME']} "
        f"user={env['DB_USER']} password={env['DB_PASSWORD']}"
    )


def list_tables(conn: psycopg.Connection, schema: str) -> list[str]:
    query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = %s AND table_type = 'BASE TABLE'
        ORDER BY table_name;
    """
    with conn.cursor() as cur:
        cur.execute(query, (schema,))
        return [row[0] for row in cur.fetchall()]


def fetch_columns(conn: psycopg.Connection, schema: str, table: str) -> list[dict[str, str]]:
    query = """
        SELECT
            a.attname AS column_name,
            pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type,
            CASE WHEN a.attnotnull THEN 'NO' ELSE 'YES' END AS is_nullable,
            COALESCE(pg_get_expr(ad.adbin, ad.adrelid), '') AS column_default
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        LEFT JOIN pg_attrdef ad ON ad.adrelid = a.attrelid AND ad.adnum = a.attnum
        WHERE n.nspname = %s
          AND c.relname = %s
          AND a.attnum > 0
          AND NOT a.attisdropped
        ORDER BY a.attnum;
    """
    with conn.cursor() as cur:
        cur.execute(query, (schema, table))
        return [
            {
                "column_name": row[0],
                "data_type": row[1],
                "is_nullable": row[2],
                "column_default": row[3],
            }
            for row in cur.fetchall()
        ]


def fetch_constraints(conn: psycopg.Connection, schema: str, table: str) -> list[dict[str, str]]:
    query = """
        SELECT
            con.conname AS constraint_name,
            con.contype AS constraint_type,
            pg_get_constraintdef(con.oid, true) AS definition
        FROM pg_constraint con
        WHERE con.conrelid = to_regclass(%s)
        ORDER BY con.contype, con.conname;
    """
    with conn.cursor() as cur:
        cur.execute(query, (f"{schema}.{table}",))
        return [
            {
                "constraint_name": row[0],
                "constraint_type": row[1],
                "definition": row[2],
            }
            for row in cur.fetchall()
        ]


def fetch_indexes(conn: psycopg.Connection, schema: str, table: str) -> list[dict[str, str]]:
    query = """
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE schemaname = %s AND tablename = %s
        ORDER BY indexname;
    """
    with conn.cursor() as cur:
        cur.execute(query, (schema, table))
        return [
            {"index_name": row[0], "definition": row[1]}
            for row in cur.fetchall()
        ]


def fetch_row_count(conn: psycopg.Connection, schema: str, table: str) -> int:
    query = sql.SQL("SELECT COUNT(*) FROM {};").format(sql.Identifier(schema, table))
    with conn.cursor() as cur:
        cur.execute(query)
        return int(cur.fetchone()[0])


def fetch_database_summary(conn: psycopg.Connection, schema: str, table_count: int, total_rows: int) -> dict[str, str]:
    query = """
        SELECT
            current_database(),
            current_user,
            COALESCE(inet_server_addr()::text, 'local socket'),
            inet_server_port()::text,
            pg_size_pretty(pg_database_size(current_database()))
    """
    with conn.cursor() as cur:
        cur.execute(query)
        database_name, current_user, server_addr, server_port, database_size = cur.fetchone()

    return {
        "database_name": str(database_name),
        "current_user": str(current_user),
        "server_addr": str(server_addr),
        "server_port": str(server_port),
        "database_size": str(database_size),
        "schema": schema,
        "table_count": str(table_count),
        "total_rows": str(total_rows),
    }


def escape_markdown_cell(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def render_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(escape_markdown_cell(cell) for cell in row) + " |")
    return lines


def render_report(summary: dict[str, str], tables: list[TableOverview]) -> str:
    lines: list[str] = []
    lines.append("# Database Schema Overview")
    lines.append("")
    lines.append("Generated from the live PostgreSQL database configured in `data_base/.env`.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Database: {summary['database_name']}")
    lines.append(f"- User: {summary['current_user']}")
    lines.append(f"- Server: {summary['server_addr']}:{summary['server_port']}")
    lines.append(f"- Schema: {summary['schema']}")
    lines.append(f"- Tables: {summary['table_count']}")
    lines.append(f"- Total rows: {summary['total_rows']}")
    lines.append(f"- Database size: {summary['database_size']}")
    lines.append("")

    lines.append("## Tables")
    lines.append("")
    for table in tables:
        lines.append(f"### {table.name}")
        lines.append(f"Rows: {table.row_count}")
        lines.append("")

        lines.append("Columns")
        lines.extend(render_table(
            ["Name", "Type", "Nullable", "Default"],
            [[col["column_name"], col["data_type"], col["is_nullable"], col["column_default"]] for col in table.columns],
        ))
        lines.append("")

        lines.append("Constraints")
        if table.constraints:
            lines.extend(render_table(
                ["Name", "Type", "Definition"],
                [[cons["constraint_name"], cons["constraint_type"], cons["definition"]] for cons in table.constraints],
            ))
        else:
            lines.append("None")
        lines.append("")

        lines.append("Indexes")
        if table.indexes:
            lines.extend(render_table(
                ["Name", "Definition"],
                [[idx["index_name"], idx["definition"]] for idx in table.indexes],
            ))
        else:
            lines.append("None")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_overview(conn: psycopg.Connection, schema: str) -> tuple[dict[str, str], list[TableOverview]]:
    table_names = list_tables(conn, schema)
    table_overviews: list[TableOverview] = []
    total_rows = 0

    for table_name in table_names:
        row_count = fetch_row_count(conn, schema, table_name)
        total_rows += row_count
        table_overviews.append(
            TableOverview(
                name=table_name,
                row_count=row_count,
                columns=fetch_columns(conn, schema, table_name),
                constraints=fetch_constraints(conn, schema, table_name),
                indexes=fetch_indexes(conn, schema, table_name),
            )
        )

    summary = fetch_database_summary(conn, schema, len(table_overviews), total_rows)
    return summary, table_overviews


def write_report(output_file: Path, report: str) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown overview of the live PostgreSQL schema and table contents."
    )
    parser.add_argument(
        "--schema",
        default="public",
        help="Database schema to inspect. Defaults to public.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help=f"Optional path to save the Markdown report. Example: {DEFAULT_OUTPUT_FILE}",
    )
    args = parser.parse_args()

    env = load_environment()
    conn_str = build_connection_string(env)

    with psycopg.connect(conn_str) as conn:
        summary, tables = build_overview(conn, args.schema)

    report = render_report(summary, tables)
    print(report, end="")

    if args.output_file is not None:
        write_report(args.output_file, report)
        print(f"\nSaved Markdown report to {args.output_file}")


if __name__ == "__main__":
    main()