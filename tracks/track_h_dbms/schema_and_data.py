"""
Synthetic TPC-H Benchmark Database Generator.
Generates schema and synthetic records for Customer, Orders, LineItem, and Part tables.
Student: Ayaan Mirza (Roll No: 241070043)
"""

import sqlite3
import numpy as np

def create_tpch_database(db_path: str = ":memory:", num_customers: int = 1000, num_orders: int = 3000, seed: int = 42) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Drop existing tables
    cur.executescript("""
    DROP TABLE IF EXISTS lineitem;
    DROP TABLE IF EXISTS orders;
    DROP TABLE IF EXISTS customer;
    DROP TABLE IF EXISTS part;

    CREATE TABLE customer (
        c_custkey INTEGER PRIMARY KEY,
        c_name TEXT NOT NULL,
        c_mktsegment TEXT NOT NULL,
        c_acctbal REAL NOT NULL
    );

    CREATE TABLE orders (
        o_orderkey INTEGER PRIMARY KEY,
        o_custkey INTEGER NOT NULL,
        o_orderdate TEXT NOT NULL,
        o_totalprice REAL NOT NULL,
        o_orderstatus TEXT NOT NULL,
        o_orderpriority TEXT NOT NULL
    );

    CREATE TABLE lineitem (
        l_orderkey INTEGER NOT NULL,
        l_partkey INTEGER NOT NULL,
        l_linenumber INTEGER NOT NULL,
        l_quantity REAL NOT NULL,
        l_extendedprice REAL NOT NULL,
        l_discount REAL NOT NULL,
        l_tax REAL NOT NULL,
        l_shipdate TEXT NOT NULL,
        l_returnflag TEXT NOT NULL
    );

    CREATE TABLE part (
        p_partkey INTEGER PRIMARY KEY,
        p_name TEXT NOT NULL,
        p_brand TEXT NOT NULL,
        p_type TEXT NOT NULL,
        p_size INTEGER NOT NULL,
        p_retailprice REAL NOT NULL
    );
    """)

    rng = np.random.default_rng(seed)
    mktsegments = ['BUILDING', 'AUTOMOBILE', 'MACHINERY', 'HOUSEHOLD', 'FURNITURE']
    statuses = ['O', 'F', 'P']
    priorities = ['1-URGENT', '2-HIGH', '3-MEDIUM', '4-LOW']
    flags = ['R', 'A', 'N']

    # Insert Customers
    cust_rows = []
    for cid in range(1, num_customers + 1):
        cust_rows.append((
            cid,
            f"Customer#{cid:06d}",
            rng.choice(mktsegments),
            float(round(rng.uniform(-500, 10000), 2))
        ))
    cur.executemany("INSERT INTO customer VALUES (?, ?, ?, ?)", cust_rows)

    # Insert Parts
    num_parts = 500
    part_rows = []
    for pid in range(1, num_parts + 1):
        part_rows.append((
            pid,
            f"Part#{pid:05d}",
            f"Brand#{rng.integers(1, 10)}",
            "PROMO_POLISHED",
            int(rng.integers(1, 50)),
            float(round(rng.uniform(10, 500), 2))
        ))
    cur.executemany("INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)", part_rows)

    # Insert Orders & LineItems
    order_rows = []
    lineitem_rows = []

    for oid in range(1, num_orders + 1):
        cid = int(rng.integers(1, num_customers + 1))
        # Random date between 2024-01-01 and 2026-12-31
        days_offset = int(rng.integers(0, 1000))
        date_str = f"2024-{1 + (days_offset % 12):02d}-{1 + (days_offset % 28):02d}"
        tot_price = float(round(rng.uniform(100, 5000), 2))
        order_rows.append((
            oid, cid, date_str, tot_price,
            rng.choice(statuses), rng.choice(priorities)
        ))

        # 2 to 4 lineitems per order
        n_items = rng.integers(2, 5)
        for line_num in range(1, n_items + 1):
            pid = int(rng.integers(1, num_parts + 1))
            qty = float(rng.integers(1, 50))
            price = float(round(qty * rng.uniform(20, 100), 2))
            disc = float(round(rng.uniform(0.01, 0.10), 2))
            tax = 0.05
            ship_offset = days_offset + int(rng.integers(5, 30))
            ship_date = f"2024-{1 + (ship_offset % 12):02d}-{1 + (ship_offset % 28):02d}"
            lineitem_rows.append((
                oid, pid, line_num, qty, price, disc, tax,
                ship_date, rng.choice(flags)
            ))

    cur.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", order_rows)
    cur.executemany("INSERT INTO lineitem VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", lineitem_rows)

    conn.commit()
    return conn
