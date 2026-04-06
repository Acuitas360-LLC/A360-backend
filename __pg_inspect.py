import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg

load_dotenv(Path('.env'))

db_uri = (
    os.getenv('DB_URI', '').strip()
    or os.getenv('POSTGRES_URL', '').strip()
    or os.getenv('POSTGRES_URI', '').strip()
)
print('DB_URI configured:', bool(db_uri))

conn = psycopg.connect(db_uri)
cur = conn.cursor()

cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name")
tables = [r[0] for r in cur.fetchall()]
print('TABLES:', tables)

targets = ['thread_registry', 'hidden_threads', 'thread_message_cache', 'message_feedback']
for t in targets:
    if t in tables:
        cur.execute(f'SELECT COUNT(*) FROM {t}')
        print(f'{t} count:', cur.fetchone()[0])

print('\nLatest thread_registry rows:')
if 'thread_registry' in tables:
    cur.execute("SELECT user_id, thread_id, created_at, title FROM thread_registry ORDER BY created_at DESC NULLS LAST LIMIT 10")
    for row in cur.fetchall():
        print(row)

print('\nLatest hidden_threads rows:')
if 'hidden_threads' in tables:
    cur.execute("SELECT user_id, thread_id, hidden_at FROM hidden_threads ORDER BY hidden_at DESC NULLS LAST LIMIT 10")
    for row in cur.fetchall():
        print(row)

print('\nLatest thread_message_cache rows:')
if 'thread_message_cache' in tables:
    cur.execute("SELECT user_id, thread_id, updated_at, LENGTH(messages_json) FROM thread_message_cache ORDER BY updated_at DESC NULLS LAST LIMIT 10")
    for row in cur.fetchall():
        print(row)

print('\nLatest message_feedback rows:')
if 'message_feedback' in tables:
    cur.execute("SELECT user_id, thread_id, message_id, rating, created_at FROM message_feedback ORDER BY id DESC LIMIT 10")
    for row in cur.fetchall():
        print(row)

conn.close()
