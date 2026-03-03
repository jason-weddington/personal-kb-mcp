# Team Setup on AWS (Aurora + IAM Auth)

This guide walks through setting up a shared personal-kb on Aurora Serverless v2 with IAM database authentication. No database passwords — each team member authenticates with their AWS credentials and gets short-lived tokens.

## Prerequisites

- An AWS account with permissions to create Aurora clusters, IAM roles, and policies
- AWS CLI configured (`aws configure` or environment variables)
- Python 3.13+ and [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally (for embeddings)

## 1. Create the Aurora cluster

```bash
# Create a subnet group (use your VPC subnets)
aws rds create-db-subnet-group \
  --db-subnet-group-name kb-subnets \
  --db-subnet-group-description "Subnets for personal-kb" \
  --subnet-ids subnet-xxxx subnet-yyyy

# Create the Aurora Serverless v2 cluster
aws rds create-db-cluster \
  --db-cluster-identifier personal-kb \
  --engine aurora-postgresql \
  --engine-version 16.4 \
  --serverless-v2-scaling-configuration MinCapacity=0.5,MaxCapacity=4 \
  --master-username kbadmin \
  --master-user-password "TEMPORARY_PASSWORD" \
  --enable-iam-database-authentication \
  --db-subnet-group-name kb-subnets \
  --vpc-security-group-ids sg-xxxx \
  --storage-encrypted

# Create the instance
aws rds create-db-instance \
  --db-instance-identifier personal-kb-1 \
  --db-cluster-identifier personal-kb \
  --db-instance-class db.serverless \
  --engine aurora-postgresql
```

Wait for the cluster to become available:

```bash
aws rds wait db-cluster-available --db-cluster-identifier personal-kb
```

Note the cluster endpoint:

```bash
aws rds describe-db-clusters \
  --db-cluster-identifier personal-kb \
  --query 'DBClusters[0].Endpoint' --output text
```

## 2. Install pgvector

Connect to the cluster as the master user and install the pgvector extension:

```bash
psql "postgresql://kbadmin:TEMPORARY_PASSWORD@<cluster-endpoint>:5432/postgres"
```

```sql
CREATE DATABASE personal_kb;
\c personal_kb
CREATE EXTENSION IF NOT EXISTS vector;
```

## 3. Create an IAM database user

Still connected as the master user:

```sql
-- Create the role that IAM users will authenticate as
CREATE USER kb_iam WITH LOGIN;

-- Grant IAM authentication
GRANT rds_iam TO kb_iam;

-- Grant permissions on the database
GRANT ALL PRIVILEGES ON DATABASE personal_kb TO kb_iam;

-- Connect to personal_kb and grant schema permissions
\c personal_kb
GRANT ALL ON SCHEMA public TO kb_iam;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO kb_iam;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO kb_iam;
```

## 4. Create an IAM policy for database access

```bash
# Get the cluster resource ID
CLUSTER_RESOURCE_ID=$(aws rds describe-db-clusters \
  --db-cluster-identifier personal-kb \
  --query 'DBClusters[0].DbClusterResourceId' --output text)

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1

# Create the IAM policy
cat > /tmp/kb-db-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "rds-db:connect",
      "Resource": "arn:aws:rds-db:${REGION}:${ACCOUNT_ID}:dbuser:${CLUSTER_RESOURCE_ID}/kb_iam"
    }
  ]
}
EOF

aws iam create-policy \
  --policy-name PersonalKBDatabaseAccess \
  --policy-document file:///tmp/kb-db-policy.json
```

Attach this policy to each team member's IAM user or role:

```bash
aws iam attach-user-policy \
  --user-name jason \
  --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/PersonalKBDatabaseAccess"
```

## 5. Install dependencies

```bash
# From a clone (recommended for team setup)
git clone https://github.com/jason-weddington/personal-kb-mcp.git
cd personal-kb-mcp
uv sync --extra postgres --extra iam

# Or with uvx (installs on first run)
uvx --from "personal-kb-mcp[postgres,iam]" personal-kb
```

## 6. Configure MCP for each team member

Each person adds this to their MCP client config (`~/.claude.json`, `claude_desktop_config.json`, etc.):

```json
{
  "mcpServers": {
    "personal-kb": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/personal-kb-mcp", "personal-kb"],
      "env": {
        "KB_DATABASE_URL": "postgresql://kb_iam@<cluster-endpoint>:5432/personal_kb",
        "KB_PG_IAM_AUTH": "TRUE",
        "KB_PG_REGION": "us-east-1",
        "KB_CONTRIBUTOR": "jason",
        "KB_TEAM": "platform",
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

Key points:
- **No password in the URL** — the IAM token factory provides it
- **`KB_PG_IAM_AUTH=TRUE`** — enables SigV4 token generation via `boto3`
- **`KB_CONTRIBUTOR`** — your name, stamped on every entry you create
- **`KB_TEAM`** — your team, shown alongside contributor in search results
- AWS credentials must be available through the standard chain (env vars, `~/.aws/credentials`, instance profile)

## 7. Verify the connection

```bash
# Quick smoke test — should print server version and exit
KB_DATABASE_URL="postgresql://kb_iam@<cluster-endpoint>:5432/personal_kb" \
KB_PG_IAM_AUTH=TRUE \
KB_PG_REGION=us-east-1 \
uv run python -c "
import asyncio
from personal_kb.db.connection import create_connection
async def check():
    db = await create_connection()
    cursor = await db.execute('SELECT version()')
    row = await cursor.fetchone()
    print(row[0])
    await db.close()
asyncio.run(check())
"
```

## Security notes

- IAM tokens are valid for ~15 minutes. The server generates a fresh token on each new pool connection — no long-lived credentials.
- SSL/TLS is mandatory with IAM auth. The server creates an SSL context automatically using system CA certificates.
- The security group (`sg-xxxx`) should restrict inbound 5432 to your team's IP ranges or VPN.
- If your team uses SSO/assumed roles, the `rds-db:connect` permission follows the normal IAM policy chain — attach the policy to the role, not the user.

---

# Migrating from SQLite to Postgres

The migration script copies your SQLite knowledge base to Postgres, preserving entries, versions, graph, and audit trail. It handles two scenarios: migrating into an empty database (fresh) and merging into a database that already has entries.

## Quick start

```bash
# Preview what will be migrated (read-only)
uv run python scripts/migrate_sqlite_to_pg.py --dry-run \
  ~/.local/share/personal_kb/knowledge.db

# Run the migration
uv run python scripts/migrate_sqlite_to_pg.py \
  ~/.local/share/personal_kb/knowledge.db

# With attribution (stamps your name on migrated entries)
uv run python scripts/migrate_sqlite_to_pg.py \
  --contributor jason --team platform \
  ~/.local/share/personal_kb/knowledge.db
```

The target Postgres connection comes from environment variables — set `KB_DATABASE_URL` before running. For Aurora IAM auth, also set `KB_PG_IAM_AUTH=TRUE` and `KB_PG_REGION`.

## CLI reference

```
uv run python scripts/migrate_sqlite_to_pg.py [OPTIONS] SOURCE

positional:
  SOURCE              Path to SQLite database file

options:
  --dry-run           Show what would be migrated without writing
  --skip-embeddings   Skip the re-embed step (rebuild later via kb_maintain)
  --contributor NAME  Override KB_CONTRIBUTOR for attribution stamping
  --team NAME         Override KB_TEAM for attribution stamping
```

## How it works

### Mode detection

The script checks whether the target Postgres database already has entries:

- **Fresh mode** (0 entries in target) — IDs are copied as-is.
- **Merge mode** (existing entries) — Source entry IDs are remapped to avoid collisions. If the target's highest ID is `kb-00319`, source `kb-00001` becomes `kb-00320`, etc.

ID remapping is applied consistently across all tables: `knowledge_entries.id`, `superseded_by` references, `entry_versions.entry_id`, `entry:kb-*` graph nodes and edges, and `entry_ids` JSON arrays in `ingested_files`.

### What's copied

| Table | Notes |
|---|---|
| `knowledge_entries` | ID + superseded_by remapped. `has_embedding` reset to 0 (rebuilt). |
| `entry_versions` | Version history preserved. |
| `graph_nodes` | `entry:kb-*` nodes remapped; entity nodes (tag/project/person/etc) deduplicated via ON CONFLICT. |
| `graph_edges` | Source/target remapped where entry-based. |
| `ingested_files` | `entry_ids` JSON array remapped. |
| `audit_events` | Historical events preserved. New "migrated" event added per entry. |

### What's skipped

| Table | Reason |
|---|---|
| `knowledge_vec` | Rebuilt by the re-embed step (binary format differs between sqlite-vec and pgvector). |
| `knowledge_fts` | Postgres tsvector trigger auto-populates on INSERT. |
| `search_events` | Session telemetry — not worth migrating. |
| `agent_feedback` | Session telemetry — not worth migrating. |
| `deployment_config` | Target seeds its own via `apply_schema()`. |
| `entry_id_seq` | Explicitly set to max(all IDs) + 1 after copy. |

### Attribution stamping

When `--contributor` (or `KB_CONTRIBUTOR`) is set, the script stamps migrated rows that have no existing attribution. Entries that already have a `contributor` set are left untouched. This is useful when migrating a personal SQLite DB into a shared team Postgres — your entries get tagged as yours.

### Embedding rebuild

After copying data, the script re-embeds all active entries via Ollama. This is necessary because sqlite-vec uses packed binary vectors while pgvector uses native arrays — the raw bytes aren't compatible.

- If Ollama isn't running, embeddings are skipped with a warning.
- Use `--skip-embeddings` to defer intentionally.
- Rebuild later with: `kb_maintain rebuild_embeddings` (force=True).
- The KB works immediately without embeddings — FTS and graph search are available from the start.

### Error handling

Individual row failures are logged but don't abort the migration. The script collects all errors and reports them at the end. This means a single problematic entry won't prevent the rest of your data from migrating. The exit code is 0 on success, 1 if any errors occurred.

## Examples

### Fresh migration (empty Postgres)

```bash
export KB_DATABASE_URL="postgresql://user:pass@localhost/my_kb"

uv run python scripts/migrate_sqlite_to_pg.py \
  ~/.local/share/personal_kb/knowledge.db
```

### Merge into team database

```bash
export KB_DATABASE_URL="postgresql://kb_iam@aurora-cluster.xxx.us-east-1.rds.amazonaws.com:5432/personal_kb"
export KB_PG_IAM_AUTH=TRUE
export KB_PG_REGION=us-east-1

uv run python scripts/migrate_sqlite_to_pg.py \
  --contributor jason --team platform \
  ~/.local/share/personal_kb/knowledge.db
```

### Dry run against Aurora

```bash
export KB_DATABASE_URL="postgresql://kb_iam@aurora-cluster.xxx.us-east-1.rds.amazonaws.com:5432/personal_kb"
export KB_PG_IAM_AUTH=TRUE

uv run python scripts/migrate_sqlite_to_pg.py --dry-run \
  ~/.local/share/personal_kb/knowledge.db
```

Output:

```
Source: /Users/jason/.local/share/personal_kb/knowledge.db

Source tables:
  knowledge_entries: 124 rows
  entry_versions: 146 rows
  graph_nodes: 819 rows
  graph_edges: 1193 rows
  ingested_files: 1 rows
  audit_events: 0 rows

Target: aurora-cluster.xxx.us-east-1.rds.amazonaws.com:5432/personal_kb
Schema applied.

Mode: MERGE (offset=319)

Dry run: 2283 rows would be migrated.
  Entry IDs remapped with offset +319
  118 active entries would be re-embedded via Ollama.
```
