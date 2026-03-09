// ============================================================================
// Schema ID Normalization Migration
// ============================================================================
//
// Migrates all graph nodes to use 'id' as the universal identifier field.
// Previously, 9 classes used non-id identifiers (path, name, hostname, symbol).
//
// Run each section separately in the Neo4j browser or via:
//   uv run imas-codex graph shell < scripts/migrate_schema_ids.cypher
//
// IMPORTANT: Back up the graph before running:
//   uv run imas-codex graph backup
// ============================================================================


// ── Phase 1: Backfill 'id' on nodes that lack it ────────────────────────

// DataNode: id = facility_id + ':' + data_source_name + ':' + path
// Most DataNodes already have id set by extraction.py; backfill any gaps
MATCH (n:DataNode)
WHERE n.id IS NULL
SET n.id = n.facility_id + ':' + n.data_source_name + ':' + n.path
RETURN 'DataNode backfilled' AS step, count(n) AS affected;

// DataSource: id = facility_id + ':' + name
MATCH (n:DataSource)
WHERE n.id IS NULL
SET n.id = n.facility_id + ':' + n.name
RETURN 'DataSource backfilled' AS step, count(n) AS affected;

// MDSplusServer: id = facility_id + ':' + hostname
MATCH (n:MDSplusServer)
WHERE n.id IS NULL
SET n.id = n.facility_id + ':' + n.hostname
RETURN 'MDSplusServer backfilled' AS step, count(n) AS affected;

// DiscoveryRoot: id = facility_id + ':' + path
MATCH (n:DiscoveryRoot)
WHERE n.id IS NULL
SET n.id = n.facility_id + ':' + n.path
RETURN 'DiscoveryRoot backfilled' AS step, count(n) AS affected;

// Diagnostic: id = facility_id + ':' + name
MATCH (n:Diagnostic)
WHERE n.id IS NULL
SET n.id = n.facility_id + ':' + n.name
RETURN 'Diagnostic backfilled' AS step, count(n) AS affected;

// AnalysisCode: id = facility_id + ':' + name
MATCH (n:AnalysisCode)
WHERE n.id IS NULL
SET n.id = n.facility_id + ':' + n.name
RETURN 'AnalysisCode backfilled' AS step, count(n) AS affected;

// IDS: id = name (globally unique, no facility prefix)
MATCH (n:IDS)
WHERE n.id IS NULL
SET n.id = n.name
RETURN 'IDS backfilled' AS step, count(n) AS affected;

// IdentifierSchema: id = name
MATCH (n:IdentifierSchema)
WHERE n.id IS NULL
SET n.id = n.name
RETURN 'IdentifierSchema backfilled' AS step, count(n) AS affected;

// Unit: id = symbol
MATCH (n:Unit)
WHERE n.id IS NULL
SET n.id = n.symbol
RETURN 'Unit backfilled' AS step, count(n) AS affected;


// ── Phase 2: Drop old constraints ───────────────────────────────────────
// These use the old identifier fields. Names follow the pattern: label_field

DROP CONSTRAINT datanode_path IF EXISTS;
DROP CONSTRAINT datasource_name IF EXISTS;
DROP CONSTRAINT mdsplusserver_hostname IF EXISTS;
DROP CONSTRAINT discoveryroot_path IF EXISTS;
DROP CONSTRAINT diagnostic_name IF EXISTS;
DROP CONSTRAINT analysiscode_name IF EXISTS;
DROP CONSTRAINT ids_name IF EXISTS;
DROP CONSTRAINT identifierschema_name IF EXISTS;
DROP CONSTRAINT unit_symbol IF EXISTS;


// ── Phase 3: Create new constraints on 'id' ─────────────────────────────
// These match the output of GraphSchema.constraint_statements()

// Composite constraints (id + facility_id) for facility-scoped nodes
CREATE CONSTRAINT datanode_id IF NOT EXISTS
FOR (n:DataNode) REQUIRE (n.id, n.facility_id) IS UNIQUE;

CREATE CONSTRAINT datasource_id IF NOT EXISTS
FOR (n:DataSource) REQUIRE (n.id, n.facility_id) IS UNIQUE;

CREATE CONSTRAINT mdsplusserver_id IF NOT EXISTS
FOR (n:MDSplusServer) REQUIRE (n.id, n.facility_id) IS UNIQUE;

CREATE CONSTRAINT discoveryroot_id IF NOT EXISTS
FOR (n:DiscoveryRoot) REQUIRE (n.id, n.facility_id) IS UNIQUE;

CREATE CONSTRAINT diagnostic_id IF NOT EXISTS
FOR (n:Diagnostic) REQUIRE (n.id, n.facility_id) IS UNIQUE;

CREATE CONSTRAINT analysiscode_id IF NOT EXISTS
FOR (n:AnalysisCode) REQUIRE (n.id, n.facility_id) IS UNIQUE;

// Simple constraints for globally unique nodes
CREATE CONSTRAINT ids_id IF NOT EXISTS
FOR (n:IDS) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT identifierschema_id IF NOT EXISTS
FOR (n:IdentifierSchema) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT unit_id IF NOT EXISTS
FOR (n:Unit) REQUIRE n.id IS UNIQUE;


// ── Phase 4: Verify ─────────────────────────────────────────────────────

// Check for any nodes still missing id
MATCH (n)
WHERE n.id IS NULL
  AND (n:DataNode OR n:DataSource OR n:MDSplusServer OR n:DiscoveryRoot
       OR n:Diagnostic OR n:AnalysisCode OR n:IDS OR n:IdentifierSchema OR n:Unit)
RETURN labels(n)[0] AS label, count(n) AS missing_id;

// Show all constraints
SHOW CONSTRAINTS;
