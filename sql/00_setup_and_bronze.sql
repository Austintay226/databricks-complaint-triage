-- Stage 1-2: create the schema and volume, then load the raw CSVs into a bronze Delta table.
-- Run in the Databricks SQL Editor. Upload complaints_sample.csv and complaints_2026.csv
-- (produced by the original complaint-triage project) to the raw_files volume between steps 1 and 2.

-- 1. A schema for the project and a volume for raw files
CREATE SCHEMA IF NOT EXISTS workspace.complaint_triage;
CREATE VOLUME IF NOT EXISTS workspace.complaint_triage.raw_files;

-- 2. Load both files into one bronze table, tagged with their source and load time.
--    multiLine and escape are needed because narratives contain line breaks and quotes.
CREATE OR REPLACE TABLE workspace.complaint_triage.bronze_complaints AS
SELECT *, '2024-2025' AS source_period, current_timestamp() AS ingested_at
FROM read_files('/Volumes/workspace/complaint_triage/raw_files/complaints_sample.csv',
                format => 'csv', header => true, multiLine => true, escape => '"')
UNION ALL
SELECT *, '2026' AS source_period, current_timestamp() AS ingested_at
FROM read_files('/Volumes/workspace/complaint_triage/raw_files/complaints_2026.csv',
                format => 'csv', header => true, multiLine => true, escape => '"');

-- 3. Reconcile against the source: expect 38,622 (2024-2025) and 2,216 (2026)
SELECT source_period, COUNT(*) AS complaints
FROM workspace.complaint_triage.bronze_complaints
GROUP BY source_period;

-- 4. Delta keeps every version of the table
DESCRIBE HISTORY workspace.complaint_triage.bronze_complaints;
