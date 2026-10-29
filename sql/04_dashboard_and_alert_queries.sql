-- ==== DATASET: kpis ====
SELECT COUNT(*) AS complaints_scored,
       ROUND(100 * AVG(CASE WHEN decision = 'auto_route' THEN 1 ELSE 0 END), 1) AS auto_routed_pct,
       ROUND(100 * AVG(CASE WHEN decision = 'auto_route' THEN CAST(correct AS INT) END), 1) AS auto_routed_accuracy_pct,
       SUM(CASE WHEN decision = 'human_review' THEN 1 ELSE 0 END) AS sent_to_review
FROM workspace.complaint_triage.routing_decisions
-- ==== DATASET: monthly_health ====
WITH monthly AS (
  SELECT to_date(concat(month, '-01')) AS month_start,
         100 * AVG(CASE WHEN decision = 'auto_route' THEN 1 ELSE 0 END) AS auto_routed,
         100 * AVG(CASE WHEN decision = 'auto_route' THEN CAST(correct AS INT) END) AS auto_routed_accuracy
  FROM workspace.complaint_triage.routing_decisions
  GROUP BY month
)
SELECT month_start, 'Auto-routed share' AS metric, ROUND(auto_routed, 1) AS pct FROM monthly
UNION ALL
SELECT month_start, 'Accuracy when auto-routed', ROUND(auto_routed_accuracy, 1) FROM monthly
UNION ALL
SELECT month_start, 'Accuracy target (95%)', 95.0 FROM monthly
-- ==== DATASET: product_mix ====
SELECT to_date(concat(month, '-01')) AS month_start, product, complaints, share_pct
FROM workspace.complaint_triage.gold_monthly_mix
-- ==== DATASET: queue_accuracy ====
SELECT actual_product AS queue,
       CASE WHEN month >= '2026-01' THEN '2026' ELSE '2025' END AS period,
       ROUND(100 * AVG(CAST(correct AS INT)), 1) AS accuracy_pct,
       COUNT(*) AS complaints
FROM workspace.complaint_triage.routing_decisions
GROUP BY actual_product, CASE WHEN month >= '2026-01' THEN '2026' ELSE '2025' END
-- ==== DATASET: scoring_runs ====
SELECT scored_at, model_version, complaints_scored,
       ROUND(100 * auto_routed_share, 1) AS auto_routed_pct,
       ROUND(100 * auto_routed_accuracy, 1) AS auto_routed_accuracy_pct,
       remaining_after_run
FROM workspace.complaint_triage.scoring_runs
ORDER BY scored_at
-- ==== ALERT QUERY (optional, Part D) ====
SELECT ROUND(100 * auto_routed_accuracy, 1) AS latest_auto_routed_accuracy_pct
FROM workspace.complaint_triage.scoring_runs
ORDER BY scored_at DESC
LIMIT 1
