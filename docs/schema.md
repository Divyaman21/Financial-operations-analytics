# BI data dictionary (Power BI / Tableau)

Grain, keys, and refresh assumptions for files under `artifacts/bi_export/` (also mirrored as CSV).

## `customers_dim`

| Column | Grain / description |
|--------|---------------------|
| `customer_id` | Surrogate key; one row per customer as of snapshot. |
| `first_purchase`, `last_purchase` | First and last observed order dates (≤ `as_of_date`). |
| `recency_days` | Days from `last_purchase` to `as_of_date`. |
| `frequency_365d` | Order count in trailing 365 days. |
| `monetary_365d` | Sum revenue in trailing 365 days. |
| `tenure_days` | Days between first and last purchase. |
| `rfm_segment` | Rule-based segment from RFM scores. |
| `kmeans_cluster` | Unsupervised cluster id (-1 if missing features). |
| `churned` | 1 if `recency_days` > inactive threshold (default 90). |
| `clv_proxy` | Heuristic value score for weighting (not GAAP CLV). |
| `as_of_date` | Snapshot date for all features (YYYY-MM-DD). |

**Primary key:** (`customer_id`, `as_of_date`).

## `orders_fact`

| Column | Description |
|--------|-------------|
| `customer_id` | Customer foreign key. |
| `order_date` | Order date (string YYYY-MM-DD in export). |
| `revenue` | Order revenue. |
| `category` | Optional product/category bucket. |

**Grain:** one row per order line (here: per order).

## `model_scores_fact`

| Column | Description |
|--------|-------------|
| `customer_id` | Join to `customers_dim`. |
| `as_of_date` | Snapshot date. |
| `rfm_segment`, `kmeans_cluster` | Segmentation at snapshot. |
| `churn_prob` | Holdout-trained model refit on full sample; churn features use a **lagged** snapshot (default 60 days before `as_of_date`) vs label at `as_of_date`. |
| `duration_days`, `event_observed` | Survival setup from first purchase; see report. |
| `forecast_model_best` | Name of lowest holdout RMSE forecast method in this run. |
| `weekly_revenue_last` | Last in-sample weekly revenue (context KPI). |

**Refresh logic:** rebuild from raw orders with a chosen `as_of_date`; never mix future orders into features.
hold-winters model 