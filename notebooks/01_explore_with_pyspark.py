# Databricks notebook source
# MAGIC %md
# MAGIC # Stage 3: Exploring complaints with notebooks and PySpark
# MAGIC
# MAGIC **Goal:** learn how notebooks work, how Spark DataFrames differ from pandas, and how to mix Python and SQL.
# MAGIC
# MAGIC **How to use this notebook:** run one cell at a time with **Shift + Enter**. Read the text cell above each code cell first.
# MAGIC Before the first cell, use the compute selector at the top right to connect to **Serverless**.
# MAGIC
# MAGIC Lessons:
# MAGIC 1. Notebooks and the `spark` session
# MAGIC 2. Lazy evaluation: transformations vs. actions
# MAGIC 3. Spark vs. pandas, side by side
# MAGIC 4. Mixing SQL and Python
# MAGIC 5. Exploring the text: did 2026 narratives change?
# MAGIC 6. Verifying an upstream claim: are the template letters really gone?
# MAGIC 7. Delta time travel: break the table, then restore it
# MAGIC 8. Moving small results to pandas

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lesson 1: Notebooks and the `spark` session
# MAGIC
# MAGIC A notebook is a list of **cells**. This one is a Markdown cell (text); the next is a Python cell.
# MAGIC Every Databricks notebook already has a variable called **`spark`**, your connection to the Spark engine.
# MAGIC `spark.table(...)` gives you a **DataFrame**: Spark's version of a table you can work with in Python.
# MAGIC
# MAGIC `printSchema()` shows the columns and their types. Notice Spark worked out the types (for example, `date`) from your CSV.

# COMMAND ----------

from pyspark.sql import functions as F

TABLE = "workspace.complaint_triage.bronze_complaints"
df = spark.table(TABLE)
df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lesson 2: Lazy evaluation
# MAGIC
# MAGIC This is the biggest difference from pandas. Spark is **lazy**:
# MAGIC - **Transformations** (`filter`, `select`, `groupBy`, `withColumn`) do no work. They just add a step to a plan.
# MAGIC - **Actions** (`count`, `display`, `collect`, writing a table) make Spark actually run the plan.
# MAGIC
# MAGIC Why: Spark looks at the whole plan before running it, so it can optimize, for example by reading only the columns you need.
# MAGIC Run the next cell: the first line returns instantly because nothing ran yet. `count()` is what does the work.
# MAGIC `display()` is a Databricks function that shows a DataFrame as an interactive table (try sorting a column, or the chart button).

# COMMAND ----------

recent = df.filter(F.col("source_period") == "2024-2025")   # transformation: instant, nothing runs
print("Rows in 2024-2025:", recent.count())                 # action: Spark runs the plan now
display(recent.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lesson 3: Spark vs. pandas, side by side
# MAGIC
# MAGIC Here is the product mix you already checked in SQL, written in PySpark. The pandas version is in the comment for comparison.
# MAGIC
# MAGIC Differences to notice:
# MAGIC - Spark DataFrames are **immutable**: every step returns a new DataFrame instead of changing one in place.
# MAGIC - Columns are referenced with `F.col("name")` and built with functions from `pyspark.sql.functions`.
# MAGIC - The work is **distributed**: on a real cluster, each machine counts its share of rows and Spark combines the results. Same code, whether it's 40,000 rows or 400 million.
# MAGIC
# MAGIC `explain()` prints the plan Spark built. You don't need to read it in detail; just notice it exists, because "check the query plan" is a real troubleshooting step for slow jobs.

# COMMAND ----------

# pandas equivalent:
#   mix = recent_pd["product"].value_counts().rename("complaints").to_frame()
#   mix["pct"] = (100 * mix["complaints"] / mix["complaints"].sum()).round(1)

total = recent.count()
mix = (recent.groupBy("product")
             .agg(F.count("*").alias("complaints"))
             .withColumn("pct", F.round(100 * F.col("complaints") / total, 1))
             .orderBy(F.desc("complaints")))
display(mix)
mix.explain()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lesson 4: Mixing SQL and Python
# MAGIC
# MAGIC Two ways to use SQL in a notebook:
# MAGIC 1. A cell that starts with **`%sql`** runs as SQL (the next cell).
# MAGIC 2. **`spark.sql("...")`** runs SQL from Python and returns a DataFrame, so you can keep working on it in Python (the cell after).
# MAGIC
# MAGIC Pick whichever is clearer for the task. Teams usually mix both.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT source_period, COUNT(*) AS complaints, COUNT(DISTINCT product) AS queues
# MAGIC FROM workspace.complaint_triage.bronze_complaints
# MAGIC GROUP BY source_period

# COMMAND ----------

by_period = spark.sql(f"""
    SELECT source_period, product, COUNT(*) AS complaints
    FROM {TABLE}
    GROUP BY source_period, product
""")
# Back in Python: the share of each product within each period
period_totals = by_period.groupBy("source_period").agg(F.sum("complaints").alias("period_total"))
shares = (by_period.join(period_totals, "source_period")
                   .withColumn("pct", F.round(100 * F.col("complaints") / F.col("period_total"), 1))
                   .orderBy("product", "source_period"))
display(shares)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lesson 5: Exploring the text
# MAGIC
# MAGIC Your Python diagnostic found that 2026 narratives were longer. Let's check that in Spark.
# MAGIC - `F.split` breaks text into words and `F.size` counts them.
# MAGIC - `F.percentile_approx(..., 0.5)` is an approximate median. Exact medians are expensive on huge data, so Spark approximates by default; at this size it's effectively exact.
# MAGIC
# MAGIC **Question to answer:** is the longer-narrative pattern true across products, or only for some?

# COMMAND ----------

with_words = df.withColumn("words", F.size(F.split(F.trim(F.col("narrative")), r"\s+")))

length_by_period = (with_words.groupBy("source_period")
                              .agg(F.count("*").alias("complaints"),
                                   F.percentile_approx("words", 0.5).alias("median_words"),
                                   F.round(F.avg("words"), 0).alias("mean_words")))
display(length_by_period)

length_by_product = (with_words.groupBy("product")
                               .pivot("source_period", ["2024-2025", "2026"])
                               .agg(F.percentile_approx("words", 0.5))
                               .orderBy("product"))
display(length_by_product)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lesson 6: Verifying an upstream claim
# MAGIC
# MAGIC Your Python prep step said it removed all duplicate template letters. A good habit: **don't trust upstream steps, verify them.**
# MAGIC This rebuilds the same normalization in Spark (lowercase, strip the `XXXX` redaction masks, tag dollar amounts, collapse spaces) and looks for any narrative that still appears more than once.
# MAGIC
# MAGIC `regexp_replace` is Spark's regex find-and-replace. Expected result: **zero** duplicate groups. If you see some, that's a real finding worth investigating.

# COMMAND ----------

normalized = F.lower(F.substring(F.col("narrative"), 1, 4000))
normalized = F.regexp_replace(normalized, r"\{\$?[\d,]*\.?\d*\}", " dollaramount ")
normalized = F.regexp_replace(normalized, r"\bx{2,}\b", " ")
normalized = F.trim(F.regexp_replace(normalized, r"\s+", " "))

dupes = (df.withColumn("normalized", normalized)
           .groupBy("normalized")
           .agg(F.count("*").alias("copies"), F.collect_set("product").alias("products"))
           .filter(F.col("copies") > 1))
print("Duplicate narrative groups:", dupes.count())
display(dupes.orderBy(F.desc("copies")).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lesson 7: Delta time travel (break it, then fix it)
# MAGIC
# MAGIC You'll simulate a real mistake: someone accidentally deletes rows from a production table.
# MAGIC Run the four SQL cells below in order and watch the counts:
# MAGIC 1. Delete every "Other" complaint (the "accident").
# MAGIC 2. Count again: rows are gone.
# MAGIC 3. Look at the history: the delete is recorded as a new version.
# MAGIC 4. Restore the table to version 0 and count again: everything is back.
# MAGIC
# MAGIC This is a real recovery technique, and a good answer to "what would you do if bad data or a bad change hit production?"

# COMMAND ----------

# MAGIC %sql
# MAGIC DELETE FROM workspace.complaint_triage.bronze_complaints WHERE product = 'Other'

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) AS rows_after_accident FROM workspace.complaint_triage.bronze_complaints

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY workspace.complaint_triage.bronze_complaints

# COMMAND ----------

# MAGIC %sql
# MAGIC RESTORE TABLE workspace.complaint_triage.bronze_complaints TO VERSION AS OF 0;
# MAGIC SELECT COUNT(*) AS rows_after_restore FROM workspace.complaint_triage.bronze_complaints;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lesson 8: Moving small results to pandas
# MAGIC
# MAGIC `toPandas()` pulls a Spark DataFrame onto a single machine as a pandas DataFrame.
# MAGIC That's fine for **small results** like this summary table, and useful when a library only speaks pandas.
# MAGIC Never call it on a huge table: everything has to fit in one machine's memory, which defeats the point of Spark.
# MAGIC Rule of thumb: do the heavy work in Spark, then bring the small answer to pandas.

# COMMAND ----------

mix_pd = mix.toPandas()
print(type(mix_pd))
mix_pd.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wrap-up
# MAGIC
# MAGIC You've used: notebooks, the `spark` session, lazy evaluation, PySpark functions, `%sql` cells, `spark.sql`, pivots, regex, Delta history and `RESTORE`, and `toPandas`.
# MAGIC
# MAGIC **Write down your answers** (good interview material):
# MAGIC 1. Were 2026 narratives longer for every product, or only some?
# MAGIC 2. Did the duplicate check come back at zero?
# MAGIC 3. In your own words: why is Spark lazy, and why does that help?
# MAGIC
# MAGIC **Next (stage 4):** build silver and gold tables in a Lakeflow pipeline with data quality rules.
