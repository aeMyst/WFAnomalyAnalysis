# Wind Farm SCADA Anomaly Analysis

Wind Farm SCADA Anomaly Analysis is a **data analysis and visualization project** designed to explore and detect anomalous behavior in wind turbine SCADA (Supervisory Control and Data Acquisition) data.

The project processes raw turbine event datasets, computes anomaly metrics, and provides an **interactive dashboard** to explore differences between normal and anomalous turbine behavior.

The dashboard allows users to:

- Compare anomaly severity across wind farms  
- Analyze sensor behavior during anomalous events  
- Visualize wind speed vs power output patterns  
- Identify sensors most associated with abnormal turbine behavior  
- Understand operational status distributions  

The project focuses on building a **reproducible pipeline from raw SCADA datasets to interactive visual insights**.

---

# Full Analysis Document

A detailed explanation of the dataset, analysis approach, anomaly metrics, and insights can be found here:

**Analysis Document**  
https://docs.google.com/document/d/16qEFLPZ0mAKWaj6AkJAsAf0o4h5gCfdwSAt3n4o_8uY/edit?tab=t.0

---

# Project Architecture

The project consists of two major components:

1. **Data Processing Pipeline**
2. **Interactive Visualization Dashboard**

```
Raw SCADA Data
      │
      ▼
Event Processing Pipeline
      │
      ▼
Processed Parquet Data
      │
      ▼
Streamlit Dashboard
      │
      ▼
Interactive Visual Analysis
```

---

# Dataset Overview

The dataset contains SCADA time series data from multiple wind farms.

Characteristics:

- **95 event datasets**
- **36 turbines**
- Multiple wind farms
- Long time series sensor data
- Hundreds of sensor features depending on farm

### Feature counts by farm

| Wind Farm | Number of Features |
|----------|-------------------|
| Wind Farm A | 86 |
| Wind Farm B | 257 |
| Wind Farm C | 957 |

Because the farms have very different feature counts, **severity metrics are normalized across sensors** to ensure fair comparisons.

---

# Key Metrics Computed

The processing pipeline computes several anomaly metrics.

### Z-Shift

Difference between event window sensor values and baseline training values normalized by standard deviation.

### Volatility Ratio

Change in sensor variance between baseline and event window.

### Energy Score

Aggregate sensor deviation across the entire event.

### Normalized Severity Score

Cross-farm comparable anomaly severity metric combining:

- RMS standardized sensor deviation  
- volatility shifts

### Log Severity Score

Log-transformed severity used in visualizations to handle extreme outliers.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/aeMyst/WFAnomalyAnalysis.git
cd WFAnomalyAnalysis
```

---

## 2. Create a Virtual Environment

Recommended:

```bash
python -m venv venv
```

Activate the environment.

### Mac / Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If a requirements file is unavailable, install manually:

```bash
pip install pandas numpy plotly streamlit python-dotenv pyarrow
```

---

# Environment Configuration

Create a `.env` file in the project root.

Example:

```
SCADA_ROOT=/path/to/raw/scada/data
OUTPUT_PATH=/path/to/processed/output
```

### Environment Variables

| Variable | Description |
|--------|--------|
| SCADA_ROOT | Location of the raw SCADA dataset |
| OUTPUT_PATH | Directory where processed data will be stored |

---

# Running the Data Processing Pipeline

The pipeline reads raw SCADA datasets and generates structured parquet outputs.

Run:

```bash
python event_processing.py
```

### Output Structure

```
OUTPUT_PATH/
    summary_all_farms.parquet
    wind_farm_a_event_summary.parquet
    wind_farm_b_event_summary.parquet
    wind_farm_c_event_summary.parquet
    events/
        wind_farm_a/
        wind_farm_b/
        wind_farm_c/
```

These outputs contain:

- event-level anomaly metrics  
- normalized severity scores  
- sensor statistics  
- processed event data  

---

# Running the Dashboard

After running the processing pipeline, launch the dashboard.

From the dashboard directory:

```bash
streamlit run app.py
```

Streamlit will launch the dashboard locally:

```
http://localhost:8501
```

---

# Dashboard Visualizations

The dashboard includes several interactive plots.

### Severity Distribution

Histogram of anomaly severity across events.

### Severity Comparison

Box plot comparing anomaly vs normal events.

### Wind Speed vs Power Output

Scatter plot showing deviations from expected turbine power curves.

### Sensor Importance

Ranking of sensors most associated with anomalous behavior.

---
