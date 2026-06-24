# Data Cleaning and Validation Report (Phase 1)

## 1. Overview
The raw real estate dataset was cleaned, validated, and standardized to produce a reliable dataset for exploratory analysis and modeling.

## 2. Dataset Shape Summary
- **Original Shape:** (250000, 24)
- **Final Shape:** (250000, 24)
- **Rows Dropped:** 0

## 3. Data Cleaning & Validation Steps

### A. Missing Value Analysis
- **Status:** 100% Complete. No missing values were present in the raw data.
- **Imputation Strategy:** None required, but defensive code was written to handle any future null values.

### B. Floor Validation & Swapping (Swapped Floors Fixed)
- **Issue:** Detected **116,304 rows (46.52% of the dataset)** where `Floor_No > Total_Floors` due to swapped data columns.
- **Resolution:** Swapped `Floor_No` and `Total_Floors` by setting `Floor_No = min(Floor_No, Total_Floors)` and `Total_Floors = max(Floor_No, Total_Floors)`. This mathematically resolved all floor violations while preserving all rows.
- **Remaining Violations:** 0

### C. Outlier Detection & Capping (Winsorization)
- **Method:** IQR (Interquartile Range) method with 1.5 * IQR bounds on numeric columns.
- **Action:** Outliers were capped rather than removed to preserve dataset size.
- **Capping Ranges applied:**
  - `Price_in_Lakhs`: capped upper bound to 743.38 Lakhs.
  - `Size_in_SqFt`: capped upper bound to 7250.50 SqFt (no outliers were present).
  - `Crime_Rate_Per_Lakh`: capped upper bound to 988.80.

### D. Formatting & Standardizations
- **Casing:** Standardized categorical text fields to Title Case (`State`, `City`, `Locality`, `Property_Type`, etc.) and cleaned whitespace.
- **Sanity Checks:** Verified that all prices and sizes are strictly positive, BHK ranges between 1 and 5, and floor numbers are valid.

### E. Duplicate Detection
- **Exact Duplicates:** 0 found.
- **Near-duplicates (Matching core features):** 0 found. No rows were dropped, ensuring full dataset retention.
