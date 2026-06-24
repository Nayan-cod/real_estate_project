import os
import sys
import joblib
import pandas as pd

def run_tests():
    print("==================================================")
    print("      REAL ESTATE ML PIPELINE VALIDATION TEST      ")
    print("==================================================")
    
    passed_tests = 0
    failed_tests = 0
    
    def assert_exists(path, description):
        def safe_print(msg):
            try:
                print(msg)
            except UnicodeEncodeError:
                print(msg.encode('ascii', errors='replace').decode('ascii'))
        nonlocal passed_tests, failed_tests
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size > 0:
                safe_print(f"[PASS] PASSED: {description} exists and is populated. ({path}, size: {size:,} bytes)")
                passed_tests += 1
                return True
            else:
                safe_print(f"[FAIL] FAILED: {description} exists but is EMPTY! ({path})")
                failed_tests += 1
                return False
        else:
            safe_print(f"[FAIL] FAILED: {description} NOT FOUND! ({path})")
            failed_tests += 1
            return False

    # 1. Test Notebooks
    print("\n--- 1. Testing Jupyter Notebooks ---")
    notebooks = [
        "notebooks/01_data_cleaning.ipynb",
        "notebooks/02_eda.ipynb",
        "notebooks/03_feature_engineering.ipynb",
        "notebooks/04_model_development.ipynb",
        "notebooks/05_explainable_ai.ipynb",
        "notebooks/06_recommendation_risk_forecast.ipynb"
    ]
    for i, nb in enumerate(notebooks, 1):
        assert_exists(nb, f"Notebook {i:02d}")

    # 2. Test Generated Datasets
    print("\n--- 2. Testing Generated CSV Datasets ---")
    datasets = [
        ("data/interim/cleaned_dataset.csv", "Cleaned interim dataset"),
        ("data/interim/feature_engineered_dataset.csv", "Feature engineered interim dataset"),
        ("data/processed/final_dataset_with_recommendations.csv", "Final enriched dataset"),
        ("data/processed/final_test_predictions.csv", "Model test set predictions"),
        ("data/processed/model_comparison_table.csv", "Model comparison metric ledger")
    ]
    for path, desc in datasets:
        if assert_exists(path, desc):
            # Check row count and columns
            df = pd.read_csv(path, nrows=5)
            print(f"   Shape Info: Columns available: {len(df.columns)}")

    # 3. Test Serialized Model and Preprocessors
    print("\n--- 3. Testing Serialized Model Binaries ---")
    binaries = [
        ("outputs/models/best_model.pkl", "Trained ML Model"),
        ("outputs/models/scaler.pkl", "Standard Scaler"),
        ("outputs/models/one_hot_encoder.pkl", "One-Hot Encoder"),
        ("outputs/models/target_encoders.pkl", "Target Encoders Map Dictionary")
    ]
    
    all_binaries_present = True
    for path, desc in binaries:
        if not assert_exists(path, desc):
            all_binaries_present = False
            
    if all_binaries_present:
        try:
            print("   Attempting to load serialized binaries...")
            model = joblib.load("outputs/models/best_model.pkl")
            scaler = joblib.load("outputs/models/scaler.pkl")
            ohe = joblib.load("outputs/models/one_hot_encoder.pkl")
            te = joblib.load("outputs/models/target_encoders.pkl")
            print("   [PASS] SUCCESS: All serialized binaries loaded into memory successfully!")
            passed_tests += 1
        except Exception as e:
            print(f"   [FAIL] ERROR: Failed to load serialized binaries: {e}")
            failed_tests += 1

    # 4. Test Streamlit Dashboard Files
    print("\n--- 4. Testing Streamlit Dashboard Files ---")
    dashboard_files = [
        "app/app.py",
        "app/utils.py",
        "app/pages/1_📊_EDA_Dashboard.py",
        "app/pages/2_🏠_Price_Prediction.py",
        "app/pages/3_💰_Investment_Recommendation.py",
        "app/pages/4_⚠️_Risk_Analysis.py",
        "app/pages/5_📈_Future_Price_Forecast.py",
        "app/pages/6_🔍_Explainable_AI.py",
        "app/pages/7_⚖️_Property_Comparison.py",
        "app/pages/8_🌍_Market_Insights.py"
    ]
    for db_file in dashboard_files:
        assert_exists(db_file, f"Dashboard file {os.path.basename(db_file)}")

    # 5. Summary
    print("\n==================================================")
    print("                 TEST RESULT SUMMARY              ")
    print("==================================================")
    print(f"Total Passed Checklist Items: {passed_tests}")
    print(f"Total Failed Checklist Items: {failed_tests}")
    
    if failed_tests == 0:
        print("\n[SUCCESS] CONGRATULATIONS! The pipeline and dashboard are 100% VALID!")
        print("Run 'streamlit run app/app.py' to launch the interactive portal.")
        print("==================================================")
        return 0
    else:
        print("\n[WARNING] WARNING: There are failing tests. Please review the errors above.")
        print("==================================================")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())
