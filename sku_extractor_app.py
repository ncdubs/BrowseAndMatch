import streamlit as st
import pandas as pd
import re
from io import BytesIO

def extract_skus_from_text(text):
    sku_pattern = re.compile(r"\b[A-Z]{2,}[0-9]{2,}[A-Z0-9]*\b")
    matches = sku_pattern.findall(text.upper())
    skus = {sku for sku in matches if len(sku) >= 6}
    return sorted(skus)

def extract_skus_from_excel(df):
    all_text = df.astype(str).values.flatten()
    sku_pattern = re.compile(r"\b[A-Z]{2,}[0-9]{2,}[A-Z0-9]*\b")
    skus = set()
    for text in all_text:
        matches = sku_pattern.findall(text.upper())
        for match in matches:
            if len(match) >= 6:
                skus.add(match)
    return sorted(skus)

st.title("SKU Similarity Finder")

# Step 1: Enter/paste competitor SKUs
st.header("Step 1: Enter Competitor SKUs")
uploaded_file = st.file_uploader("Upload Excel file with SKUs", type=["xlsx", "xls"])
st.markdown("### OR")
pasted_data = st.text_area("Paste competitor SKU data here:")

skus = []

if uploaded_file:
    df = pd.read_excel(uploaded_file, header=None)
    skus = extract_skus_from_excel(df)
elif pasted_data.strip():
    skus = extract_skus_from_text(pasted_data)

if skus:
    st.success(f"✅ Found {len(skus)} unique SKUs:")
    st.dataframe(pd.DataFrame({'SKU': skus}))

# Step 2: Upload appliance catalog
st.header("Step 2: Upload Appliance Catalog (GE + Competitors)")
appliance_file = st.file_uploader("Upload appliance catalog Excel file (with specs)", type=["xlsx", "xls"], key="appliance_upload")
appliance_df = None

if appliance_file is not None:
    appliance_df = pd.read_excel(appliance_file)
    st.success(f"✅ Loaded appliance catalog with {appliance_df.shape[0]} products and {appliance_df.shape[1]} columns.")
    st.dataframe(appliance_df.head(20))

# Step 3: Run similarity matching
if skus and appliance_df is not None:
    if not {'SKU', 'Brand'}.issubset(appliance_df.columns):
        st.error("Appliance catalog must contain 'SKU' and 'Brand' columns.")
    else:
        feature_cols = [col for col in appliance_df.columns if col not in ['SKU', 'Brand']]
        ge_products = appliance_df[appliance_df['Brand'].str.contains("GE", na=False, case=False)].reset_index(drop=True)
        results = []

        for competitor_sku in skus:
            # Find the competitor product row
            competitor_row = appliance_df[appliance_df['SKU'].astype(str).str.upper() == competitor_sku]
            if competitor_row.empty:
                results.append({'Entered SKU': competitor_sku, 'Closest GE SKU': 'Not found'})
                continue

            competitor_features = competitor_row[feature_cols].iloc[0]

            # Compute "similarity" (count matching features) for all GE products
            def feature_similarity(ge_row):
                return sum(
                    (str(competitor_features[col]).strip().lower() == str(ge_row[col]).strip().lower())
                    for col in feature_cols if pd.notnull(competitor_features[col]) and pd.notnull(ge_row[col])
                )

            ge_products['SimilarityScore'] = ge_products.apply(feature_similarity, axis=1)
            best_match_row = ge_products.sort_values('SimilarityScore', ascending=False).iloc[0]

            if best_match_row['SimilarityScore'] == 0:
                closest_ge = 'Not found'
            else:
                closest_ge = best_match_row['SKU']

            results.append({'Entered SKU': competitor_sku, 'Closest GE SKU': closest_ge})

        results_df = pd.DataFrame(results)
        st.subheader("Matching Results")
        st.dataframe(results_df)

