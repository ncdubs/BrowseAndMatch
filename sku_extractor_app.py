import streamlit as st
import pandas as pd
import re
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Bulk SKU Similarity Finder (TF-IDF Cosine)")

# STEP 1: Paste or upload list of SKUs
st.header("Step 1: Enter Competitor SKUs")
uploaded_file = st.file_uploader("Upload Excel file with SKUs", type=["xlsx", "xls"])
pasted_data = st.text_area("Paste competitor SKU data here:")

def extract_skus_from_excel(df):
    all_text = df.astype(str).values.flatten()
    sku_pattern = re.compile(r"[A-Z]{2,}[0-9]{2,}[A-Z0-9]*")
    skus = []
    seen = set()
    for text in all_text:
        matches = sku_pattern.findall(text)
        for match in matches:
            if len(match) >= 6 and match not in seen:
                skus.append(match)
                seen.add(match)
    return skus


def extract_skus_from_text(text):
    sku_pattern = re.compile(r"[A-Z]{2,}[0-9]{2,}[A-Z0-9]*")
    skus = []
    seen = set()
    for line in text.upper().splitlines():
        matches = sku_pattern.findall(line)
        for sku in matches:
            if len(sku) >= 6 and sku not in seen:
                skus.append(sku)
                seen.add(sku)
    return skus


def to_excel(sku_list):
    df = pd.DataFrame({'SKU': sku_list, 'GE SKU': ''})
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

skus = []
if uploaded_file:
    df_upload = pd.read_excel(uploaded_file, header=None)
    skus = extract_skus_from_excel(df_upload)
elif pasted_data.strip():
    skus = extract_skus_from_text(pasted_data)

if skus:
    st.success(f"✅ Found {len(skus)} unique SKUs:")
    st.dataframe(pd.DataFrame({'SKU': skus}))
    excel_data = to_excel(skus)
    st.download_button("Download SKUs to Excel", data=excel_data, file_name="sku_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# STEP 2: Upload catalog
st.header("Step 2: Upload Appliance Catalog (Tall Format)")
appliance_file = st.file_uploader("Upload catalog Excel (tall, features as columns)", type=["xlsx", "xls"], key="appliance_upload")
if not appliance_file:
    st.stop()

df = pd.read_excel(appliance_file)
if 'SKU' not in df.columns:
    st.error("No 'SKU' column found!")
    st.stop()
if 'Brand' not in df.columns:
    st.error("No 'Brand' column found!")
    st.stop()
if 'Model Status' not in df.columns:
    st.error("No 'Model Status' column found!")
    st.stop()
if 'Configuration' not in df.columns:
    st.error("No 'Configuration' column found!")
    st.stop()

# Use all features except for core metadata columns for matching
discard_cols = ['SKU', 'Brand', 'Model Status', 'combined_specs']
all_features = [col for col in df.columns if col not in discard_cols]

# Create combined specs string (default weighting = 1)
df['combined_specs'] = df[all_features].astype(str).agg(' '.join, axis=1)

# Build TF-IDF matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['combined_specs'])

# Filter for active models only, and GE only for matching
ge_mask = (df['Brand'].str.lower() == 'ge') & (df['Model Status'].str.lower() == 'active model')
ge_df = df[ge_mask].reset_index(drop=True)
ge_tfidf = vectorizer.transform(ge_df['combined_specs'])

# BULK MATCH: For each entered competitor SKU, find the best GE match
results = []
for sku in skus:
    comp_row = df[df['SKU'] == sku]
    if comp_row.empty:
        results.append({
            'Entered SKU': sku,
            'Closest GE SKU': 'Not found (SKU not in catalog)',
            'Similarity Score': 0
        })
        continue
    # Check configuration for strict match
    comp_config = comp_row.iloc[0]['Configuration']
    config_mask = ge_df['Configuration'].str.lower() == str(comp_config).lower()
    filtered_ge = ge_df[config_mask]
    filtered_ge_tfidf = ge_tfidf[config_mask.values]

    if filtered_ge.empty:
        results.append({
            'Entered SKU': sku,
            'Closest GE SKU': 'Not found (no GE match for config)',
            'Similarity Score': 0
        })
        continue

    comp_tfidf = vectorizer.transform([comp_row['combined_specs'].values[0]])
    sims = cosine_similarity(comp_tfidf, filtered_ge_tfidf)[0]
    if sims.max() == 0:
        results.append({
            'Entered SKU': sku,
            'Closest GE SKU': 'Not found (no similar GE model)',
            'Similarity Score': 0
        })
    else:
        best_idx = sims.argmax()
        best_sku = filtered_ge.iloc[best_idx]['SKU']
        best_score = round(sims[best_idx], 3)
        results.append({
            'Entered SKU': sku,
            'Closest GE SKU': best_sku,
            'Similarity Score': best_score
        })

results_df = pd.DataFrame(results)

st.subheader("Matching Results")
st.dataframe(results_df)
