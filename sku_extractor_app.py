import streamlit as st
import pandas as pd
import re
from io import BytesIO

st.title("SKU Extractor & Appliance Catalog Uploader")

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

def to_excel(sku_list):
    df = pd.DataFrame({'SKU': sku_list, 'GE SKU': ''})
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

# --- Step 1: Input SKUs ---
st.header("Step 1: Enter competitor SKUs")
uploaded_file = st.file_uploader("Upload Excel file with SKUs", type=["xlsx", "xls"])
st.markdown("### OR")
pasted_data = st.text_area("Paste competitor SKU data here (from Excel, CSV, etc):")

skus = []

if uploaded_file:
    df = pd.read_excel(uploaded_file, header=None)
    skus = extract_skus_from_excel(df)
elif pasted_data.strip():
    skus = extract_skus_from_text(pasted_data)

if skus:
    st.success(f"✅ Found {len(skus)} unique SKUs:")
    st.dataframe(pd.DataFrame({'SKU': skus}))
    excel_data = to_excel(skus)
    st.download_button("Download SKUs to Excel", data=excel_data, file_name="sku_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Upload an Excel file or paste your data above to extract SKUs.")

# --- Step 2: Upload Appliance Folder ---
st.header("Step 2: Upload Appliance Catalog (GE or other brands)")

appliance_file = st.file_uploader("Upload appliance catalog Excel file (with specs)", type=["xlsx", "xls"], key="appliance_upload")
appliance_df = None

if appliance_file is not None:
    appliance_df = pd.read_excel(appliance_file)
    st.success(f"✅ Loaded appliance catalog with {appliance_df.shape[0]} products and {appliance_df.shape[1]} columns.")
    st.dataframe(appliance_df.head(20))

# --- Ready for Next Steps (Matching) ---
if skus and appliance_df is not None:
    st.info("✅ Both SKU list and appliance catalog loaded! Ready for matching logic in next step.")
