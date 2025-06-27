import streamlit as st
import pandas as pd
import re
from io import BytesIO

st.title("SKU Extractor from Excel")

# File uploader
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

def extract_skus_from_excel(df):
    all_text = df.astype(str).values.flatten()  # Flatten all cells to a list of strings
    sku_pattern = re.compile(r"\b[A-Z]{2,}[0-9]{2,}[A-Z0-9]*\b")  # SKU-like pattern
    skus = set()

    for text in all_text:
        matches = sku_pattern.findall(text)
        for match in matches:
            # Filter out overly short matches and known false positives
            if len(match) >= 6:
                skus.add(match)

    return sorted(skus)

def to_excel(sku_list):
    df = pd.DataFrame({'SKU': sku_list, 'GE SKU': ''})
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

if uploaded_file:
    df = pd.read_excel(uploaded_file, header=None)
    skus = extract_skus_from_excel(df)
    
    st.success(f"✅ Found {len(skus)} unique SKUs:")
    st.dataframe(pd.DataFrame({'SKU': skus}))

    excel_data = to_excel(skus)
    st.download_button("Download SKUs to Excel", data=excel_data, file_name="sku_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
