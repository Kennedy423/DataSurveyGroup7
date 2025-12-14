import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, mode, pearsonr
from fpdf import FPDF
import io

# Language dictionaries (expanded)
texts = {
    'en': {
        'title': 'Survey Data Analysis Webapp',
        'lang_select': 'Select Language',
        'upload': 'Upload Survey Data (CSV)',
        'no_data': 'Please upload a CSV file.',
        'descriptive': 'Descriptive Analysis',
        'association': 'Association Analysis',
        'select_col': 'Select Column for Analysis',
        'stats': 'Statistics',
        'plot': 'Plot',
        'corr': 'Correlation Analysis',
        'select_vars': 'Select Two Variables',
        'chi2': 'Chi-Square Test',
        'pearson': 'Pearson Correlation',
        'error': 'Error',
        'invalid_file': 'Invalid file format. Please upload a CSV.',
        'no_numeric': 'No numeric columns available.',
        'no_cat': 'No categorical columns available.',
        'results': 'Results',
        'composite': 'Composite Scores',
        'select_cols_composite': 'Select Columns to Sum for Composite Score',
        'composite_name': 'Enter Name for Composite Score (e.g., X_total)',
        'create_composite': 'Create Composite Score',
        'mean': 'Mean',
        'median': 'Median',
        'mode': 'Mode',
        'min': 'Minimum',
        'max': 'Maximum',
        'std': 'Standard Deviation',
        'freq_table': 'Frequency Table',
        'perc_table': 'Percentage Table',
        'hist': 'Histogram',
        'boxplot': 'Boxplot',
        'scatter': 'Scatterplot',
        'only_cat': 'Please select two categorical variables for Chi-Square Test.',
        'auto_corr': 'Automatic Correlation (Numeric)',
        'export_pdf': 'Export PDF Report',
        'group_members': 'Group Members'
    },
    'id': {
        'title': 'Aplikasi Web Analisis Data Survei',
        'lang_select': 'Pilih Bahasa',
        'upload': 'Unggah Data Survei (CSV)',
        'no_data': 'Silakan unggah file CSV.',
        'descriptive': 'Analisis Deskriptif',
        'association': 'Analisis Asosiasi',
        'select_col': 'Pilih Kolom untuk Analisis',
        'stats': 'Statistik',
        'plot': 'Plot',
        'corr': 'Analisis Korelasi',
        'select_vars': 'Pilih Dua Variabel',
        'chi2': 'Uji Chi-Square',
        'pearson': 'Korelasi Pearson',
        'error': 'Kesalahan',
        'invalid_file': 'Format file tidak valid. Silakan unggah CSV.',
        'no_numeric': 'Tidak ada kolom numerik yang tersedia.',
        'no_cat': 'Tidak ada kolom kategorikal yang tersedia.',
        'results': 'Hasil',
        'composite': 'Skor Komposit',
        'select_cols_composite': 'Pilih Kolom untuk Dijumlahkan Skor Komposit',
        'composite_name': 'Masukkan Nama untuk Skor Komposit (mis. X_total)',
        'create_composite': 'Buat Skor Komposit',
        'mean': 'Rata-rata',
        'median': 'Median',
        'mode': 'Modus',
        'min': 'Minimum',
        'max': 'Maksimum',
        'std': 'Deviasi Standar',
        'freq_table': 'Tabel Frekuensi',
        'perc_table': 'Tabel Persentase',
        'hist': 'Histogram',
        'boxplot': 'Boxplot',
        'scatter': 'Scatterplot',
        'only_cat': 'Silakan pilih dua variabel kategorikal untuk Uji Chi-Square.',
        'auto_corr': 'Korelasi Otomatis (Numerik)',
        'export_pdf': 'Ekspor Laporan PDF',
        'group_members': 'Anggota Kelompok'
    }
}

# Group members (edit this list as needed)
group_members = ['Member 1: Ahmad Galan Ali', 'Member 2: Kennedy Ibrahim Ubaldus', 'Member 3: Raffi Ardiansyah Zulin']

def generate_pdf(data, lang, txt):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt['title'], ln=True, align='C')
    pdf.ln(10)
    
    # Add descriptive stats summary
    pdf.cell(200, 10, f"{txt['descriptive']} Summary", ln=True)
    for col in data.columns[:5]:  # Limit to first 5 columns for brevity
        if data[col].dtype in ['int64', 'float64']:
            pdf.cell(200, 10, f"{col}: Mean={data[col].mean():.2f}, Std={data[col].std():.2f}", ln=True)
        else:
            pdf.cell(200, 10, f"{col}: Top Category={data[col].mode()[0]}", ln=True)
    
    # Add association summary
    pdf.ln(10)
    pdf.cell(200, 10, f"{txt['association']} Summary", ln=True)
    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    if len(cat_cols) >= 2:
        contingency = pd.crosstab(data[cat_cols[0]], data[cat_cols[1]])
        pdf.cell(200, 10, f"Chi-Square between {cat_cols[0]} and {cat_cols[1]}: Computed", ln=True)
    
    return pdf.output(dest='S').encode('latin1')

def main():
    # Language selection
    lang = st.sidebar.selectbox('Select Language / Pilih Bahasa', ['en', 'id'])
    txt = texts[lang]
    
    # Group members display
    st.sidebar.header(txt['group_members'])
    for member in group_members:
        st.sidebar.write(member)
    
    st.title(txt['title'])
    
    # File uploader
    uploaded_file = st.file_uploader(txt['upload'], type=['csv'])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.dataframe(data)
        except Exception as e:
            st.error(f"{txt['error']}: {txt['invalid_file']}")
            return
        
        # Composite scores creation
        st.header(txt['composite'])
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_cols = st.multiselect(txt['select_cols_composite'], numeric_cols)
            composite_name = st.text_input(txt['composite_name'])
            if st.button(txt['create_composite']) and selected_cols and composite_name:
                data[composite_name] = data[selected_cols].sum(axis=1)
                st.success(f"Composite score '{composite_name}' created!")
                st.write(data[[composite_name]].head())
                numeric_cols.append(composite_name)
        else:
            st.warning(txt['no_numeric'])
        
        # PDF Export
        if st.button(txt['export_pdf']):
            pdf_bytes = generate_pdf(data, lang, txt)
            st.download_button(label="Download PDF", data=pdf_bytes, file_name="survey_report.pdf", mime="application/pdf")
        
        # Tabs for analysis
        tab1, tab2 = st.tabs([txt['descriptive'], txt['association']])
        
        with tab1:
            st.header(txt['descriptive'])
            
            # Analysis for each survey item (column)
            all_cols = data.columns.tolist()
            selected_col = st.selectbox(txt['select_col'], all_cols)
            
            if selected_col in numeric_cols:
                # Numeric stats
                st.subheader(f"Stats for {selected_col}")
                col_data = data[selected_col].dropna()
                st.write(f"**{txt['mean']}**: {col_data.mean():.2f}")
                st.write(f"**{txt['median']}**: {col_data.median():.2f}")
                st.write(f"**{txt['mode']}**: {mode(col_data, keepdims=False)[0]}")
                st.write(f"**{txt['min']}**: {col_data.min():.2f}")
                st.write(f"**{txt['max']}**: {col_data.max():.2f}")
                st.write(f"**{txt['std']}**: {col_data.std():.2f}")
                
                # Optional plots (added scatterplot)
                plot_type = st.selectbox("Optional Plot", [txt['hist'], txt['boxplot'], txt['scatter']])
                if st.button(txt['plot']):
                    fig, ax = plt.subplots()
                    if plot_type == txt['hist']:
                        sns.histplot(col_data, ax=ax)
                    elif plot_type == txt['boxplot']:
                        sns.boxplot(y=col_data, ax=ax)
                    elif plot_type == txt['scatter']:
                        # Scatterplot against index or another numeric col (simple example)
                        other_num = [c for c in numeric_cols if c != selected_col]
                        if other_num:
                            sns.scatterplot(x=data[other_num[0]], y=col_data, ax=ax)
                        else:
                            st.warning("Need another numeric column for scatterplot.")
                    st.pyplot(fig)
            
            else:
                # Categorical stats
                st.subheader(f"Stats for {selected_col}")
                col_data = data[selected_col].dropna()
                freq = col_data.value_counts()
                perc = col_data.value_counts(normalize=True) * 100
                st.write(f"**{txt['freq_table']}**")
                st.write(freq)
                st.write(f"**{txt['perc_table']}**")
                st.write(perc)
                
                # Optional bar plot
                if st.button(txt['plot']):
                    fig, ax = plt.subplots()
                    freq.plot(kind='bar', ax=ax)
                    st.pyplot(fig)
        
        with tab2:
            st.header(txt['association'])
            
            # Automatic correlation for numeric
            if numeric_cols and len(numeric_cols) >= 2:
                st.subheader(txt['auto_corr'])
                corr_matrix = data[numeric_cols].corr()
                st.write("Correlation Matrix:")
                st.write(corr_matrix)
                fig, ax = plt.subplots()
                sns.heatmap(corr_matrix, annot=True, ax=ax)
                st.pyplot(fig)
            
            # Chi-square for categorical
            cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                st.subheader(txt['chi2'])
                if len(cat_cols) >= 2:
                    var1, var2 = st.selectbox(txt['select_vars'] + ' (X - Categorical)', cat_cols), st.selectbox(txt['select_vars'] + ' (Y - Categorical)', cat_cols)
                    if var1 != var2 and st.button(txt['chi2']):
                        contingency = pd.crosstab(data[var1], data[var2])
                        st.write("Contingency Table (Summary):")
                        st.write(contingency)
                        chi2, p, dof, expected = chi2_contingency(contingency)
                        st.write(f"{txt['results']}: Chi2 = {chi2:.2f}, p-value = {p:.2f}")
                    else:
                        st.info(txt['only_cat'])
                else:
                    st.warning("Need at least two categorical columns.")
            else:
                st.warning(txt['no_cat'])
    else:
        st.info(txt['no_data'])

if __name__ == '__main__':
    main()

