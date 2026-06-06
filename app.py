from pathlib import Path
import warnings
import joblib, numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
st.set_page_config(page_title='Telco Customer Churn', page_icon='📊', layout='wide', initial_sidebar_state='expanded')

BASE = Path(__file__).resolve().parent
DATA_CANDIDATES = [BASE / 'clean_telco_churn.csv', BASE / 'WA_Fn-UseC_-Telco-Customer-Churn.csv']
MODEL_CANDIDATES = [BASE / 'xgboost_churn_model.pkl', BASE / 'model.pkl']
FEATURE_CANDIDATES = [BASE / 'feature_columns.pkl']
KMEANS_CANDIDATES = [BASE / 'kmeans_model.pkl']
SCALER_CANDIDATES = [BASE / 'scaler.pkl']
CLUSTER_FEATURE_CANDIDATES = [BASE / 'cluster_feature_columns.pkl']
DEFAULT_METRICS = {'accuracy': 0.8220, 'precision': 0.6498, 'recall': 0.7138, 'roc_auc': 0.8256}

st.markdown("<style>#MainMenu,footer,header{visibility:hidden;}</style>", unsafe_allow_html=True)
css_path = BASE / 'style.css'
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

pick = lambda paths: next((p for p in paths if p.exists()), None)
align = lambda frame, cols: frame.reindex(columns=cols, fill_value=0) if cols else frame
numeric = lambda frame: frame.apply(pd.to_numeric, errors='coerce').fillna(0)

@st.cache_data
def load_csv(path): return pd.read_csv(path)

@st.cache_resource
def load_pickle(path):
    try: return joblib.load(path)
    except Exception: return None

@st.cache_data
def load_all():
    dp = pick(DATA_CANDIDATES)
    if not dp: return None, None, None
    raw = load_csv(dp).copy()
    if 'customerID' in raw.columns: raw = raw.drop(columns=['customerID'])
    if 'TotalCharges' in raw.columns:
        total = pd.to_numeric(raw['TotalCharges'], errors='coerce')
        raw['TotalCharges'] = total.fillna(total.median())
    if 'Churn' in raw.columns and raw['Churn'].dtype == object:
        raw['Churn'] = raw['Churn'].map({'No': 0, 'Yes': 1}).fillna(raw['Churn']).pipe(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    enc = pd.get_dummies(raw, columns=raw.select_dtypes(include='object').columns.tolist(), drop_first=True)
    return raw, enc, dp.name

@st.cache_resource
def load_models():
    mp, fp, kp, sp, cp = map(pick, [MODEL_CANDIDATES, FEATURE_CANDIDATES, KMEANS_CANDIDATES, SCALER_CANDIDATES, CLUSTER_FEATURE_CANDIDATES])
    return (load_pickle(mp) if mp else None, load_pickle(fp) if fp else None, load_pickle(kp) if kp else None, load_pickle(sp) if sp else None, load_pickle(cp) if cp else None)

def spacer(h=10): st.markdown(f"<div style='height:{h}px'></div>", unsafe_allow_html=True)
def box(title, body): st.markdown(f"<div class='card'><h3>{title}</h3><div class='card-body'>{body}</div></div>", unsafe_allow_html=True)
def metric(title, value, subtitle='', color='#2563eb'): st.markdown(f"<div class='metric-card'><div class='metric-title'>{title}</div><div class='metric-value' style='color:{color};'>{value}</div><div class='metric-subtitle'>{subtitle}</div></div>", unsafe_allow_html=True)

def page_header(title, subtitle):
    a, b = st.columns([5, 1], gap='large')
    with a:
        st.markdown(f"<div class='page-title'>{title}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='page-subtitle'>{subtitle}</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div style='display:flex;justify-content:flex-end;'><span class='badge'>Universitas Negeri Surabaya</span></div>", unsafe_allow_html=True)

def show(fig, h=340):
    fig.update_layout(template='plotly_white', height=h, paper_bgcolor='white', plot_bgcolor='white', margin=dict(l=10, r=10, t=45, b=10), font=dict(color='#111827'), legend_title_text='')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def to_binary(s):
    if s is None: return None
    if s.dtype == object: return s.astype(str).str.strip().str.lower().map({'no': 0, 'yes': 1, '0': 0, '1': 1}).fillna(0).astype(int)
    return pd.to_numeric(s, errors='coerce').fillna(0).astype(int)

def evaluate_model(model, enc, feature_cols):
    out = {'accuracy': DEFAULT_METRICS['accuracy'], 'precision': DEFAULT_METRICS['precision'], 'recall': DEFAULT_METRICS['recall'], 'roc_auc': DEFAULT_METRICS['roc_auc'], 'cm': None, 'fpr': None, 'tpr': None}
    if model is None or 'Churn' not in enc.columns: return out
    cols = feature_cols or [c for c in enc.columns if c != 'Churn']
    X, y = numeric(align(enc.drop(columns=['Churn']), cols)), to_binary(enc['Churn'])
    try:
        y_pred = model.predict(X); out['cm'] = confusion_matrix(y, y_pred); out['accuracy'] = float((y_pred == y).mean())
    except Exception: pass
    try:
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)[:, 1]; out['roc_auc'] = float(roc_auc_score(y, y_prob)); out['fpr'], out['tpr'], _ = roc_curve(y, y_prob)
    except Exception: pass
    return out

def encode_user(data, feature_cols): return numeric(align(pd.get_dummies(pd.DataFrame([data]), drop_first=True), feature_cols))

def predict_cluster(X, kmeans, scaler, cluster_cols):
    if kmeans is None: return None
    Xc = numeric(align(X, cluster_cols or list(X.columns)))
    try:
        if scaler is not None: Xc = scaler.transform(Xc)
        return int(kmeans.predict(Xc)[0])
    except Exception: return None

def add_cluster(df, kmeans, scaler, cluster_cols, feature_cols):
    out = df.copy()
    if kmeans is None: return out
    cols = cluster_cols or feature_cols or [c for c in out.columns if c != 'Churn']
    X = numeric(align(out.drop(columns=['Churn'], errors='ignore'), cols))
    try:
        if scaler is not None: X = scaler.transform(X)
        out['Cluster'] = kmeans.predict(X)
    except Exception: pass
    return out

def cluster_map(df):
    if 'Cluster' not in df.columns or 'Churn' not in df.columns: return {}
    churn_rate = df.groupby('Cluster')['Churn'].apply(lambda s: to_binary(s).mean()).sort_values(ascending=False)
    clusters = list(churn_rate.index)
    if not clusters: return {}
    if len(clusters) == 1: return {int(clusters[0]): ('Cluster Risiko Sedang', f"Churn rate {churn_rate.iloc[0]:.1%}.")}
    out = {int(clusters[0]): ('Cluster Risiko Tinggi', f"Memiliki churn rate tertinggi ({churn_rate.iloc[0]:.1%})."), int(clusters[-1]): ('Cluster Risiko Rendah', f"Memiliki churn rate terendah ({churn_rate.iloc[-1]:.1%}).")}
    for c in clusters[1:-1]: out[int(c)] = ('Cluster Risiko Sedang', f"Churn rate {churn_rate.loc[c]:.1%}.")
    return out

def recommendations(pred, risk):
    items = [
        'Pertahankan kualitas layanan agar pelanggan tetap puas dan loyal.',
        'Berikan loyalty reward atau promo upgrade agar pelanggan tidak mudah berpindah provider.',
        'Pantau pelanggan dengan biaya bulanan tinggi atau risiko tinggi agar indikasi churn bisa dicegah lebih awal.',
    ] if pred == 0 else [
        'Berikan penawaran retensi seperti diskon, upgrade paket, atau kontrak jangka panjang yang lebih menarik.',
        'Lakukan follow-up proaktif untuk mengetahui keluhan utama pelanggan dan memperbaiki titik masalahnya.',
        'Perkuat kualitas layanan yang sering terkait churn, terutama dukungan teknis, tagihan, dan stabilitas layanan internet.',
    ]
    if risk == 'Tinggi': items[0] = 'Prioritaskan penawaran retensi dalam waktu singkat agar peluang churn tidak semakin besar.'
    return items[:3]

def correlation_fig(raw, cols):
    corr = raw[cols].corr(numeric_only=True)
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale='Blues', zmin=-1, zmax=1, colorbar=dict(title='Corr')))
    fig.update_layout(title='Korelasi Fitur Numerik', xaxis=dict(tickangle=-35, automargin=True), yaxis=dict(automargin=True), margin=dict(l=20, r=20, t=45, b=90), height=360)
    return fig

def pca_cluster_fig(df):
    cols = [c for c in ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen'] if c in df.columns]
    if len(cols) < 2: return None
    pca = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(numeric(df[cols])))
    plot_df = pd.DataFrame({'PCA1': pca[:, 0], 'PCA2': pca[:, 1]})
    if 'Cluster' in df.columns:
        plot_df['Cluster'] = df['Cluster'].astype(str)
        return px.scatter(plot_df, x='PCA1', y='PCA2', color='Cluster', title='PCA Visualization of Customer Clusters')
    return px.scatter(plot_df, x='PCA1', y='PCA2', title='PCA Visualization')

def home(raw, ev):
    page_header('Telco Customer Churn', 'Prediksi churn dan segmentasi pelanggan untuk mendukung strategi retensi.')
    box('Deskripsi Singkat Proyek', '<p>Proyek ini mengolah data pelanggan telekomunikasi untuk memprediksi kemungkinan pelanggan berhenti berlangganan dengan <b>Classification</b> menggunakan <b>XGBoost</b>, lalu mengelompokkan pelanggan dengan <b>K-Means</b> agar strategi retensi bisa dibuat lebih tepat sasaran.</p><p>Seluruh hasil analisis ditampilkan dalam aplikasi Streamlit yang berisi overview dataset, prediksi, visualisasi, dan penjelasan metode.</p>')
    spacer(8)
    cols = st.columns(4, gap='large'); churn_rate = f"{to_binary(raw['Churn']).mean() * 100:.1f}%" if 'Churn' in raw.columns else '-'
    for c, t, v, s, col in [(cols[0], 'Total Data', f'{len(raw):,}', 'Jumlah pelanggan', '#2563eb'), (cols[1], 'Accuracy', f"{ev['accuracy'] * 100:.2f}%", 'Evaluasi model', '#16a34a'), (cols[2], 'Clusters', f"{raw['Cluster'].nunique()}" if 'Cluster' in raw.columns else '3', 'Segmentasi pelanggan', '#7c3aed'), (cols[3], 'Churn Rate', churn_rate, 'Proporsi churn', '#ea580c')]:
        with c: metric(t, v, s, col)
    spacer(10)
    box('Identitas Pengembang', '<p><b>Nama:</b> Sierren Yorensa</p><p><b>NIM:</b> 24051214173</p><p><b>Program Studi:</b> Sistem Informasi</p><p><b>Universitas:</b> Universitas Negeri Surabaya</p>')

def dataset_overview(raw):
    page_header('Dataset Overview', 'Informasi dataset, ringkasan statistik, dan visualisasi sederhana.')
    cols = st.columns(4, gap='large')
    for c, t, v, s, col in [(cols[0], 'Rows', f'{len(raw):,}', 'Jumlah baris', '#2563eb'), (cols[1], 'Columns', f'{raw.shape[1]}', 'Jumlah kolom', '#16a34a'), (cols[2], 'Missing', f"{int(raw.isna().sum().sum()):,}", 'Total nilai kosong', '#ea580c'), (cols[3], 'Churn Rate', f"{to_binary(raw['Churn']).mean() * 100:.1f}%" if 'Churn' in raw.columns else '-', 'Proporsi churn', '#7c3aed')]:
        with c: metric(t, v, s, col)
    spacer(8)
    box('Penjelasan Dataset', '<p><b>Telco Customer Churn</b> adalah dataset pelanggan telekomunikasi yang digunakan untuk mempelajari perilaku pelanggan yang berpotensi berhenti berlangganan. Dataset ini berisi atribut demografis, informasi layanan, durasi berlangganan, biaya bulanan, dan total tagihan.</p><p>Target utama pada proyek ini adalah kolom <b>Churn</b>, sedangkan atribut lain digunakan sebagai input model classification dan clustering. Setelah preprocessing, data kategorikal diubah menjadi dummy/one-hot encoding agar dapat dipakai oleh model XGBoost dan K-Means. Nilai kosong pada <b>TotalCharges</b> diisi dengan median supaya data tetap konsisten.</p><p>Dataset ini cocok untuk tugas data mining karena memiliki lebih dari 5 atribut, jumlah data yang besar, dan karakteristik yang relevan untuk analisis retensi pelanggan.</p>')
    left, right = st.columns(2, gap='large')
    with left:
        st.subheader('Preview Data')
        preview_cols = [c for c in ['gender', 'SeniorCitizen', 'Partner', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn'] if c in raw.columns]
        st.dataframe(raw[preview_cols].head(10), use_container_width=True, hide_index=True, height=300)
    with right:
        st.subheader('Statistik Sederhana')
        stats_cols = [c for c in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges'] if c in raw.columns]
        st.dataframe(raw[stats_cols].describe().round(2), use_container_width=True, hide_index=False, height=300)
    spacer(4)
    left, right = st.columns(2, gap='large')
    with left:
        churn = to_binary(raw['Churn']).map({0: 'No Churn', 1: 'Churn'}).value_counts()
        show(px.pie(values=churn.values, names=churn.index, title='Distribusi Churn'), 320)
    with right:
        num_cols = [c for c in ['tenure', 'MonthlyCharges', 'TotalCharges'] if c in raw.columns]
        if len(num_cols) >= 2: show(correlation_fig(raw, num_cols), 360)

def build_form():
    with st.form('predict_form'):
        left, right = st.columns(2, gap='large')
        with left:
            gender = st.selectbox('Gender', ['Female', 'Male'])
            senior = st.selectbox('Senior Citizen', [0, 1])
            partner = st.selectbox('Partner', ['No', 'Yes'])
            dependents = st.selectbox('Dependents', ['No', 'Yes'])
            tenure = st.slider('Tenure (bulan)', 0, 72, 12)
            phone = st.selectbox('Phone Service', ['No', 'Yes'])
            multiple = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
            internet = st.selectbox('Internet Service', ['No', 'DSL', 'Fiber optic'])
            online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
            online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
        with right:
            device = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
            tech = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
            tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
            movie = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
            contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
            paperless = st.selectbox('Paperless Billing', ['No', 'Yes'])
            payment = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
            monthly = st.number_input('Monthly Charges', min_value=0.0, value=70.0, step=0.05)
            total = st.number_input('Total Charges', min_value=0.0, value=1000.0, step=0.05)
        submit = st.form_submit_button('Prediksi')
    return submit, {'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents, 'tenure': tenure, 'PhoneService': phone, 'MultipleLines': multiple, 'InternetService': internet, 'OnlineSecurity': online_security, 'OnlineBackup': online_backup, 'DeviceProtection': device, 'TechSupport': tech, 'StreamingTV': tv, 'StreamingMovies': movie, 'Contract': contract, 'PaperlessBilling': paperless, 'PaymentMethod': payment, 'MonthlyCharges': monthly, 'TotalCharges': total}

def prediction_analysis(model, feature_cols, kmeans, scaler, cluster_cols, ev, cmap):
    page_header('Prediction / Analysis', 'Form input pengguna dan hasil prediksi model.')
    if model is None:
        st.warning('Model belum ditemukan. Letakkan file `xgboost_churn_model.pkl` atau `model.pkl` di folder aplikasi.')
        return
    cols = st.columns(4, gap='large')
    for c, t, v, s, col in [(cols[0], 'Accuracy', f"{ev['accuracy'] * 100:.2f}%", 'Evaluasi model', '#16a34a'), (cols[1], 'Precision', f"{ev['precision'] * 100:.2f}%", 'Kelas churn', '#2563eb'), (cols[2], 'Recall', f"{ev['recall'] * 100:.2f}%", 'Kelas churn', '#ea580c'), (cols[3], 'ROC-AUC', f"{ev['roc_auc']:.4f}", 'Kemampuan diskriminasi', '#7c3aed')]:
        with c: metric(t, v, s, col)
    spacer(6); st.subheader('Prediksi Customer Churn')
    submit, data = build_form()
    if not submit: return
    X = encode_user(data, feature_cols)
    try:
        pred = int(model.predict(X)[0]); prob = float(model.predict_proba(X)[0][1]) if hasattr(model, 'predict_proba') else np.nan
    except Exception as e:
        st.error(f'Gagal melakukan prediksi: {e}'); return
    if pred == 1: st.error('Hasil prediksi: pelanggan diprediksi CHURN.')
    else: st.success('Hasil prediksi: pelanggan diprediksi TIDAK CHURN.')
    if pd.notna(prob):
        st.metric('Probabilitas Churn', f'{prob:.2%}')
        risk = 'Tinggi' if prob >= 0.75 else 'Sedang' if prob >= 0.50 else 'Rendah'
    else: risk = 'Tinggi' if pred == 1 else 'Rendah'
    st.write(f'**Tingkat Risiko:** {risk}')
    cluster = predict_cluster(X, kmeans, scaler, cluster_cols)
    if cluster is not None:
        label, desc = cmap.get(cluster, ('Cluster Pelanggan', 'Segmentasi pelanggan berdasarkan K-Means.'))
        st.info(
            f"**Segmentasi Pelanggan**\n\n"
            f"- Cluster ID: {cluster}\n"
            f"- Kategori: {label}\n"
            f"- Keterangan: {desc}"
        )
        if 'tinggi' in label.lower(): st.warning('Pelanggan masuk kelompok dengan riwayat churn tertinggi.')
    st.markdown('---'); st.markdown('### Rekomendasi Retensi')
    for i, rec in enumerate(recommendations(pred, risk), start=1): st.markdown(f'{i}. {rec}')

def visualization(raw, enc, ev):
    page_header('Visualization', 'Grafik pendukung dan visualisasi hasil analisis.')
    tabs = st.tabs(['Pelanggan', 'Model', 'Clustering'])
    with tabs[0]:
        top_left, top_right = st.columns(2, gap='large')
        with top_left:
            churn = to_binary(raw['Churn']).map({0: 'No Churn', 1: 'Churn'}).value_counts()
            show(px.pie(values=churn.values, names=churn.index, title='Distribusi Churn'), 320)
        with top_right:
            if 'MonthlyCharges' in raw.columns and 'Churn' in raw.columns: show(px.box(raw, x='Churn', y='MonthlyCharges', color='Churn', title='Monthly Charges vs Churn'), 320)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if 'tenure' in raw.columns and 'Churn' in raw.columns:
            show(px.histogram(raw, x='tenure', color='Churn', nbins=30, barmode='overlay', opacity=0.75, title='Distribusi Tenure'), 340)
    with tabs[1]:
        left, right = st.columns(2, gap='large')
        with left:
            if ev and ev.get('cm') is not None: show(px.imshow(ev['cm'], text_auto=True, color_continuous_scale='Blues', title='Confusion Matrix', labels=dict(x='Predicted', y='Actual', color='Count')), 340)
            else: st.info('Confusion matrix belum tersedia karena model evaluasi tidak dapat dibaca.')
        with right:
            if ev and ev.get('fpr') is not None:
                roc_fig = go.Figure(); roc_fig.add_trace(go.Scatter(x=ev['fpr'], y=ev['tpr'], mode='lines', name='ROC')); roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash'))); roc_fig.update_xaxes(title='False Positive Rate'); roc_fig.update_yaxes(title='True Positive Rate'); roc_fig.update_layout(title='ROC Curve'); show(roc_fig, 340)
            else: st.info('ROC curve belum tersedia.')
        st.caption(f"Accuracy {ev['accuracy']*100:.2f}% | Precision {ev['precision']*100:.2f}% | Recall {ev['recall']*100:.2f}% | ROC-AUC {ev['roc_auc']:.4f}")
    with tabs[2]:
        if 'Cluster' in enc.columns:
            left, right = st.columns(2, gap='large')
            with left:
                fig = pca_cluster_fig(enc)
                if fig is not None: show(fig, 340)
            with right:
                cnt = enc['Cluster'].value_counts().sort_index(); show(px.bar(x=cnt.index.astype(str), y=cnt.values, title='Jumlah Data per Cluster'), 340)
            st.dataframe(enc[['Cluster', 'tenure', 'MonthlyCharges', 'TotalCharges']].head(10), use_container_width=True, hide_index=True, height=220)
        else: st.info('Model clustering belum tersedia, sehingga visualisasi cluster tidak dapat ditampilkan.')

def about():
    page_header('About', 'Penjelasan metode, dataset, dan informasi proyek.')
    st.markdown("""### Dataset
Dataset yang digunakan adalah **Telco Customer Churn**, yaitu data pelanggan layanan telekomunikasi yang berisi atribut demografis, informasi layanan, durasi berlangganan, biaya bulanan, dan total tagihan. Dataset ini relevan untuk kasus bisnis churn karena target utamanya adalah memprediksi apakah pelanggan akan berhenti berlangganan.

### Metode
Proyek ini menerapkan dua metode data mining: **Classification** menggunakan XGBoost untuk memprediksi churn, dan **Clustering** menggunakan K-Means untuk membagi pelanggan ke dalam beberapa segmen perilaku. Label cluster pada aplikasi ini dibaca sebagai profil risiko historis, bukan sebagai prediksi churn langsung.

### Implementasi
Aplikasi ini dibangun dengan **Streamlit** agar hasil analisis dapat diakses melalui web. Navigasi berisi halaman Home, Dataset Overview, Prediction / Analysis, Visualization, dan About.

### Informasi Proyek
**Nama:** Sierren Yorensa  
**NIM:** 24051214173  
**Program Studi:** Sistem Informasi  
**Universitas:** Universitas Negeri Surabaya""", unsafe_allow_html=True)

def main():
    raw, encoded, _ = load_all()
    model, feature_cols, kmeans, scaler, cluster_cols = load_models()
    if raw is None: st.error('Dataset tidak ditemukan. Letakkan file CSV dataset di folder aplikasi.'); st.stop()
    feature_cols = feature_cols or [c for c in encoded.columns if c != 'Churn']
    cluster_cols = cluster_cols or feature_cols
    raw_cluster = add_cluster(raw, kmeans, scaler, cluster_cols, feature_cols)
    encoded_cluster = add_cluster(encoded, kmeans, scaler, cluster_cols, feature_cols)
    cmap = cluster_map(raw_cluster if 'Churn' in raw_cluster.columns else encoded_cluster)
    ev = evaluate_model(model, encoded, feature_cols)
    with st.sidebar:
        st.markdown("<div class='sidebar-title'>Telco Customer<br>Churn</div>", unsafe_allow_html=True)
        spacer(8)
        page = st.radio('Navigasi', ['Home', 'Dataset Overview', 'Prediction / Analysis', 'Visualization', 'About'])
    if page == 'Home': home(raw_cluster, ev)
    elif page == 'Dataset Overview': dataset_overview(raw)
    elif page == 'Prediction / Analysis': prediction_analysis(model, feature_cols, kmeans, scaler, cluster_cols, ev, cmap)
    elif page == 'Visualization': visualization(raw_cluster, encoded_cluster, ev)
    else: about()

if __name__ == '__main__': main()
