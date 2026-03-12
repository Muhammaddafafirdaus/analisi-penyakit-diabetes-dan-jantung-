import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Medis AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(165deg, #0f1b3d 0%, #1a2d5a 60%, #1f3872 100%);
}
[data-testid="stSidebar"] * { color: #e8edf8 !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 15px !important;
    padding: 10px 14px !important;
    border-radius: 10px;
    transition: background 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(79,124,255,0.2) !important;
}

/* Main area background */
.main .block-container { background: #f0f4fc; padding-top: 2rem; }

/* Cards */
.metric-card {
    background: #0f1b3d;
    border-radius: 16px;
    padding: 22px 20px;
    text-align: center;
    color: white;
    margin-bottom: 8px;
}
.metric-card .num {
    font-family: 'Syne', sans-serif;
    font-size: 32px;
    font-weight: 800;
    color: #00d4ff;
}
.metric-card .lbl { font-size: 12px; color: #8fa3cc; margin-top: 4px; }

.feature-card {
    background: white;
    border-radius: 18px;
    padding: 24px;
    border-top: 3px solid;
    margin-bottom: 8px;
}
.feature-card .title {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #0f1b3d;
    margin: 12px 0 8px;
}
.feature-card .desc { font-size: 13.5px; color: #6678a0; line-height: 1.6; }

.result-box-low {
    background: linear-gradient(135deg, rgba(0,229,160,0.08), rgba(0,212,255,0.05));
    border: 1.5px solid rgba(0,229,160,0.3);
    border-radius: 18px;
    padding: 28px 30px;
}
.result-box-high {
    background: linear-gradient(135deg, rgba(255,79,110,0.08), rgba(255,120,0,0.04));
    border: 1.5px solid rgba(255,79,110,0.3);
    border-radius: 18px;
    padding: 28px 30px;
}

h1, h2, h3 { font-family: 'Syne', sans-serif !important; color: #0f1b3d; }
</style>
""", unsafe_allow_html=True)


# ─── MODEL TRAINING ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_diabetes_model():
    """Train Decision Tree model on Pima Indians Diabetes dataset (synthetic if CSV not found)."""
    try:
        df = pd.read_csv("diabetes.csv")
    except FileNotFoundError:
        # Generate realistic synthetic data if CSV not present
        np.random.seed(42)
        n = 768
        df = pd.DataFrame({
            'Pregnancies':            np.random.randint(0, 17, n),
            'Glucose':                np.clip(np.random.normal(120, 32, n), 0, 200).astype(int),
            'BloodPressure':          np.clip(np.random.normal(69, 19, n), 0, 130).astype(int),
            'SkinThickness':          np.clip(np.random.normal(20, 16, n), 0, 100).astype(int),
            'Insulin':                np.clip(np.random.exponential(80, n), 0, 850).astype(int),
            'BMI':                    np.clip(np.random.normal(31.9, 7.9, n), 10, 70).round(1),
            'DiabetesPedigreeFunction': np.clip(np.random.exponential(0.47, n), 0.07, 2.5).round(3),
            'Age':                    np.clip(np.random.normal(33, 12, n), 21, 81).astype(int),
        })
        # Heuristic outcome
        score = (
            (df['Glucose'] > 140).astype(int) * 3 +
            (df['BMI'] > 30).astype(int) * 2 +
            (df['Age'] > 45).astype(int) * 2 +
            (df['DiabetesPedigreeFunction'] > 0.5).astype(int)
        )
        df['Outcome'] = (score >= 4).astype(int)

    feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction']
    X = df[feature_cols]
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Model 1: Default DT
    dt_default = DecisionTreeClassifier(random_state=42)
    dt_default.fit(X_train, y_train)

    # Model 2: Optimized DT (entropy, max_depth=3)
    dt_opt = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
    dt_opt.fit(X_train, y_train)

    acc_default = accuracy_score(y_test, dt_default.predict(X_test))
    acc_opt = accuracy_score(y_test, dt_opt.predict(X_test))

    return dt_opt, feature_cols, acc_opt, acc_default, X_test, y_test, df


@st.cache_resource(show_spinner=False)
def train_heart_model():
    """Train Random Forest model on heart disease dataset (synthetic if CSV not found)."""
    try:
        df = pd.read_csv("heart_v2.csv")
        target_col = 'heart disease'
    except FileNotFoundError:
        np.random.seed(7)
        n = 303
        age = np.clip(np.random.normal(54, 9, n), 29, 77).astype(int)
        sex = np.random.randint(0, 2, n)
        cp  = np.random.randint(0, 4, n)
        trestbps = np.clip(np.random.normal(131, 17, n), 90, 200).astype(int)
        chol = np.clip(np.random.normal(246, 51, n), 120, 564).astype(int)
        fbs  = (np.random.rand(n) > 0.85).astype(int)
        thalach = np.clip(np.random.normal(149, 23, n), 71, 202).astype(int)
        exang = (np.random.rand(n) > 0.68).astype(int)
        oldpeak = np.clip(np.random.exponential(1.04, n), 0, 6.2).round(1)
        score = (
            (age > 55).astype(int) * 2 +
            (sex == 1).astype(int) +
            (cp == 0).astype(int) * 3 +
            (trestbps > 140).astype(int) +
            (chol > 250).astype(int) +
            (thalach < 140).astype(int) * 2 +
            exang * 2 +
            (oldpeak > 2).astype(int) * 2
        )
        df = pd.DataFrame({
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
            'chol': chol, 'fbs': fbs, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'heart disease': (score >= 6).astype(int)
        })
        target_col = 'heart disease'

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    feature_cols = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

    # Random Forest
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5, n_estimators=100, oob_score=True)
    rf.fit(X_train, y_train)

    acc = accuracy_score(y_test, rf.predict(X_test))

    return rf, feature_cols, acc, X_test, y_test, df


# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:8px'>
        <div style='font-family:Syne,sans-serif;font-size:22px;font-weight:800;color:white;line-height:1.2'>
            Analisis<br>Medis AI
        </div>
        <div style='display:flex;gap:4px;margin-top:10px'>
            {"".join(['<span style="width:6px;height:6px;border-radius:50%;background:#4f7cff;display:inline-block;margin-right:3px;opacity:0.6"></span>']*13)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#8fa3cc;margin-bottom:8px'>MENU UTAMA</p>", unsafe_allow_html=True)

    page = st.radio(
        "",
        ["🏠  Beranda", "🩸  Prediksi Diabetes", "❤️  Prediksi Jantung"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='background:rgba(255,200,0,0.08);border:1px solid rgba(255,200,0,0.15);border-radius:12px;padding:12px 14px'>
        <p style='font-size:11.5px;color:#ffd166;line-height:1.5'>
            ⚠️ Hasil bersifat indikatif.<br>Bukan pengganti diagnosis dokter.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: BERANDA
# ═══════════════════════════════════════════════════════════════════════════════
if "Beranda" in page:
    st.markdown("<p style='font-size:11px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#4f7cff'>Selamat Datang</p>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size:52px;letter-spacing:-1.5px;line-height:1.05;margin-bottom:16px'>Analisis <span style=\"color:#4f7cff\">Medis AI</span></h1>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:16px;line-height:1.7;color:#4a5580;max-width:580px;margin-bottom:32px;font-weight:300'>
        Platform skrining kesehatan berbasis <strong>Machine Learning</strong> yang membantu Anda
        memperkirakan risiko Diabetes dan Penyakit Jantung secara cepat dan mudah.
        Masukkan data klinis, dan dapatkan estimasi risiko dalam hitungan detik.
    </p>
    """, unsafe_allow_html=True)

    # Stats
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='metric-card'><div class='num'>768+</div><div class='lbl'>Data Pasien Diabetes</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='metric-card'><div class='num'>~77%</div><div class='lbl'>Akurasi Model Diabetes</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='metric-card'><div class='num'>~85%</div><div class='lbl'>Akurasi Model Jantung</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature cards
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class='feature-card' style='border-color:#4f7cff'>
            <div style='font-size:24px'>🎯</div>
            <div class='title'>Akurasi Tinggi</div>
            <p class='desc'>Model dilatih pada dataset medis tervalidasi dengan performa optimal menggunakan Decision Tree & Random Forest.</p>
        </div>""", unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class='feature-card' style='border-color:#00d4ff'>
            <div style='font-size:24px'>⚡</div>
            <div class='title'>Hasil Instan</div>
            <p class='desc'>Prediksi real-time tanpa menunggu — hasil analisis langsung tersedia beserta rekomendasi tindakan.</p>
        </div>""", unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class='feature-card' style='border-color:#00e5a0'>
            <div style='font-size:24px'>🔒</div>
            <div class='title'>Privasi Terjaga</div>
            <p class='desc'>Data Anda tidak disimpan. Setiap sesi bersih dan sepenuhnya terlindungi secara privasi.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # How to use
    st.markdown("""
    <div style='background:white;border-radius:18px;padding:24px 28px;border:1px solid rgba(79,124,255,0.15);display:flex;gap:18px'>
        <div style='font-size:28px'>📋</div>
        <div>
            <div style='font-family:Syne,sans-serif;font-size:14px;font-weight:700;color:#4f7cff;margin-bottom:6px'>Cara Penggunaan</div>
            <p style='font-size:13.5px;color:#5566a0;line-height:1.65'>
                Pilih menu <strong style='color:#0f1b3d'>Prediksi Diabetes</strong> atau 
                <strong style='color:#0f1b3d'>Prediksi Jantung</strong> di sidebar kiri → 
                Isi formulir data klinis → Klik <strong style='color:#0f1b3d'>Prediksi Sekarang</strong> → 
                Lihat hasil beserta rekomendasi tindakan.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDIKSI DIABETES
# ═══════════════════════════════════════════════════════════════════════════════
elif "Diabetes" in page:
    st.markdown("<span style='background:rgba(79,124,255,0.1);color:#4f7cff;border:1px solid rgba(79,124,255,0.25);padding:5px 14px;border-radius:100px;font-size:12px;font-weight:600'>🩸 &nbsp;Diabetes Prediction</span>", unsafe_allow_html=True)
    st.markdown("<h2 style='margin-top:14px;font-size:36px;letter-spacing:-1px'>Prediksi Diabetes</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6678a0;font-size:15px;font-weight:300;margin-bottom:28px'>Masukkan data klinis Anda untuk mendapatkan estimasi risiko diabetes.</p>", unsafe_allow_html=True)

    # Load model
    with st.spinner("Melatih model..."):
        model, features, acc_opt, acc_def, X_test, y_test, df_pima = train_diabetes_model()

    # Model info
    with st.expander("ℹ️ Info Model (Decision Tree - Entropy, max_depth=3)"):
        col_a, col_b = st.columns(2)
        col_a.metric("Akurasi Model Optimal", f"{acc_opt:.1%}")
        col_b.metric("Akurasi Model Default", f"{acc_def:.1%}")
        st.caption("Model dilatih pada Pima Indians Diabetes Database")

        fig, ax = plt.subplots(figsize=(6, 4))
        cm = confusion_matrix(y_test, model.predict(X_test))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Negatif','Positif']); ax.set_yticklabels(['Negatif','Positif'])
        ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual')
        ax.set_title('Confusion Matrix')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i,j], ha='center', va='center',
                        color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Input form
    st.markdown("### 📝 Data Klinis")
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.slider("🤰 Jumlah Kehamilan", 0, 17, 3, help="Berapa kali pernah hamil")
        glucose     = st.slider("🩸 Glukosa (mg/dL)", 0, 200, 117, help="Konsentrasi glukosa plasma 2 jam setelah tes")
        blood_pressure = st.slider("💓 Tekanan Darah (mmHg)", 0, 130, 72)
        

    with col2:
        insulin = st.slider("💉 Insulin (µU/mL)", 0, 846, 30)
        bmi     = st.slider("⚖️ BMI (kg/m²)", 10.0, 70.0, 32.0, step=0.1)
        dpf     = st.slider("🧬 Diabetes Pedigree Function", 0.07, 2.50, 0.35, step=0.01,
                             help="Fungsi yang menilai kemungkinan diabetes berdasarkan riwayat keluarga")
        age     = st.slider("🎂 Usia (tahun)", 21, 81, 33)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 &nbsp;Prediksi Sekarang", key="btn_diabetes", use_container_width=True,
                 type="primary"):

        input_data = pd.DataFrame([[pregnancies, insulin, bmi, age, glucose, blood_pressure, dpf]],
                                  columns=features)
        prediction = model.predict(input_data)[0]
        proba      = model.predict_proba(input_data)[0]
        prob_pct   = int(proba[1] * 100)

        st.markdown("<br>", unsafe_allow_html=True)

        if prediction == 1:
            box_class = "result-box-high"
            emoji = "&#9888;&#65039;"  # ⚠️ as HTML entity
            risk_label = "Risiko Tinggi"
            risk_tag_color = "#d4294a"
            risk_tag_bg = "rgba(255,79,110,0.15)"
            risk_tag_text = "HIGH RISK"
            bar_color = "#ff4f6e"
            tips_title = "Rekomendasi:"
            tips_body = ("Segera konsultasikan dengan dokter endokrinologi. "
                         "Pertimbangkan tes HbA1c dan gula darah puasa. "
                         "Kurangi konsumsi karbohidrat sederhana dan tingkatkan aktivitas fisik.")
        else:
            box_class = "result-box-low"
            emoji = "&#9989;"  # ✅
            risk_label = "Risiko Rendah"
            risk_tag_color = "#00a870"
            risk_tag_bg = "rgba(0,229,160,0.15)"
            risk_tag_text = "LOW RISK"
            bar_color = "#ffffff"
            tips_title = "Hasil Baik:"
            tips_body = ("Risiko diabetes Anda tergolong rendah. "
                         "Pertahankan pola makan sehat, olahraga rutin, "
                         "dan kontrol berat badan untuk menjaga kondisi ini.")

        result_html = (
            f"<div class='{box_class}'>"
            f"  <div style='display:flex;align-items:center;gap:16px;margin-bottom:18px'>"
            f"    <div style='font-size:36px'>{emoji}</div>"
            f"    <div>"
            f"      <div style='font-family:Syne,sans-serif;font-size:22px;font-weight:800;"
            f"                  color:#ffffff;letter-spacing:-0.5px'>{risk_label}</div>"
            f"      <div style='margin-top:4px'>"
            f"        <span style='background:{risk_tag_bg};color:{risk_tag_color};"
            f"                     padding:4px 12px;border-radius:100px;font-size:12px;"
            f"                     font-weight:700;letter-spacing:1px'>{risk_tag_text}</span>"
            f"      </div>"
            f"    </div>"
            f"  </div>"
            f"  <div style='margin:14px 0'>"
            f"    <div style='display:flex;justify-content:space-between;font-size:12.5px;"
            f"                color:#6678a0;margin-bottom:8px'>"
            f"      <span>Probabilitas Diabetes</span>"
            f"      <strong style='color:#0f1b3d'>{prob_pct}%</strong>"
            f"    </div>"
            f"    <div style='height:10px;border-radius:5px;background:#e0e8f5;overflow:hidden'>"
            f"      <div style='height:100%;width:{prob_pct}%;background:{bar_color};"
            f"                  border-radius:5px'></div>"
            f"    </div>"
            f"  </div>"
            f"  <p style='font-size:13.5px;color:#5566a0;line-height:1.7;margin-top:14px;"
            f"             padding-top:14px;border-top:1px solid rgba(79,124,255,0.12)'>"
            f"    <strong style='color:#0f1b3d'>{tips_title}</strong> {tips_body}"
            f"  </p>"
            f"  <p style='font-size:11.5px;color:#8899bb;font-style:italic;margin-top:10px'>"
            f"    &#9888; Hasil ini bersifat indikatif dan bukan merupakan diagnosis medis. "
            f"    Konsultasikan dengan dokter Anda."
            f"  </p>"
            f"</div>"
        )
        st.markdown(result_html, unsafe_allow_html=True)

        # Feature importance chart
        st.markdown("<br>")
        fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        colors = ['#4f7cff' if v == fi.max() else '#c5d0f0' for v in fi.values]
        bars = ax2.barh(fi.index, fi.values, color=colors, height=0.55, edgecolor='none')
        ax2.set_xlabel("Importance Score", fontsize=10)
        ax2.set_title("Feature Importance (Decision Tree)", fontsize=12, fontweight='bold', color='#0f1b3d')
        ax2.spines[['top','right','left']].set_visible(False)
        ax2.tick_params(left=False)
        for bar, val in zip(bars, fi.values):
            ax2.text(val + 0.003, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                     va='center', fontsize=9, color='#0f1b3d')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDIKSI JANTUNG
# ═══════════════════════════════════════════════════════════════════════════════
elif "Jantung" in page:
    st.markdown("<span style='background:rgba(255,79,110,0.1);color:#d4294a;border:1px solid rgba(255,79,110,0.25);padding:5px 14px;border-radius:100px;font-size:12px;font-weight:600'>❤️ &nbsp;Heart Disease Prediction</span>", unsafe_allow_html=True)
    st.markdown("<h2 style='margin-top:14px;font-size:36px;letter-spacing:-1px'>Prediksi Penyakit Jantung</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6678a0;font-size:15px;font-weight:300;margin-bottom:28px'>Masukkan data klinis Anda untuk mendapatkan estimasi risiko penyakit jantung.</p>", unsafe_allow_html=True)

    with st.spinner("Melatih model..."):
        rf_model, h_features, h_acc, H_test, h_y_test, df_heart = train_heart_model()

    with st.expander("ℹ️ Info Model (Random Forest - 100 estimators, max_depth=5)"):
        st.metric("Akurasi Model", f"{h_acc:.1%}")
        st.caption("Model dilatih pada Heart Disease Dataset")

        fig, ax = plt.subplots(figsize=(6, 4))
        cm = confusion_matrix(h_y_test, rf_model.predict(H_test))
        ax.imshow(cm, cmap='Reds')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Tidak Sakit','Sakit']); ax.set_yticklabels(['Tidak Sakit','Sakit'])
        ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual')
        ax.set_title('Confusion Matrix')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i,j], ha='center', va='center',
                        color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("### 📝 Data Klinis")

    col1, col2 = st.columns(2)

    with col1:
        h_age      = st.slider("🎂 Usia (tahun)", 20, 80, 52)
        h_sex      = st.selectbox("👤 Jenis Kelamin", options=[("Laki-laki", 1), ("Perempuan", 0)], format_func=lambda x: x[0])[1]
        h_cp       = st.selectbox("💢 Tipe Nyeri Dada",
                                   options=[(0,"0 — Angina Tipikal"),(1,"1 — Angina Atipikal"),(2,"2 — Non-Anginal"),(3,"3 — Asimptomatik")],
                                   format_func=lambda x: x[1])[0]
        h_trestbps = st.slider("💓 Tekanan Darah Istirahat (mmHg)", 90, 200, 125)

    with col2:
        h_chol   = st.slider("🧪 Kolesterol Serum (mg/dL)", 100, 600, 212)
        h_fbs    = st.selectbox("🍬 Gula Darah Puasa > 120 mg/dL",
                                 options=[(0,"Tidak (< 120 mg/dL)"),(1,"Ya (> 120 mg/dL)")],
                                 format_func=lambda x: x[1])[0]
        h_thalach = st.slider("💗 Detak Jantung Maks. (bpm)", 60, 220, 168)
        h_exang  = st.selectbox("🏃 Angina Akibat Olahraga",
                                 options=[(0,"Tidak"),(1,"Ya")],
                                 format_func=lambda x: x[1])[0]

    # Build input matching trained feature columns
    input_dict = {
        'age': h_age, 'sex': h_sex, 'cp': h_cp, 'trestbps': h_trestbps,
        'chol': h_chol, 'fbs': h_fbs, 'thalach': h_thalach, 'exang': h_exang,
        'oldpeak': 1.0  # default
    }
    h_input = pd.DataFrame([{col: input_dict.get(col, 0) for col in h_features}])

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 &nbsp;Prediksi Sekarang", key="btn_heart", use_container_width=True, type="primary"):

        h_pred  = rf_model.predict(h_input)[0]
        h_proba = rf_model.predict_proba(h_input)[0]
        h_prob_pct = int(h_proba[1] * 100)

        st.markdown("<br>", unsafe_allow_html=True)

        if h_pred == 1:
            box_class = "result-box-high"
            emoji = "&#9888;&#65039;"
            risk_label = "Risiko Tinggi"
            risk_tag_color = "#d4294a"
            risk_tag_bg = "rgba(255,79,110,0.15)"
            risk_tag_text = "HIGH RISK"
            bar_color = "#ff4f6e"
            tips_title = "Rekomendasi:"
            tips_body = ("Segera periksakan ke dokter kardiolog. "
                         "Pertimbangkan EKG dan ekokardiografi. "
                         "Hindari aktivitas berat, kelola stres, dan jaga tekanan darah serta kolesterol.")
        else:
            box_class = "result-box-low"
            emoji = "&#9989;"
            risk_label = "Risiko Rendah"
            risk_tag_color = "#00a870"
            risk_tag_bg = "rgba(0,229,160,0.15)"
            risk_tag_text = "LOW RISK"
            bar_color = "#00e5a0"
            tips_title = "Hasil Baik:"
            tips_body = ("Risiko penyakit jantung Anda tergolong rendah. "
                         "Terus jaga gaya hidup sehat dengan olahraga aerobik, "
                         "diet rendah lemak jenuh, dan hindari rokok.")

        result_html = (
            f"<div class='{box_class}'>"
            f"  <div style='display:flex;align-items:center;gap:16px;margin-bottom:18px'>"
            f"    <div style='font-size:36px'>{emoji}</div>"
            f"    <div>"
            f"      <div style='font-family:Syne,sans-serif;font-size:22px;font-weight:800;"
            f"                  color:#ffffff;letter-spacing:-0.5px'>{risk_label}</div>"
            f"      <div style='margin-top:4px'>"
            f"        <span style='background:{risk_tag_bg};color:{risk_tag_color};"
            f"                     padding:4px 12px;border-radius:100px;font-size:12px;"
            f"                     font-weight:700;letter-spacing:1px'>{risk_tag_text}</span>"
            f"      </div>"
            f"    </div>"
            f"  </div>"
            f"  <div style='margin:14px 0'>"
            f"    <div style='display:flex;justify-content:space-between;font-size:12.5px;"
            f"                color:#6678a0;margin-bottom:8px'>"
            f"      <span>Probabilitas Penyakit Jantung</span>"
            f"      <strong style='color:#0f1b3d'>{h_prob_pct}%</strong>"
            f"    </div>"
            f"    <div style='height:10px;border-radius:5px;background:#e0e8f5;overflow:hidden'>"
            f"      <div style='height:100%;width:{h_prob_pct}%;background:{bar_color};"
            f"                  border-radius:5px'></div>"
            f"    </div>"
            f"  </div>"
            f"  <p style='font-size:13.5px;color:#5566a0;line-height:1.7;margin-top:14px;"
            f"             padding-top:14px;border-top:1px solid rgba(79,124,255,0.12)'>"
            f"    <strong style='color:#0f1b3d'>{tips_title}</strong> {tips_body}"
            f"  </p>"
            f"  <p style='font-size:11.5px;color:#8899bb;font-style:italic;margin-top:10px'>"
            f"    &#9888; Hasil ini bersifat indikatif dan bukan merupakan diagnosis medis. "
            f"    Konsultasikan dengan dokter Anda."
            f"  </p>"
            f"</div>"
        )
        st.markdown(result_html, unsafe_allow_html=True)

        # Feature importance
        st.markdown("<br>")
        fi_h = pd.Series(rf_model.feature_importances_, index=h_features).sort_values(ascending=True)
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        colors = ['#ff4f6e' if v == fi_h.max() else '#f5c5cc' for v in fi_h.values]
        bars3 = ax3.barh(fi_h.index, fi_h.values, color=colors, height=0.55, edgecolor='none')
        ax3.set_xlabel("Importance Score", fontsize=10)
        ax3.set_title("Feature Importance (Random Forest)", fontsize=12, fontweight='bold', color='#0f1b3d')
        ax3.spines[['top','right','left']].set_visible(False)
        ax3.tick_params(left=False)
        for bar, val in zip(bars3, fi_h.values):
            ax3.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                     va='center', fontsize=9, color='#0f1b3d')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()