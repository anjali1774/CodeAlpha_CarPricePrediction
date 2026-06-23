# ============================================================
#   CarPrice AI Pro — Streamlit Web Application
#   Used Car Price Prediction using Machine Learning
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="CarPrice AI Pro",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #0f1117 100%);
        border-right: 1px solid #2d3748;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1f2e, #2d3748);
        border: 1px solid #4a5568; border-radius: 12px; padding: 16px;
    }
    [data-testid="stMetricValue"] { color: #63b3ed !important; font-size: 1.8rem !important; }
    [data-testid="stMetricLabel"] { color: #a0aec0 !important; }
    h1, h2, h3 { color: #e2e8f0 !important; }
    p, li { color: #a0aec0; }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border: none; border-radius: 10px;
        padding: 12px 30px; font-size: 1rem; font-weight: 600;
        width: 100%; transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.4);
    }
    .card {
        background: linear-gradient(135deg, #1a1f2e, #2d3748);
        border: 1px solid #4a5568; border-radius: 16px;
        padding: 24px; margin: 10px 0;
    }
    .hero-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px; padding: 40px;
        text-align: center; margin-bottom: 30px;
    }
    .hero-card h1 { color: white !important; font-size: 2.5rem !important; }
    .hero-card p  { color: rgba(255,255,255,0.85) !important; font-size: 1.1rem; }
    .price-card {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 16px; padding: 30px; text-align: center;
    }
    .price-card h2 { color: white !important; font-size: 2.2rem !important; }
    .price-card p  { color: rgba(255,255,255,0.9) !important; }
    .stat-card {
        background: linear-gradient(135deg, #1a1f2e, #2d3748);
        border-left: 4px solid #667eea; border-radius: 12px;
        padding: 20px; margin: 8px 0;
    }
    .stat-card h3 { color: #63b3ed !important; margin: 0; }
    .stat-card p  { color: #a0aec0; margin: 0; font-size: 0.9rem; }
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background-color: #2d3748 !important;
        color: #e2e8f0 !important;
        border: 1px solid #4a5568 !important;
        border-radius: 8px !important;
    }
    .stSlider > div > div > div { background: #667eea !important; }
    hr { border-color: #2d3748; }
    .badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; padding: 4px 12px;
        border-radius: 20px; font-size: 0.8rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Data & Model ────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model/best_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("model/processed_data.csv")

@st.cache_data
def load_metadata():
    with open("model/metadata.json") as f:
        return json.load(f)

model = load_model()
df    = load_data()
meta  = load_metadata()

# ─── Session State Init ───────────────────────────────────────
if 'show_result'   not in st.session_state: st.session_state['show_result']   = False
if 'show_fair'     not in st.session_state: st.session_state['show_fair']     = False
if 'show_emi'      not in st.session_state: st.session_state['show_emi']      = False
if 'pred_price'    not in st.session_state: st.session_state['pred_price']    = None
if 'present_price' not in st.session_state: st.session_state['present_price'] = None
if 'driven_kms'    not in st.session_state: st.session_state['driven_kms']    = None
if 'car_age'       not in st.session_state: st.session_state['car_age']       = None
if 'fuel_type'     not in st.session_state: st.session_state['fuel_type']     = None
if 'asking_price_val' not in st.session_state: st.session_state['asking_price_val'] = None

# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0;'>
        <div style='font-size:3rem'>🚗</div>
        <h2 style='color:#63b3ed !important; margin:0'>CarPrice AI</h2>
        <p style='color:#a0aec0; font-size:0.85rem'>Professional Edition</p>
        <span class='badge'>v2.0</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='color:#a0aec0;font-size:0.8rem;text-transform:uppercase;letter-spacing:2px'>Navigation</p>", unsafe_allow_html=True)

    pages = {
        "🏠  Home Dashboard"   : "home",
        "🤖  Price Prediction" : "predict",
        "💰  EMI Calculator"   : "emi",
        "📈  Market Insights"  : "market",
        "🚘  Car Comparator"   : "compare",
    }
    selected = st.radio("", list(pages.keys()), label_visibility="collapsed")
    page = pages[selected]

    st.markdown("---")
    st.markdown(f"""
    <div class='stat-card'>
        <h3>🏆 {meta['best_model_name']}</h3>
        <p>Best Model — {meta['best_r2']*100:.1f}% Accurate</p>
    </div>
    <div class='stat-card'>
        <h3>📦 {meta['total_cars']} Cars</h3>
        <p>Training Dataset Size</p>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#   PAGE 1 — HOME DASHBOARD
# ════════════════════════════════════════════════════════════
if page == "home":
    st.markdown("""
    <div class='hero-card'>
        <h1>🚗 CarPrice AI Pro</h1>
        <p>Professional Machine Learning System for Used Car Price Prediction</p>
        <p>Powered by Gradient Boosting | 96.53% Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🚗 Total Cars",     f"{meta['total_cars']}")
    c2.metric("🎯 Model Accuracy", f"{meta['best_r2']*100:.1f}%")
    c3.metric("🤖 Models Trained", "4")
    c4.metric("📅 Year Range",     f"{meta['year_min']} – {meta['year_max']}")

    st.markdown("---")
    st.markdown("### 🌟 Application Features")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown("""<div class='card'>
            <h3>🤖 AI Price Prediction</h3>
            <p>Enter car details and get an instant AI-powered predicted selling price with depreciation analysis and smart insights.</p>
        </div>""", unsafe_allow_html=True)
    with r1c2:
        st.markdown("""<div class='card'>
            <h3>🏷️ Fair Price Checker</h3>
            <p>Enter the seller's asking price and instantly know if it's an excellent deal, fair, slightly overpriced, or to avoid.</p>
        </div>""", unsafe_allow_html=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown("""<div class='card'>
            <h3>💰 EMI Calculator</h3>
            <p>Calculate your monthly car loan EMI based on predicted price, down payment, interest rate and tenure.</p>
        </div>""", unsafe_allow_html=True)
    with r2c2:
        st.markdown("""<div class='card'>
            <h3>🚘 Car Comparator</h3>
            <p>Select any two cars and compare them side by side — price, mileage, depreciation and more.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Price Distribution by Fuel Type")
    fig = px.box(df, x="Fuel_Type", y="Selling_Price", color="Fuel_Type",
                 color_discrete_sequence=["#667eea","#11998e","#f093fb"],
                 template="plotly_dark",
                 title="Selling Price Distribution by Fuel Type")
    fig.update_layout(plot_bgcolor="#1a1f2e", paper_bgcolor="#1a1f2e",
                      font_color="#e2e8f0", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════
#   PAGE 2 — PRICE PREDICTION
# ════════════════════════════════════════════════════════════
elif page == "predict":
    st.markdown("## 🤖 Car Price Prediction")
    st.markdown("<p>Enter the car details below to get an AI-powered price estimate.</p>", unsafe_allow_html=True)

    # Photo Upload
    st.markdown("### 📸 Upload Your Car Photo (Optional)")
    uploaded_photo = st.file_uploader(
        "Upload a photo of your car",
        type=["jpg","jpeg","png"],
        help="This is optional and does not affect the price prediction."
    )
    if uploaded_photo is not None:
        p1, p2, p3 = st.columns([1,2,1])
        with p2:
            st.image(uploaded_photo, caption="📸 Your Car", use_container_width=True)
        st.success("✅ Car photo uploaded! Now fill in the details below.")
    else:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1a1f2e,#2d3748);
                    border:2px dashed #4a5568;border-radius:12px;
                    padding:18px;text-align:center;margin-bottom:10px;'>
            <p style='color:#a0aec0;margin:0'>
                📷 No photo uploaded — upload your car photo above for a personalized experience!
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown("### 📝 Enter Car Details")
        present_price = st.number_input("💰 Present Showroom Price (Lakhs)",
                                         min_value=0.5, max_value=100.0,
                                         value=5.0, step=0.1,
                                         help="Current ex-showroom price of the car")
        driven_kms = st.number_input("🛣️ Kilometers Driven",
                                      min_value=0, max_value=500000,
                                      value=30000, step=1000,
                                      help="Total kilometers the car has been driven")
        car_year = st.slider("📅 Manufacturing Year",
                              min_value=2000, max_value=2024, value=2018,
                              help="Year the car was manufactured")
        car_age = 2024 - car_year
        st.info(f"🕐 Car Age: **{car_age} years old**")
        fuel_type    = st.selectbox("⛽ Fuel Type",    meta['fuel_classes'])
        selling_type = st.selectbox("👤 Seller Type",  meta['selling_classes'])
        transmission = st.selectbox("⚙️ Transmission", meta['transmission_classes'])
        owner = st.selectbox("🔑 Number of Previous Owners", [0,1,2,3],
                              format_func=lambda x: ["1st Owner","2nd Owner","3rd Owner","4th+ Owner"][x])

        if st.button("🚀 Predict Price Now!", key="predict_btn"):
            fuel_enc  = meta['fuel_classes'].index(fuel_type)
            sell_enc  = meta['selling_classes'].index(selling_type)
            trans_enc = meta['transmission_classes'].index(transmission)
            inp  = np.array([[present_price, driven_kms, car_age,
                               fuel_enc, sell_enc, trans_enc, owner]])
            pred = float(model.predict(inp)[0])
            pred = max(0.1, pred)
            st.session_state['show_result']   = True
            st.session_state['show_fair']     = False
            st.session_state['pred_price']    = pred
            st.session_state['present_price'] = present_price
            st.session_state['driven_kms']    = driven_kms
            st.session_state['car_age']       = car_age
            st.session_state['fuel_type']     = fuel_type

    with col2:
        st.markdown("### 💡 Prediction Result")

        if st.session_state.get('show_result') and st.session_state['pred_price']:
            predicted_price = st.session_state['pred_price']
            pres_price      = st.session_state['present_price']
            driv_kms        = st.session_state['driven_kms']
            c_age           = st.session_state['car_age']
            f_type          = st.session_state['fuel_type']
            depreciation    = ((pres_price - predicted_price) / pres_price) * 100

            # Price card
            st.markdown(f"""
            <div class='price-card'>
                <p style='font-size:1rem;margin:0'>🎯 Predicted Selling Price</p>
                <h2>₹ {predicted_price:.2f} Lakhs</h2>
                <p>≈ ₹ {predicted_price*100000:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            m1.metric("🏷️ Showroom Price", f"₹{pres_price:.1f}L")
            m2.metric("📉 Depreciation",   f"{depreciation:.1f}%",
                      delta=f"-₹{pres_price-predicted_price:.1f}L",
                      delta_color="inverse")

            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=predicted_price,
                delta={"reference": pres_price, "valueformat": ".2f"},
                title={"text": "Predicted vs Showroom Price (Lakhs)", "font": {"color": "#e2e8f0"}},
                number={"suffix": "L", "font": {"color": "#38ef7d", "size": 40}},
                gauge={
                    "axis"       : {"range": [0, pres_price*1.2], "tickcolor": "#e2e8f0"},
                    "bar"        : {"color": "#38ef7d"},
                    "bgcolor"    : "#1a1f2e",
                    "bordercolor": "#4a5568",
                    "steps": [
                        {"range": [0,              pres_price*0.3], "color": "rgba(255,107,107,0.2)"},
                        {"range": [pres_price*0.3, pres_price*0.6], "color": "rgba(255,217,61,0.2)"},
                        {"range": [pres_price*0.6, pres_price*1.2], "color": "rgba(107,203,119,0.2)"},
                    ],
                    "threshold": {"line": {"color": "#667eea","width": 4},
                                  "thickness": 0.75, "value": pres_price}
                }
            ))
            fig.update_layout(paper_bgcolor="#1a1f2e", font_color="#e2e8f0", height=300)
            st.plotly_chart(fig, use_container_width=True)

            # AI Insights
            st.markdown("### 💬 AI Insights")
            if depreciation > 60:
                st.warning(f"⚠️ High depreciation ({depreciation:.1f}%) — The car has lost significant value.")
            elif depreciation > 30:
                st.info(f"ℹ️ Moderate depreciation ({depreciation:.1f}%) — Fair used car value.")
            else:
                st.success(f"✅ Low depreciation ({depreciation:.1f}%) — This car holds its value well!")
            if driv_kms > 100000:
                st.warning("🛣️ High mileage detected — This significantly affects the resale value.")
            if c_age <= 3:
                st.success("🆕 Relatively new car — Expect a strong resale value.")
            if f_type == "Diesel":
                st.info("⛽ Diesel cars typically retain better resale value than petrol cars.")

            # ── Fair Price Checker ─────────────────────────────
            st.markdown("---")
            st.markdown("### 🏷️ Fair Price Checker")
            st.markdown("<p>Enter the seller's asking price to check if it is a fair deal.</p>", unsafe_allow_html=True)

            asking_price = st.number_input(
                "💬 Seller's Asking Price (Lakhs)",
                min_value=0.1, max_value=100.0,
                value=round(predicted_price, 1), step=0.1,
                key="asking_price_input"
            )
            if st.button("🔍 Check Fair Price", key="check_fair_btn"):
                st.session_state['show_fair']        = True
                st.session_state['asking_price_val'] = asking_price

            if st.session_state.get('show_fair') and st.session_state['asking_price_val']:
                ask_p       = st.session_state['asking_price_val']
                difference  = ask_p - predicted_price
                diff_pct    = (difference / predicted_price) * 100
                lower_bound = predicted_price * 0.90
                upper_bound = predicted_price * 1.10

                bar_color = "#ff6b6b" if ask_p > upper_bound else "#ffd93d" if ask_p > predicted_price else "#38ef7d"
                fig_fair = go.Figure()
                fig_fair.add_trace(go.Bar(
                    x=["AI Predicted Price","Seller Asking Price"],
                    y=[predicted_price, ask_p],
                    marker_color=["#38ef7d", bar_color],
                    text=[f"₹{predicted_price:.2f}L", f"₹{ask_p:.2f}L"],
                    textposition="outside",
                    textfont=dict(color="#e2e8f0", size=14)
                ))
                fig_fair.update_layout(
                    template="plotly_dark", plot_bgcolor="#1a1f2e",
                    paper_bgcolor="#1a1f2e", font_color="#e2e8f0",
                    title="AI Predicted Price vs Seller Asking Price",
                    yaxis_title="Price (Lakhs)", showlegend=False, height=300
                )
                st.plotly_chart(fig_fair, use_container_width=True)

                v1, v2, v3 = st.columns(3)
                v1.metric("🤖 AI Fair Price", f"₹{predicted_price:.2f}L")
                v2.metric("💬 Seller Asking", f"₹{ask_p:.2f}L")
                v3.metric("📊 Difference",    f"₹{abs(difference):.2f}L",
                          delta=f"{diff_pct:+.1f}%",
                          delta_color="inverse" if difference > 0 else "normal")
                st.markdown("<br>", unsafe_allow_html=True)

                if ask_p < lower_bound:
                    st.markdown(f"""<div style='background:linear-gradient(135deg,#11998e,#38ef7d);
                        border-radius:14px;padding:24px;text-align:center;'>
                        <h2 style='color:white !important;margin:0'>🤩 Excellent Deal!</h2>
                        <p style='color:white !important;font-size:1.1rem'>
                            Seller is asking <b>₹{abs(difference):.2f}L LESS</b> than fair price.<br>
                            Great bargain — go ahead and buy!</p>
                        <p style='color:white !important'>Fair Range: ₹{lower_bound:.2f}L — ₹{upper_bound:.2f}L</p>
                    </div>""", unsafe_allow_html=True)
                elif lower_bound <= ask_p <= upper_bound:
                    st.markdown(f"""<div style='background:linear-gradient(135deg,#667eea,#764ba2);
                        border-radius:14px;padding:24px;text-align:center;'>
                        <h2 style='color:white !important;margin:0'>✅ Fair Deal!</h2>
                        <p style='color:white !important;font-size:1.1rem'>
                            The asking price is within the fair price range.<br>
                            You can proceed — a small negotiation is still worth trying!</p>
                        <p style='color:white !important'>Fair Range: ₹{lower_bound:.2f}L — ₹{upper_bound:.2f}L</p>
                    </div>""", unsafe_allow_html=True)
                elif ask_p <= predicted_price * 1.20:
                    st.markdown(f"""<div style='background:linear-gradient(135deg,#f7971e,#ffd200);
                        border-radius:14px;padding:24px;text-align:center;'>
                        <h2 style='color:white !important;margin:0'>⚠️ Slightly Overpriced</h2>
                        <p style='color:white !important;font-size:1.1rem'>
                            Seller is asking <b>₹{difference:.2f}L MORE</b> than fair price.<br>
                            Try negotiating down to ₹{predicted_price:.2f}L — ₹{upper_bound:.2f}L.</p>
                        <p style='color:white !important'>Fair Range: ₹{lower_bound:.2f}L — ₹{upper_bound:.2f}L</p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div style='background:linear-gradient(135deg,#cb2d3e,#ef473a);
                        border-radius:14px;padding:24px;text-align:center;'>
                        <h2 style='color:white !important;margin:0'>❌ Overpriced — Avoid!</h2>
                        <p style='color:white !important;font-size:1.1rem'>
                            Seller is asking <b>₹{difference:.2f}L MORE</b> than fair price.<br>
                            This car is significantly overpriced. Walk away or negotiate hard!</p>
                        <p style='color:white !important'>Fair Range: ₹{lower_bound:.2f}L — ₹{upper_bound:.2f}L</p>
                    </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='card' style='text-align:center;padding:60px 20px'>
                <div style='font-size:4rem'>🤖</div>
                <h3>Ready to Predict!</h3>
                <p>Fill in the car details on the left and click<br>
                <b>"Predict Price Now!"</b> to get the AI estimate.</p>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#   PAGE 3 — EMI CALCULATOR
# ════════════════════════════════════════════════════════════
elif page == "emi":
    st.markdown("## 💰 EMI Calculator")
    st.markdown("<p>Calculate your monthly car loan EMI based on the predicted or actual price.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("### 📝 Loan Details")
        default_price = round(float(st.session_state['pred_price']), 1) if st.session_state['pred_price'] else 5.0
        car_price = st.number_input("🚗 Car Price (Lakhs)",
                                     min_value=0.5, max_value=100.0,
                                     value=default_price, step=0.1,
                                     help="Auto-filled from prediction if available")
        down_payment = st.number_input("💵 Down Payment (Lakhs)",
                                        min_value=0.0, max_value=car_price,
                                        value=round(car_price*0.2, 1), step=0.1)
        interest_rate = st.slider("📈 Annual Interest Rate (%)",
                                   min_value=5.0, max_value=20.0,
                                   value=8.5, step=0.1)
        tenure_years = st.selectbox("📅 Loan Tenure",
                                     [1,2,3,4,5,6,7], index=2,
                                     format_func=lambda x: f"{x} Year{'s' if x>1 else ''}")
        if st.button("💰 Calculate EMI", key="calc_emi"):
            st.session_state['show_emi'] = True

    with col2:
        st.markdown("### 📊 EMI Result")
        if st.session_state.get('show_emi'):
            loan_amount  = car_price - down_payment
            monthly_rate = interest_rate / (12*100)
            n_months     = tenure_years * 12

            if monthly_rate > 0:
                emi = loan_amount*100000 * monthly_rate * (1+monthly_rate)**n_months / \
                      ((1+monthly_rate)**n_months - 1)
            else:
                emi = loan_amount*100000 / n_months

            total_payment  = emi * n_months
            total_interest = total_payment - (loan_amount*100000)

            st.markdown(f"""
            <div class='price-card'>
                <p style='font-size:1rem;margin:0'>💰 Monthly EMI</p>
                <h2>₹ {emi:,.0f}</h2>
                <p>Per month for {tenure_years} year{'s' if tenure_years>1 else ''}</p>
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            e1, e2, e3 = st.columns(3)
            e1.metric("🚗 Loan Amount",    f"₹{loan_amount:.1f}L")
            e2.metric("💸 Total Payment",  f"₹{total_payment/100000:.2f}L")
            e3.metric("📈 Total Interest", f"₹{total_interest/100000:.2f}L")

            fig_emi = px.pie(
                values=[loan_amount*100000, total_interest],
                names=["Principal Amount","Total Interest"],
                color_discrete_sequence=["#667eea","#f093fb"],
                template="plotly_dark",
                title="Principal vs Interest Breakdown",
                hole=0.4
            )
            fig_emi.update_layout(plot_bgcolor="#1a1f2e", paper_bgcolor="#1a1f2e", font_color="#e2e8f0")
            st.plotly_chart(fig_emi, use_container_width=True)

            st.markdown("### 📅 Yearly Payment Summary")
            yearly_data = []
            balance = loan_amount * 100000
            for yr in range(1, tenure_years+1):
                y_principal = 0
                y_interest  = 0
                for mo in range(12):
                    int_pay  = balance * monthly_rate
                    prin_pay = emi - int_pay
                    balance  = max(0, balance - prin_pay)
                    y_principal += prin_pay
                    y_interest  += int_pay
                yearly_data.append({
                    "Year"               : f"Year {yr}",
                    "Principal Paid (₹)" : f"{y_principal:,.0f}",
                    "Interest Paid (₹)"  : f"{y_interest:,.0f}",
                    "Balance (₹)"        : f"{balance:,.0f}"
                })
            st.dataframe(pd.DataFrame(yearly_data), use_container_width=True, hide_index=True)
        else:
            st.markdown("""
            <div class='card' style='text-align:center;padding:60px 20px'>
                <div style='font-size:4rem'>💰</div>
                <h3>Ready to Calculate!</h3>
                <p>Fill in the loan details on the left and click<br>
                <b>"Calculate EMI"</b> to see your monthly installment.</p>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#   PAGE 4 — MARKET INSIGHTS
# ════════════════════════════════════════════════════════════
elif page == "market":
    st.markdown("## 📈 Car Market Insights")
    st.markdown("<p>Explore price trends and market analysis across the used car dataset.</p>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        top_cars = df.groupby('Car_Name')['Selling_Price'].mean().nlargest(10).reset_index()
        fig = px.bar(top_cars, x="Selling_Price", y="Car_Name", orientation='h',
                     color="Selling_Price", color_continuous_scale="viridis",
                     template="plotly_dark", title="🏆 Top 10 Highest Average Priced Cars")
        fig.update_layout(plot_bgcolor="#1a1f2e", paper_bgcolor="#1a1f2e", font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fuel_avg = df.groupby('Fuel_Type')['Selling_Price'].mean().reset_index()
        fig = px.bar(fuel_avg, x="Fuel_Type", y="Selling_Price", color="Fuel_Type",
                     color_discrete_sequence=["#667eea","#11998e","#f093fb"],
                     template="plotly_dark", title="⛽ Average Selling Price by Fuel Type")
        fig.update_layout(plot_bgcolor="#1a1f2e", paper_bgcolor="#1a1f2e",
                          font_color="#e2e8f0", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        year_data = df.groupby('Year').agg(
            Avg_Price=('Selling_Price','mean'),
            Count=('Selling_Price','count')
        ).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=year_data['Year'], y=year_data['Count'],
                             name='Number of Cars',
                             marker_color='rgba(102,126,234,0.3)'), secondary_y=False)
        fig.add_trace(go.Scatter(x=year_data['Year'], y=year_data['Avg_Price'],
                                  name='Average Price',
                                  line=dict(color='#38ef7d', width=3),
                                  mode='lines+markers'), secondary_y=True)
        fig.update_layout(template="plotly_dark",
                          title="📅 Year-wise Cars Count & Average Price",
                          plot_bgcolor="#1a1f2e", paper_bgcolor="#1a1f2e", font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        owner_avg = df.groupby('Owner')['Selling_Price'].mean().reset_index()
        owner_avg['Owner_Label'] = owner_avg['Owner'].map(
            {0:"1st Owner",1:"2nd Owner",2:"3rd Owner",3:"4th+ Owner"})
        fig = px.funnel(owner_avg, x="Selling_Price", y="Owner_Label",
                        color_discrete_sequence=["#667eea"],
                        template="plotly_dark",
                        title="👤 Average Price by Number of Owners")
        fig.update_layout(plot_bgcolor="#1a1f2e", paper_bgcolor="#1a1f2e", font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📉 Depreciation Analysis")
    df['Depreciation_Pct'] = ((df['Present_Price'] - df['Selling_Price']) / df['Present_Price']) * 100
    fig = px.scatter(df, x="Car_Age", y="Depreciation_Pct",
                     color="Fuel_Type", size="Present_Price",
                     color_discrete_sequence=["#667eea","#11998e","#f093fb"],
                     template="plotly_dark",
                     title="Car Age vs Depreciation % (bubble size = Present Price)")
    fig.update_layout(plot_bgcolor="#1a1f2e", paper_bgcolor="#1a1f2e", font_color="#e2e8f0")
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════
#   PAGE 5 — CAR COMPARATOR
# ════════════════════════════════════════════════════════════
elif page == "compare":
    st.markdown("## 🚘 Car Comparator")
    st.markdown("<p>Select two cars to compare them side by side.</p>", unsafe_allow_html=True)

    car_names = sorted(df['Car_Name'].unique())
    c1, c2    = st.columns(2)
    with c1:
        st.markdown("### 🔵 Car A")
        car_a  = st.selectbox("Select Car A", car_names, index=0, key="car_a")
        data_a = df[df['Car_Name'] == car_a].iloc[0]
    with c2:
        st.markdown("### 🔴 Car B")
        car_b  = st.selectbox("Select Car B", car_names, index=min(5,len(car_names)-1), key="car_b")
        data_b = df[df['Car_Name'] == car_b].iloc[0]

    st.markdown("---")
    st.markdown("### 📊 Side-by-Side Comparison")

    labels = ["Selling Price (L)","Present Price (L)","Driven KMs","Car Age (yr)","Owner"]
    vals_a = [data_a['Selling_Price'], data_a['Present_Price'],
              data_a['Driven_kms'],    data_a['Car_Age'], data_a['Owner']]
    vals_b = [data_b['Selling_Price'], data_b['Present_Price'],
              data_b['Driven_kms'],    data_b['Car_Age'], data_b['Owner']]

    m_cols = st.columns(5)
    for i, (lbl, va, vb) in enumerate(zip(labels, vals_a, vals_b)):
        with m_cols[i]:
            st.markdown(f"**{lbl}**")
            color_a = "#38ef7d" if va <= vb else "#ff6b6b"
            color_b = "#38ef7d" if vb <= va else "#ff6b6b"
            st.markdown(f"<p style='color:{color_a};font-size:1.3rem;font-weight:700'>{va:,.0f}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:{color_b};font-size:1.3rem;font-weight:700'>{vb:,.0f}</p>", unsafe_allow_html=True)

    max_vals = [df['Selling_Price'].max(), df['Present_Price'].max(),
                df['Driven_kms'].max(),    df['Car_Age'].max(), 3]
    norm_a = [v/m for v,m in zip(vals_a, max_vals)]
    norm_b = [v/m for v,m in zip(vals_b, max_vals)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=norm_a+[norm_a[0]], theta=labels+[labels[0]],
                                   fill='toself', name=car_a.upper(), line_color='#667eea'))
    fig.add_trace(go.Scatterpolar(r=norm_b+[norm_b[0]], theta=labels+[labels[0]],
                                   fill='toself', name=car_b.upper(), line_color='#f093fb'))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1]), bgcolor="#1a1f2e"),
        paper_bgcolor="#1a1f2e", font_color="#e2e8f0",
        title=f"🔵 {car_a.upper()}  vs  🔴 {car_b.upper()}"
    )
    st.plotly_chart(fig, use_container_width=True)

    dep_a  = ((data_a['Present_Price']-data_a['Selling_Price'])/data_a['Present_Price'])*100
    dep_b  = ((data_b['Present_Price']-data_b['Selling_Price'])/data_b['Present_Price'])*100
    winner = car_a if dep_a < dep_b else car_b
    st.markdown(f"""
    <div class='price-card'>
        <h2>🏆 Better Value: {winner.upper()}</h2>
        <p>{car_a.upper()} Depreciation: {dep_a:.1f}% &nbsp;|&nbsp;
           {car_b.upper()} Depreciation: {dep_b:.1f}%</p>
        <p>Lower depreciation means better resale value.</p>
    </div>""", unsafe_allow_html=True)