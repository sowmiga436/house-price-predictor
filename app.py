import streamlit as st
import pandas as pd
import joblib
import time

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

# ---------- CUSTOM CSS (Dark Mode + Animations) ----------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    h1 {
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        animation: fadeInDown 0.8s ease;
    }
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 16px;
        padding: 20px 25px;
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
        animation: fadeInUp 0.6s ease;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(56, 189, 248, 0.15);
    }
    .stButton > button {
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.3);
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(129, 140, 248, 0.5);
    }
    label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }
    .result-box {
        animation: fadeInUp 0.7s ease;
        background: linear-gradient(135deg, rgba(56,189,248,0.15), rgba(129,140,248,0.15));
        border: 1px solid rgba(56, 189, 248, 0.4);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
    }
    .result-price {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .listing-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: #7dd3fc;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin-top: 8px;
    }
    footer, .stCaption {
        color: #64748b !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")

model = load_model()

# ---------- SAMPLE LISTINGS DATA (zone -> list of properties) ----------
ZONE_LISTINGS = {
    "City Center": [
        {"name": "Compact 2BHK Apartment", "area_sqft": 850,  "bedrooms": 2, "bathrooms": 2, "age_years": 3,  "distance_from_city_km": 1.2, "locality_score": 9.0},
        {"name": "Premium 3BHK Flat",       "area_sqft": 1450, "bedrooms": 3, "bathrooms": 3, "age_years": 2,  "distance_from_city_km": 0.8, "locality_score": 9.5},
        {"name": "Studio Apartment",        "area_sqft": 550,  "bedrooms": 1, "bathrooms": 1, "age_years": 5,  "distance_from_city_km": 1.5, "locality_score": 8.5},
        {"name": "Luxury Penthouse",        "area_sqft": 2400, "bedrooms": 4, "bathrooms": 4, "age_years": 1,  "distance_from_city_km": 0.5, "locality_score": 10.0},
        {"name": "Renovated 2BHK",          "area_sqft": 950,  "bedrooms": 2, "bathrooms": 2, "age_years": 8,  "distance_from_city_km": 2.0, "locality_score": 8.0},
    ],
    "Suburb": [
        {"name": "Family 3BHK House",       "area_sqft": 1800, "bedrooms": 3, "bathrooms": 2, "age_years": 6,  "distance_from_city_km": 8.0,  "locality_score": 6.5},
        {"name": "Cozy 2BHK Flat",          "area_sqft": 1100, "bedrooms": 2, "bathrooms": 2, "age_years": 4,  "distance_from_city_km": 10.0, "locality_score": 6.0},
        {"name": "Independent 4BHK Villa",  "area_sqft": 2600, "bedrooms": 4, "bathrooms": 3, "age_years": 3,  "distance_from_city_km": 7.5,  "locality_score": 7.0},
        {"name": "Budget 1BHK",             "area_sqft": 650,  "bedrooms": 1, "bathrooms": 1, "age_years": 10, "distance_from_city_km": 9.0,  "locality_score": 5.5},
        {"name": "New 3BHK Gated Community","area_sqft": 1650, "bedrooms": 3, "bathrooms": 3, "age_years": 1,  "distance_from_city_km": 11.0, "locality_score": 7.5},
    ],
    "Outskirts": [
        {"name": "Spacious Farmhouse",      "area_sqft": 3200, "bedrooms": 4, "bathrooms": 3, "age_years": 12, "distance_from_city_km": 22.0, "locality_score": 3.0},
        {"name": "Affordable 2BHK",         "area_sqft": 900,  "bedrooms": 2, "bathrooms": 1, "age_years": 7,  "distance_from_city_km": 18.0, "locality_score": 3.5},
        {"name": "Plotted 3BHK House",      "area_sqft": 2000, "bedrooms": 3, "bathrooms": 2, "age_years": 5,  "distance_from_city_km": 20.0, "locality_score": 4.0},
        {"name": "Old 1BHK Cottage",        "area_sqft": 500,  "bedrooms": 1, "bathrooms": 1, "age_years": 25, "distance_from_city_km": 24.0, "locality_score": 2.5},
        {"name": "New Layout 3BHK",         "area_sqft": 1750, "bedrooms": 3, "bathrooms": 2, "age_years": 2,  "distance_from_city_km": 19.0, "locality_score": 4.5},
    ],
}

# ---------- SESSION STATE DEFAULTS ----------
defaults = {
    "area_sqft": 1200, "bedrooms": 3, "bathrooms": 2,
    "age_years": 5, "distance_from_city_km": 5.0, "locality_score": 6.0
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def apply_listing(listing):
    st.session_state["area_sqft"] = listing["area_sqft"]
    st.session_state["bedrooms"] = listing["bedrooms"]
    st.session_state["bathrooms"] = listing["bathrooms"]
    st.session_state["age_years"] = listing["age_years"]
    st.session_state["distance_from_city_km"] = listing["distance_from_city_km"]
    st.session_state["locality_score"] = listing["locality_score"]

# ---------- HEADER ----------
st.title("🏠 House Price Predictor")
st.write("An ML model estimating house prices (in lakhs ₹) using **Random Forest Regression** — built with scikit-learn & Streamlit.")

st.divider()

# ---------- ZONE + LISTING PICKER ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 🌍 Pick a Zone & Property")

pick_col1, pick_col2 = st.columns(2)
with pick_col1:
    zone = st.selectbox("Select Zone", list(ZONE_LISTINGS.keys()))
with pick_col2:
    listing_names = [l["name"] for l in ZONE_LISTINGS[zone]]
    selected_name = st.selectbox("Select Property", listing_names)

selected_listing = next(l for l in ZONE_LISTINGS[zone] if l["name"] == selected_name)

if st.button("📥 Load this property's details", use_container_width=True):
    apply_listing(selected_listing)
    st.rerun()

st.markdown(f'<span class="listing-badge">📍 {zone} · {selected_listing["area_sqft"]} sqft · {selected_listing["bedrooms"]}BHK</span>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- INPUT CARDS (auto-filled, still editable) ----------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📐 Property Details")
    area_sqft = st.slider("Area (sq ft)", 300, 4000, step=50, key="area_sqft")
    bedrooms = st.selectbox("Bedrooms (BHK)", [1, 2, 3, 4, 5], key="bedrooms")
    bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4], key="bathrooms")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📍 Location & Age")
    age_years = st.slider("Age of property (years)", 0, 30, key="age_years")
    distance_from_city_km = st.slider("Distance from city center (km)", 0.5, 25.0, key="distance_from_city_km")
    locality_score = st.slider("Locality demand score (1=low, 10=prime)", 1.0, 10.0, key="locality_score")
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# ---------- PREDICTION ----------
if st.button("✨ Predict Price", type="primary", use_container_width=True):
    with st.spinner("Crunching numbers..."):
        time.sleep(0.6)
        input_df = pd.DataFrame([{
            "area_sqft": st.session_state["area_sqft"],
            "bedrooms": st.session_state["bedrooms"],
            "bathrooms": st.session_state["bathrooms"],
            "age_years": st.session_state["age_years"],
            "distance_from_city_km": st.session_state["distance_from_city_km"],
            "locality_score": st.session_state["locality_score"]
        }])
        prediction = model.predict(input_df)[0]

    st.markdown(f"""
    <div class="result-box">
        <div style="font-size:16px; color:#94a3b8;">Estimated Price</div>
        <div class="result-price">₹ {prediction:.2f} lakhs</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Note: this is a demo model trained on synthetic data for portfolio purposes, not real market data.")

st.divider()
st.caption("Built by Sowmiga S V · scikit-learn · Streamlit")