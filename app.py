import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# =====================================================================
# 1. CẤU HÌNH TRANG STREAMLIT
# =====================================================================
st.set_page_config(
    page_title="Dự đoán Ý định Mua hàng E-Commerce | UEL",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS cho giao diện hiện đại, chuyên nghiệp
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        padding: 24px 28px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
        color: #f8fafc;
    }
    
    .main-header p {
        font-size: 0.95rem;
        color: #cbd5e1;
        margin: 6px 0 0 0;
    }
    
    .badge-uel {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.25);
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 10px;
        color: #e0f2fe;
    }
    
    .card-kpi {
        background: white;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        margin-bottom: 12px;
    }
    
    .card-kpi h4 {
        margin: 0;
        font-size: 0.8rem;
        text-transform: uppercase;
        color: #64748b;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    
    .card-kpi .value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 4px;
    }
    
    .card-kpi .subtext {
        font-size: 0.8rem;
        color: #94a3b8;
    }
    
    .pred-box-success {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        color: #065f46;
        margin: 15px 0;
    }
    
    .pred-box-danger {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        color: #991b1b;
        margin: 15px 0;
    }
    
    .recom-card {
        background: #f8fafc;
        border-left: 4px solid #0284c7;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin-bottom: 10px;
        font-size: 0.92rem;
    }
    
    .recom-card strong {
        color: #0369a1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# 2. XÁC ĐỊNH ĐƯỜNG DẪN DỮ LIỆU & TẢI MÔ HÌNH
# =====================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
POSSIBLE_DIRS = [
    CURRENT_DIR,
    r"D:\OneDrive - uel.edu.vn\ML\model",
    os.path.join(r"D:\OneDrive - uel.edu.vn\ML", "model"),
]

def find_asset_dir():
    for d in POSSIBLE_DIRS:
        if os.path.exists(d):
            m_path = os.path.join(d, "random_forest_model.pkl")
            c_path = os.path.join(d, "model_config.json")
            if os.path.exists(m_path) and os.path.exists(c_path):
                return d
    return CURRENT_DIR

DEFAULT_DIR = find_asset_dir()

@st.cache_resource
def load_ml_pipeline(model_dir: str):
    model_path = os.path.join(model_dir, "random_forest_model.pkl")
    cfg_path = os.path.join(model_dir, "model_config.json")
    
    if not os.path.exists(model_path):
        # Fallback thử tên có dấu space "(1)" nếu cần
        alt = os.path.join(model_dir, "random_forest_model (1).pkl")
        if os.path.exists(alt):
            model_path = alt
        else:
            return None, None, f"Không tìm thấy file mô hình: {model_path}"
            
    if not os.path.exists(cfg_path):
        return None, None, f"Không tìm thấy file cấu hình: {cfg_path}"
        
    try:
        model = joblib.load(model_path)
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return model, cfg, None
    except Exception as e:
        return None, None, str(e)

@st.cache_data
def load_cleaned_data(model_dir: str):
    data_path = os.path.join(model_dir, "online_shoppers_cleaned.csv")
    if not os.path.exists(data_path):
        alt = os.path.join(os.path.dirname(model_dir), "online_shoppers_intention.csv")
        if os.path.exists(alt):
            data_path = alt
        else:
            return None, f"Không tìm thấy file dữ liệu: {data_path}"
    try:
        df = pd.read_csv(data_path)
        return df, None
    except Exception as e:
        return None, str(e)

# =====================================================================
# 3. SIDEBAR: THÔNG TIN & THIẾT LẬP
# =====================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=64)
    st.markdown("### 🛒 E-Commerce Predictor")
    st.caption("Mô hình Machine Learning Dự đoán Khả năng Mua hàng")
    st.divider()
    
    # Cho phép người dùng tùy chọn hoặc xác nhận thư mục
    chosen_dir = st.text_input(
        "Thư mục chứa Model & Data:",
        value=DEFAULT_DIR,
        help="Đường dẫn chứa random_forest_model.pkl, model_config.json và online_shoppers_cleaned.csv",
    )
    
    model, config, err_model = load_ml_pipeline(chosen_dir)
    df_clean, err_data = load_cleaned_data(chosen_dir)
    
    if err_model:
        st.error(f"⚠️ Lỗi tải mô hình: {err_model}")
    else:
        st.success(f"✅ Mô hình: **{config.get('model_name', 'Random Forest')}**")
    
    st.divider()
    
    # Điều chỉnh Ngưỡng phân loại (Decision Threshold)
    default_thresh = float(config.get("threshold", 0.30)) if config else 0.30
    user_threshold = st.slider(
        "Ngưỡng xác suất phân loại (Threshold):",
        min_value=0.05,
        max_value=0.95,
        value=default_thresh,
        step=0.05,
        help="Mặc định theo nghiên cứu là 0.30 nhằm tối ưu hóa độ nhạy (Recall = 95%) để không bỏ sót khách mua tiềm năng trong tập dữ liệu mất cân bằng.",
    )
    
    if user_threshold < 0.30:
        st.info("💡 **Chế độ Nhạy cao (High Recall)**: Bắt trọn tối đa người mua, chấp nhận có thể nhầm lẫn nhẹ khách xem bình thường.")
    elif user_threshold > 0.50:
        st.warning("⚠️ **Chế độ Thận trọng (High Precision)**: Chỉ dự đoán 'Mua' khi xác suất cực kỳ chắc chắn.")
    else:
        st.success("🎯 **Ngưỡng Tối ưu (0.30)**: Cân bằng tốt nhất giữa Precision và Recall theo kết quả thực nghiệm.")
        
    st.divider()
    st.markdown(
        """
        **Đồ án Machine Learning**  
        🏛️ Trường ĐH Kinh tế - Luật (UEL)  
        📂 UCI Dataset: *Online Shoppers*  
        🤖 Thuật toán: *Random Forest Pipeline*  
        """
    )

# =====================================================================
# 4. HEADER CHÍNH
# =====================================================================
st.markdown(
    """
    <div class="main-header">
        <span class="badge-uel">UEL - KHOA HỆ THỐNG THÔNG TIN / TOÁN KINH TẾ</span>
        <h1>🛒 Hệ Thống Dự Đoán Khả Năng Mua Hàng Trực Tuyến</h1>
        <p>Phân tích hành vi phiên truy cập (Session-based E-Commerce) & Dự đoán giao dịch bằng Random Forest Pipeline</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None or df_clean is None:
    st.error("Không thể tiếp tục vì chưa tải được Model hoặc Dataset. Vui lòng kiểm tra lại đường dẫn ở Sidebar.")
    st.stop()

# =====================================================================
# 5. TABS ĐIỀU HƯỚNG
# =====================================================================
tab_pred, tab_batch, tab_eda, tab_model, tab_about = st.tabs(
    [
        "🎯 Dự đoán Phiên Đơn Lẻ",
        "📁 Dự đoán Hàng Loạt (Batch)",
        "📊 Trực quan Dữ liệu (EDA)",
        "🧠 Giải thích & Tối ưu Ngưỡng",
        "📑 Giới thiệu & Tài liệu",
    ]
)

# =====================================================================
# TAB 1: DỰ ĐOÁN PHIÊN ĐƠN LẺ (WHAT-IF SIMULATION)
# =====================================================================
with tab_pred:
    st.markdown("### 🎯 Dự đoán Ý định Mua hàng cho một Phiên truy cập mới")
    st.write(
        "Nhập các chỉ số tương tác và bối cảnh của phiên truy cập hoặc chọn hồ sơ mẫu để kiểm tra kết quả ngay lập tức."
    )
    
    # Profile Presets
    col_pre1, col_pre2, col_pre3, col_pre4 = st.columns(4)
    preset_choice = "Custom"
    
    with col_pre1:
        if st.button("🌟 Khách tiềm năng (Sắp chốt đơn)", use_container_width=True):
            st.session_state["p_admin"] = 3
            st.session_state["p_admin_d"] = 145.0
            st.session_state["p_info"] = 1
            st.session_state["p_info_d"] = 60.0
            st.session_state["p_prod"] = 45
            st.session_state["p_prod_d"] = 1950.0
            st.session_state["p_bounce"] = 0.005
            st.session_state["p_exit"] = 0.018
            st.session_state["p_pageval"] = 42.5
            st.session_state["p_spec"] = 0.0
            st.session_state["p_month"] = "Nov"
            st.session_state["p_os"] = 2
            st.session_state["p_browser"] = 2
            st.session_state["p_region"] = 1
            st.session_state["p_traffic"] = 2
            st.session_state["p_visitor"] = "New_Visitor"
            st.session_state["p_weekend"] = True
            st.rerun()

    with col_pre2:
        if st.button("🚶 Khách xem lướt (Thoát sớm)", use_container_width=True):
            st.session_state["p_admin"] = 0
            st.session_state["p_admin_d"] = 0.0
            st.session_state["p_info"] = 0
            st.session_state["p_info_d"] = 0.0
            st.session_state["p_prod"] = 2
            st.session_state["p_prod_d"] = 35.0
            st.session_state["p_bounce"] = 0.10
            st.session_state["p_exit"] = 0.14
            st.session_state["p_pageval"] = 0.0
            st.session_state["p_spec"] = 0.0
            st.session_state["p_month"] = "May"
            st.session_state["p_os"] = 1
            st.session_state["p_browser"] = 1
            st.session_state["p_region"] = 3
            st.session_state["p_traffic"] = 3
            st.session_state["p_visitor"] = "Returning_Visitor"
            st.session_state["p_weekend"] = False
            st.rerun()

    with col_pre3:
        if st.button("🔍 Khách tìm thông tin (Support)", use_container_width=True):
            st.session_state["p_admin"] = 4
            st.session_state["p_admin_d"] = 210.0
            st.session_state["p_info"] = 5
            st.session_state["p_info_d"] = 320.0
            st.session_state["p_prod"] = 1
            st.session_state["p_prod_d"] = 20.0
            st.session_state["p_bounce"] = 0.02
            st.session_state["p_exit"] = 0.05
            st.session_state["p_pageval"] = 0.0
            st.session_state["p_spec"] = 0.2
            st.session_state["p_month"] = "Mar"
            st.session_state["p_os"] = 3
            st.session_state["p_browser"] = 2
            st.session_state["p_region"] = 2
            st.session_state["p_traffic"] = 4
            st.session_state["p_visitor"] = "Returning_Visitor"
            st.session_state["p_weekend"] = False
            st.rerun()

    with col_pre4:
        if st.button("🔄 Đặt lại giá trị Trung bình", use_container_width=True):
            st.session_state["p_admin"] = int(df_clean["Administrative"].median())
            st.session_state["p_admin_d"] = float(df_clean["Administrative_Duration"].median())
            st.session_state["p_info"] = int(df_clean["Informational"].median())
            st.session_state["p_info_d"] = float(df_clean["Informational_Duration"].median())
            st.session_state["p_prod"] = int(df_clean["ProductRelated"].median())
            st.session_state["p_prod_d"] = float(df_clean["ProductRelated_Duration"].median())
            st.session_state["p_bounce"] = float(df_clean["BounceRates"].median())
            st.session_state["p_exit"] = float(df_clean["ExitRates"].median())
            st.session_state["p_pageval"] = 0.0
            st.session_state["p_spec"] = 0.0
            st.session_state["p_month"] = "May"
            st.session_state["p_os"] = 2
            st.session_state["p_browser"] = 2
            st.session_state["p_region"] = 1
            st.session_state["p_traffic"] = 2
            st.session_state["p_visitor"] = "Returning_Visitor"
            st.session_state["p_weekend"] = False
            st.rerun()

    st.markdown("---")
    
    # Form nhập liệu phân theo 4 nhóm logic
    form_col1, form_col2 = st.columns(2)
    
    with form_col1:
        st.markdown("#### 🛍️ 1. Tương tác Trang Sản phẩm (Quan trọng)")
        prod_related = st.number_input(
            "Số trang liên quan đến Sản phẩm (ProductRelated):",
            min_value=0,
            max_value=1000,
            value=st.session_state.get("p_prod", 20),
            help="Số lượt xem các trang sản phẩm, danh mục hàng hóa.",
        )
        prod_duration = st.number_input(
            "Tổng thời gian xem sản phẩm (giây) (ProductRelated_Duration):",
            min_value=0.0,
            max_value=100000.0,
            value=float(st.session_state.get("p_prod_d", 800.0)),
            step=10.0,
            help="Tổng thời lượng tương tác với trang sản phẩm trong phiên.",
        )
        
        st.markdown("#### 💎 2. Chỉ số Chất lượng & Giá trị Phiên")
        page_values = st.number_input(
            "⭐ Giá trị trang (PageValues - Điểm then chốt):",
            min_value=0.0,
            max_value=500.0,
            value=float(st.session_state.get("p_pageval", 5.0)),
            step=0.5,
            help="Điểm số giá trị trung bình do Google Analytics tính dựa trên các trang được xem trước khi hoàn tất mua hàng. Đây là biến có tương quan mạnh nhất (0.49) với Revenue!",
        )
        
        col_rates1, col_rates2 = st.columns(2)
        with col_rates1:
            bounce_rate = st.slider(
                "Tỷ lệ thoát (BounceRates):",
                min_value=0.0,
                max_value=0.20,
                value=float(st.session_state.get("p_bounce", 0.01)),
                step=0.005,
                format="%.3f",
                help="Tỷ lệ người dùng rời đi ngay từ trang đầu tiên.",
            )
        with col_rates2:
            exit_rate = st.slider(
                "Tỷ lệ rời khỏi (ExitRates):",
                min_value=0.0,
                max_value=0.20,
                value=float(st.session_state.get("p_exit", 0.03)),
                step=0.005,
                format="%.3f",
                help="Tỷ lệ phiên kết thúc tại trang đang xét.",
            )

    with form_col2:
        st.markdown("#### 🏢 3. Trang Hành chính & Thông tin")
        col_adm1, col_adm2 = st.columns(2)
        with col_adm1:
            admin = st.number_input(
                "Số trang Administrative:",
                min_value=0,
                max_value=100,
                value=st.session_state.get("p_admin", 1),
                help="Trang quản lý tài khoản, đơn hàng, giỏ hàng,...",
            )
        with col_adm2:
            admin_duration = st.number_input(
                "Thời gian Admin (giây):",
                min_value=0.0,
                max_value=10000.0,
                value=float(st.session_state.get("p_admin_d", 30.0)),
                step=5.0,
            )

        col_inf1, col_inf2 = st.columns(2)
        with col_inf1:
            info = st.number_input(
                "Số trang Informational:",
                min_value=0,
                max_value=100,
                value=st.session_state.get("p_info", 0),
                help="Trang về chính sách, địa chỉ, liên hệ,...",
            )
        with col_inf2:
            info_duration = st.number_input(
                "Thời gian Info (giây):",
                min_value=0.0,
                max_value=10000.0,
                value=float(st.session_state.get("p_info_d", 0.0)),
                step=5.0,
            )

        st.markdown("#### 🌐 4. Bối cảnh & Đặc tính Khách truy cập")
        col_ctx1, col_ctx2 = st.columns(2)
        
        months_list = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        with col_ctx1:
            cur_month = st.session_state.get("p_month", "Nov")
            m_idx = months_list.index(cur_month) if cur_month in months_list else 8
            month = st.selectbox("Tháng truy cập (Month):", options=months_list, index=m_idx)
            
            special_day = st.slider(
                "Gần ngày lễ (SpecialDay):",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.get("p_spec", 0.0)),
                step=0.2,
                help="Mức độ gần các ngày lễ như Valentine, Ngày của Mẹ,... (0.0: bình thường, 1.0: sát ngày lễ).",
            )

        with col_ctx2:
            visitor_list = ["Returning_Visitor", "New_Visitor", "Other"]
            cur_vis = st.session_state.get("p_visitor", "Returning_Visitor")
            v_idx = visitor_list.index(cur_vis) if cur_vis in visitor_list else 0
            visitor_type = st.selectbox("Loại khách (VisitorType):", options=visitor_list, index=v_idx)
            
            weekend = st.checkbox("Truy cập vào Cuối tuần (Weekend)?", value=st.session_state.get("p_weekend", False))

        with st.expander("⚙️ Thông số Kỹ thuật & Kênh truy cập (Hệ điều hành, Trình duyệt, Khu vực)"):
            c_tech1, c_tech2, c_tech3, c_tech4 = st.columns(4)
            with c_tech1:
                os_val = st.number_input("HĐH (OS 1-8):", min_value=1, max_value=8, value=int(st.session_state.get("p_os", 2)))
            with c_tech2:
                browser_val = st.number_input("Browser (1-13):", min_value=1, max_value=13, value=int(st.session_state.get("p_browser", 2)))
            with c_tech3:
                region_val = st.number_input("Khu vực (1-9):", min_value=1, max_value=9, value=int(st.session_state.get("p_region", 1)))
            with c_tech4:
                traffic_val = st.number_input("Nguồn Traffic (1-20):", min_value=1, max_value=20, value=int(st.session_state.get("p_traffic", 2)))

    # Tạo DataFrame dự đoán
    input_data = pd.DataFrame(
        [
            {
                "Administrative": int(admin),
                "Administrative_Duration": float(admin_duration),
                "Informational": int(info),
                "Informational_Duration": float(info_duration),
                "ProductRelated": int(prod_related),
                "ProductRelated_Duration": float(prod_duration),
                "BounceRates": float(bounce_rate),
                "ExitRates": float(exit_rate),
                "PageValues": float(page_values),
                "SpecialDay": float(special_day),
                "Month": str(month),
                "OperatingSystems": int(os_val),
                "Browser": int(browser_val),
                "Region": int(region_val),
                "TrafficType": int(traffic_val),
                "VisitorType": str(visitor_type),
                "Weekend": bool(weekend),
            }
        ]
    )

    st.markdown("---")
    btn_predict = st.button("🚀 Chạy Dự Đoán Phiên Này", type="primary", use_container_width=True)

    if btn_predict:
        # Dự đoán xác suất
        probas = model.predict_proba(input_data)[0]
        prob_no_purchase = probas[0]
        prob_purchase = probas[1]
        is_buyer = prob_purchase >= user_threshold

        res_col1, res_col2 = st.columns([1.1, 0.9])

        with res_col1:
            if is_buyer:
                st.markdown(
                    f"""
                    <div class="pred-box-success">
                        <h2 style="margin:0; font-size: 1.8rem;">🎉 DỰ ĐOÁN: CÓ KHẢ NĂNG MUA HÀNG</h2>
                        <h3 style="margin:8px 0; font-size: 2.3rem; font-weight:800; color:#059669;">
                            {prob_purchase*100:.1f}%
                        </h3>
                        <p style="margin:0; font-size:1rem; font-weight:600;">
                            Xác suất vượt ngưỡng phân loại ({user_threshold*100:.0f}%) &rarr; Đánh dấu: <strong>Revenue = True</strong>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="pred-box-danger">
                        <h2 style="margin:0; font-size: 1.8rem;">⚠️ DỰ ĐOÁN: ÍT KHẢ NĂNG MUA HÀNG</h2>
                        <h3 style="margin:8px 0; font-size: 2.3rem; font-weight:800; color:#dc2626;">
                            {prob_purchase*100:.1f}%
                        </h3>
                        <p style="margin:0; font-size:1rem; font-weight:600;">
                            Xác suất nằm dưới ngưỡng phân loại ({user_threshold*100:.0f}%) &rarr; Đánh dấu: <strong>Revenue = False</strong>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Gauge Chart biểu thị xác suất
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob_purchase * 100,
                    number={"suffix": "%", "font": {"size": 28, "color": "#0f172a"}},
                    title={"text": "Xác Suất Phát Sinh Giao Dịch", "font": {"size": 16}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                        "bar": {"color": "#10b981" if is_buyer else "#ef4444", "thickness": 0.3},
                        "bgcolor": "white",
                        "borderwidth": 1,
                        "bordercolor": "#cbd5e1",
                        "steps": [
                            {"range": [0, user_threshold * 100], "color": "#fee2e2"},
                            {"range": [user_threshold * 100, 100], "color": "#dcfce7"},
                        ],
                        "threshold": {
                            "line": {"color": "#0284c7", "width": 4},
                            "thickness": 0.8,
                            "value": user_threshold * 100,
                        },
                    },
                )
            )
            fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.markdown("#### 💡 Khuyến nghị Hành động Kinh doanh & Marketing")
            
            if is_buyer:
                st.markdown(
                    """
                    <div class="recom-card">
                        <strong>🔥 Ưu tiên Chốt đơn Ngay:</strong> Khách hàng đang thể hiện mức độ quan tâm vượt trội (tỷ lệ chuyển đổi cao).
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if visitor_type == "New_Visitor":
                    st.markdown(
                        """
                        <div class="recom-card">
                            <strong>🎁 Khách hàng mới tiềm năng:</strong> Tự động kích hoạt pop-up tặng mã giảm giá chào mừng <em>(Welcome Voucher 10% + Miễn phí vận chuyển)</em> để chốt đơn ngay tại phiên đầu tiên.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="recom-card">
                            <strong>⭐ Tri ân Khách quay lại:</strong> Áp dụng điểm tích lũy hoặc hiển thị sản phẩm tương thích với lịch sử xem để tăng giá trị đơn hàng trung bình (AOV).
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                if page_values > 20:
                    st.markdown(
                        """
                        <div class="recom-card">
                            <strong>⚡ Đẩy nhanh Thanh toán:</strong> Hiển thị nút "Mua ngay trong 1-Click" và gợi ý phương thức thanh toán ví điện tử/thẻ tiện lợi nhất.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    """
                    <div class="recom-card">
                        <strong>🛡️ Phòng ngừa Thoát trang (Churn Prevention):</strong> Khách chưa có ý định mua rõ ràng hoặc đang gặp rào cản thông tin.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if exit_rate > 0.05 or bounce_rate > 0.05:
                    st.markdown(
                        """
                        <div class="recom-card">
                            <strong>⚠️ Kích hoạt Exit-Intent Popup:</strong> Khi con trỏ chuột rời khỏi khung màn hình, hiển thị pop-up giữ chân khách cùng danh sách <em>Top 5 sản phẩm bán chạy nhất tháng</em> hoặc hỗ trợ tư vấn qua Chatbot.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                if page_values == 0:
                    st.markdown(
                        """
                        <div class="recom-card">
                            <strong>📢 Tái tiếp thị (Remarketing):</strong> Thu thập cookie / email để đưa vào chiến dịch Facebook/Google Retargeting nhắc nhở danh mục sản phẩm đã xem.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Phân tích đóng góp đặc trưng (Perturbation Attribution)
        st.markdown("#### 🔬 Đóng góp của từng Yếu tố vào Xác suất Mua hàng (Local Explainability)")
        st.caption(
            "Biểu đồ phân tích mức độ tác động của từng đặc trưng trong phiên này so với mức chuẩn bình quân của toàn website."
        )

        with st.spinner("Đang tính toán mức độ tác động của từng biến..."):
            # Lấy baseline từ tập dữ liệu
            baseline_row = {}
            X_data = df_clean.drop("Revenue", axis=1)
            for c in X_data.columns:
                if c in ["Month", "VisitorType"]:
                    baseline_row[c] = X_data[c].mode()[0]
                elif c in ["Weekend"]:
                    baseline_row[c] = bool(X_data[c].mode()[0])
                elif c in ["OperatingSystems", "Browser", "Region", "TrafficType"]:
                    baseline_row[c] = int(X_data[c].mode()[0])
                elif np.issubdtype(X_data[c].dtype, np.integer):
                    baseline_row[c] = int(X_data[c].median())
                else:
                    baseline_row[c] = float(X_data[c].median())

            base_df = pd.DataFrame([baseline_row])
            base_prob = model.predict_proba(base_df)[0, 1]

            impacts = []
            for col_name in input_data.columns:
                perturbed = input_data.copy()
                perturbed[col_name] = baseline_row[col_name]
                p_prob = model.predict_proba(perturbed)[0, 1]
                impact = prob_purchase - p_prob
                if abs(impact) > 0.001:
                    impacts.append({"Đặc trưng": col_name, "Tác động vào Xác suất": impact})

            if impacts:
                impact_df = pd.DataFrame(impacts).sort_values("Tác động vào Xác suất", key=abs, ascending=True).tail(10)
                colors = ["#10b981" if v > 0 else "#ef4444" for v in impact_df["Tác động vào Xác suất"]]
                fig_imp = go.Figure(
                    go.Bar(
                        x=impact_df["Tác động vào Xác suất"],
                        y=impact_df["Đặc trưng"],
                        orientation="h",
                        marker_color=colors,
                        text=[f"{v:+.3f}" for v in impact_df["Tác động vào Xác suất"]],
                        textposition="auto",
                    )
                )
                fig_imp.update_layout(
                    title="Top yếu tố làm Tăng (Xanh lá) hoặc Giảm (Đỏ) xác suất mua hàng trong phiên này",
                    xaxis_title="Thay đổi xác suất (so với khi giá trị ở mức trung bình)",
                    height=360,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Các thuộc tính của phiên này đang ở mức gần sát với mức trung bình của toàn website.")

# =====================================================================
# TAB 2: DỰ ĐOÁN HÀNG LOẠT (BATCH CSV)
# =====================================================================
with tab_batch:
    st.markdown("### 📁 Dự đoán Khả năng Mua hàng Hàng loạt từ File CSV")
    st.write(
        "Tải lên tệp CSV chứa danh sách các phiên truy cập của khách hàng để mô hình phân loại và chấm điểm xác suất chuyển đổi cho toàn bộ tập dữ liệu."
    )
    
    col_b1, col_b2 = st.columns([1, 1])
    with col_b1:
        # Nút lấy mẫu 20 dòng từ dataset
        if st.button("📥 Nạp nhanh 20 phiên ngẫu nhiên từ Dataset sạch để kiểm thử", use_container_width=True):
            sample_df = df_clean.sample(20, random_state=42).copy()
            st.session_state["batch_input_df"] = sample_df
            st.success("Đã nạp 20 phiên mẫu thành công! Bấm 'Tiến hành Dự đoán' bên dưới.")

    with col_b2:
        # Tải template CSV mẫu
        sample_template = df_clean.drop("Revenue", axis=1).iloc[:5]
        csv_template = sample_template.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📄 Tải File CSV Mẫu (Template Header)",
            data=csv_template,
            file_name="online_shoppers_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    uploaded_file = st.file_uploader("Hoặc tải lên tệp CSV của bạn:", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.session_state["batch_input_df"] = uploaded_df
            st.success(f"Đã đọc file: {uploaded_file.name} ({len(uploaded_df)} dòng)")
        except Exception as e:
            st.error(f"Lỗi khi đọc file CSV: {e}")

    batch_df = st.session_state.get("batch_input_df", None)

    if batch_df is not None:
        st.markdown("#### 📋 Xem trước Dữ liệu Đầu vào (5 dòng đầu)")
        st.dataframe(batch_df.head(5), use_container_width=True)

        # Kiểm tra các cột bắt buộc
        req_features = config.get("features", [])
        missing_cols = [c for c in req_features if c not in batch_df.columns]

        if missing_cols:
            st.error(f"❌ File CSV thiếu các cột bắt buộc sau: {', '.join(missing_cols)}")
        else:
            if st.button("⚡ Tiến hành Chấm Điểm & Dự Đoán Hàng Loạt", type="primary", use_container_width=True):
                with st.spinner("Đang thực hiện phân loại qua Random Forest Pipeline..."):
                    features_df = batch_df[req_features].copy()
                    
                    # Đảm bảo đúng định dạng kiểu dữ liệu
                    features_df["Month"] = features_df["Month"].astype(str)
                    features_df["VisitorType"] = features_df["VisitorType"].astype(str)
                    features_df["Weekend"] = features_df["Weekend"].astype(bool)

                    probabilities = model.predict_proba(features_df)[:, 1]
                    preds = (probabilities >= user_threshold).astype(int)

                    results_df = batch_df.copy()
                    results_df["Xac_Suat_Mua"] = np.round(probabilities * 100, 2)
                    results_df["Du_Doan_Mua"] = ["Có Mua (True)" if p == 1 else "Không Mua (False)" for p in preds]
                    
                    # Phân loại phân khúc tiềm năng
                    def assign_segment(p):
                        if p >= 0.70:
                            return "1. Tiềm năng Rất cao (Hot Lead)"
                        elif p >= user_threshold:
                            return "2. Tiềm năng Khá (Warm Lead)"
                        elif p >= 0.15:
                            return "3. Đang cân nhắc (Cool Lead)"
                        else:
                            return "4. Rời đi/Vãng lai (Cold Lead)"

                    results_df["Phan_Khuc"] = [assign_segment(p) for p in probabilities]

                    # Thống kê KPI
                    total_sessions = len(results_df)
                    predicted_buyers = sum(preds)
                    conversion_rate = (predicted_buyers / total_sessions) * 100

                    st.markdown("---")
                    st.markdown("#### 📊 Tổng Hợp Kết Quả Phân Tích")
                    k1, k2, k3, k4 = st.columns(4)
                    with k1:
                        st.markdown(
                            f"""
                            <div class="card-kpi">
                                <h4>Tổng Phiên Phân Tích</h4>
                                <div class="value">{total_sessions:,}</div>
                                <div class="subtext">Phiên truy cập website</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with k2:
                        st.markdown(
                            f"""
                            <div class="card-kpi">
                                <h4>Số Phiên Mua Hàng</h4>
                                <div class="value" style="color:#10b981;">{predicted_buyers:,}</div>
                                <div class="subtext">Vượt ngưỡng {user_threshold*100:.0f}%</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with k3:
                        st.markdown(
                            f"""
                            <div class="card-kpi">
                                <h4>Tỷ Lệ Chuyển Đổi Dự Kiến</h4>
                                <div class="value" style="color:#0284c7;">{conversion_rate:.2f}%</div>
                                <div class="subtext">So với chuẩn ~15.6%</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with k4:
                        hot_leads = sum(probabilities >= 0.70)
                        st.markdown(
                            f"""
                            <div class="card-kpi">
                                <h4>Khách Hot Lead (&ge;70%)</h4>
                                <div class="value" style="color:#f59e0b;">{hot_leads:,}</div>
                                <div class="subtext">Khách hàng cần chốt ngay</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Biểu đồ phân khúc và phân phối
                    c_chart1, c_chart2 = st.columns(2)
                    with c_chart1:
                        seg_counts = results_df["Phan_Khuc"].value_counts().reset_index()
                        seg_counts.columns = ["Phân khúc", "Số lượng"]
                        fig_seg = px.pie(
                            seg_counts,
                            names="Phân khúc",
                            values="Số lượng",
                            hole=0.45,
                            title="Tỷ trọng Phân khúc Khách hàng theo Ý định Mua",
                            color_discrete_sequence=["#10b981", "#38bdf8", "#fbbf24", "#94a3b8"],
                        )
                        fig_seg.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
                        st.plotly_chart(fig_seg, use_container_width=True)

                    with c_chart2:
                        fig_hist = px.histogram(
                            results_df,
                            x="Xac_Suat_Mua",
                            nbins=25,
                            title="Phân phối Xác suất Mua hàng của Tập dữ liệu (%)",
                            color_discrete_sequence=["#0284c7"],
                        )
                        fig_hist.add_vline(
                            x=user_threshold * 100,
                            line_dash="dash",
                            line_color="#ef4444",
                            annotation_text=f"Ngưỡng ({user_threshold*100:.0f}%)",
                        )
                        fig_hist.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
                        st.plotly_chart(fig_hist, use_container_width=True)

                    st.markdown("#### 📋 Chi Tiết Kết Quả Chấm Điểm")
                    st.dataframe(results_df, use_container_width=True)

                    # Nút tải file kết quả
                    res_csv = results_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="💾 Tải Toàn Bộ Kết Quả Dự Đoán (.CSV - UTF-8 Excel)",
                        data=res_csv,
                        file_name="du_doan_mua_hang_results.csv",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True,
                    )

# =====================================================================
# TAB 3: TRỰC QUAN DỮ LIỆU & EDA
# =====================================================================
with tab_eda:
    st.markdown("### 📊 Khám phá Dữ liệu & Hành vi Người dùng (EDA Insights)")
    st.write(
        "Bộ dữ liệu sau khi làm sạch bao gồm **12.205 phiên truy cập**, phân tích các yếu tố quyết định chuyển đổi mua sắm."
    )

    # 4 chỉ số tổng quát
    e1, e2, e3, e4 = st.columns(4)
    total_sess = len(df_clean)
    num_rev_true = int(df_clean["Revenue"].sum())
    num_rev_false = total_sess - num_rev_true
    pct_rev_true = (num_rev_true / total_sess) * 100

    with e1:
        st.markdown(
            f"""
            <div class="card-kpi">
                <h4>Tổng Số Phiên Đã Làm Sạch</h4>
                <div class="value">{total_sess:,}</div>
                <div class="subtext">125 bản ghi trùng lặp đã loại bỏ</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with e2:
        st.markdown(
            f"""
            <div class="card-kpi">
                <h4>Phiên Có Mua Hàng (Revenue=True)</h4>
                <div class="value" style="color:#10b981;">{num_rev_true:,}</div>
                <div class="subtext">Tỷ lệ: {pct_rev_true:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with e3:
        st.markdown(
            f"""
            <div class="card-kpi">
                <h4>Phiên Không Mua (Revenue=False)</h4>
                <div class="value" style="color:#ef4444;">{num_rev_false:,}</div>
                <div class="subtext">Tỷ lệ: {100-pct_rev_true:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with e4:
        st.markdown(
            f"""
            <div class="card-kpi">
                <h4>Mức Độ Mất Cân Bằng Lớp</h4>
                <div class="value" style="color:#f59e0b;">5.4 : 1</div>
                <div class="subtext">Lý do cần hiệu chỉnh ngưỡng 0.30</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Biểu đồ EDA 1 & 2
    eda_col1, eda_col2 = st.columns(2)
    with eda_col1:
        # Tỷ lệ chuyển đổi theo Tháng
        month_order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        df_month = df_clean.groupby("Month")["Revenue"].agg(["count", "mean"]).reset_index()
        df_month["mean"] = df_month["mean"] * 100
        # Sắp xếp theo thứ tự tháng
        df_month["month_cat"] = pd.Categorical(df_month["Month"], categories=month_order, ordered=True)
        df_month = df_month.sort_values("month_cat")

        fig_month = px.bar(
            df_month,
            x="Month",
            y="mean",
            text=df_month["mean"].apply(lambda x: f"{x:.1f}%"),
            title="Tỷ lệ Chuyển đổi (%) theo Tháng (Đỉnh cao Tháng 11 - Black Friday)",
            color="mean",
            color_continuous_scale="Viridis",
            labels={"mean": "Tỷ lệ mua (%)", "Month": "Tháng"},
        )
        fig_month.update_layout(height=360, coloraxis_showscale=False)
        st.plotly_chart(fig_month, use_container_width=True)

    with eda_col2:
        # Tỷ lệ chuyển đổi theo Loại Khách (VisitorType)
        df_vis = df_clean.groupby("VisitorType")["Revenue"].agg(["count", "mean"]).reset_index()
        df_vis["mean"] = df_vis["mean"] * 100
        fig_vis = px.bar(
            df_vis,
            x="VisitorType",
            y="mean",
            text=df_vis["mean"].apply(lambda x: f"{x:.1f}%"),
            title="Tỷ lệ Chuyển đổi (%) theo Loại Khách (Khách mới vượt trội: 24.9%)",
            color="VisitorType",
            color_discrete_sequence=["#06b6d4", "#f43f5e", "#8b5cf6"],
            labels={"mean": "Tỷ lệ mua (%)", "VisitorType": "Loại khách"},
        )
        fig_vis.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_vis, use_container_width=True)

    # Biểu đồ EDA 3 & 4
    eda_col3, eda_col4 = st.columns(2)
    with eda_col3:
        # Phân bố PageValues giữa Người mua và Không mua
        fig_pv = px.box(
            df_clean[df_clean["PageValues"] < 100],
            x="Revenue",
            y="PageValues",
            color="Revenue",
            title="So sánh Phân phối Điểm PageValues giữa Nhóm Mua và Không Mua",
            labels={"Revenue": "Có Mua Hàng?", "PageValues": "Điểm Giá trị Trang (PageValues)"},
            color_discrete_map={True: "#10b981", False: "#ef4444"},
        )
        fig_pv.update_layout(height=360)
        st.plotly_chart(fig_pv, use_container_width=True)

    with eda_col4:
        # Tỷ lệ chuyển đổi Cuối tuần vs Ngày thường
        df_wk = df_clean.groupby("Weekend")["Revenue"].agg(["count", "mean"]).reset_index()
        df_wk["mean"] = df_wk["mean"] * 100
        df_wk["Label"] = df_wk["Weekend"].map({True: "Cuối tuần (Weekend)", False: "Ngày thường (Weekday)"})
        fig_wk = px.bar(
            df_wk,
            x="Label",
            y="mean",
            text=df_wk["mean"].apply(lambda x: f"{x:.1f}%"),
            title="Tỷ lệ Chuyển đổi: Cuối tuần (17.4%) vs Ngày thường (14.9%)",
            color="Label",
            color_discrete_sequence=["#3b82f6", "#10b981"],
            labels={"mean": "Tỷ lệ mua (%)", "Label": "Thời điểm"},
        )
        fig_wk.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_wk, use_container_width=True)

# =====================================================================
# TAB 4: GIẢI THÍCH MÔ HÌNH & TỐI ƯU NGƯỠNG (XAI & THRESHOLD LAB)
# =====================================================================
with tab_model:
    st.markdown("### 🧠 Kiến trúc Mô hình & Phòng Thí nghiệm Tối ưu Ngưỡng (Threshold Lab)")
    st.write(
        "Mô hình được huấn luyện bằng thuật toán **Random Forest Tuned** kết hợp bộ tiền xử lý **ColumnTransformer (Passthrough + OneHotEncoder)**."
    )

    # Trích xuất Feature Importance từ Pipeline
    preprocessor = model.named_steps["preprocessor"]
    rf_clf = model.named_steps["model"]

    num_cols = preprocessor.transformers_[0][2]
    cat_cols = preprocessor.transformers_[1][2]
    ohe = preprocessor.named_transformers_["cat"]
    cat_features = ohe.get_feature_names_out(cat_cols)
    all_features = list(num_cols) + list(cat_features)
    importances = rf_clf.feature_importances_

    fi_df = pd.DataFrame({"Feature": all_features, "Importance": importances}).sort_values("Importance", ascending=True)

    col_m1, col_m2 = st.columns([1.1, 0.9])
    with col_m1:
        fig_fi = px.bar(
            fi_df.tail(15),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 15 Đặc trưng Quan trọng Nhất (Feature Importance)",
            color="Importance",
            color_continuous_scale="Blues",
        )
        fig_fi.update_layout(height=450, coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_m2:
        st.markdown("#### 📌 Nhận định Từ Mô Hình")
        st.markdown(
            """
            1. **`PageValues` chiếm ~36.7% trọng số**: Là biến có sức mạnh dự báo áp đảo nhất. Khách hàng ghé thăm những trang có gắn liền với chuyển đổi trước đó có xác suất mua tăng vọt.
            2. **`ExitRates` & `ProductRelated_Duration`**: Thể hiện chiều sâu quan tâm. Khách xem sản phẩm lâu và có tỷ lệ thoát thấp là dấu hiệu tích cực của việc cân nhắc mua.
            3. **`Month_Nov`**: Yếu tố mùa vụ tháng 11 (Black Friday, khuyến mãi lớn) đóng góp đáng kể vào quyết định chốt đơn.
            4. **`VisitorType_New_Visitor`**: Khách mới có tỷ lệ mua thực tế cao hơn (24.9% so với 13.9% của khách cũ), phản ánh chiến dịch quảng cáo thu hút người mới hoạt động tốt.
            """
        )

    st.markdown("---")
    st.markdown("### 🎛️ Phòng Thí Nghiệm Hiệu Năng & Điều Chỉnh Ngưỡng (Threshold Simulator)")
    st.write(
        "Do tập dữ liệu có sự mất cân bằng lớn (chỉ 15.6% là người mua), việc sử dụng ngưỡng mặc định 0.50 sẽ khiến mô hình bỏ sót rất nhiều khách hàng tiềm năng. Dưới đây là phân tích chi tiết hiệu năng theo ngưỡng:"
    )

    # Đánh giá trên tập dữ liệu sạch
    X_full = df_clean.drop("Revenue", axis=1)
    y_full = df_clean["Revenue"].astype(int)

    @st.cache_data
    def get_full_predictions():
        probas = model.predict_proba(X_full)[:, 1]
        return probas

    full_probas = get_full_predictions()
    y_pred_user = (full_probas >= user_threshold).astype(int)

    # Tính toán các chỉ số
    cm = confusion_matrix(y_full, y_pred_user)
    acc = accuracy_score(y_full, y_pred_user)
    prec = precision_score(y_full, y_pred_user)
    rec = recall_score(y_full, y_pred_user)
    f1 = f1_score(y_full, y_pred_user)

    col_sim1, col_sim2 = st.columns([1, 1])

    with col_sim1:
        st.markdown(f"#### Ma Trận Nhầm Lẫn (Confusion Matrix) tại Ngưỡng = **{user_threshold}**")
        cm_labels = ["Không Mua (0)", "Mua Hàng (1)"]
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Dự đoán của Mô hình", y="Thực tế", color="Số phiên"),
            x=cm_labels,
            y=cm_labels,
            text_auto=True,
            color_continuous_scale="Blues",
        )
        fig_cm.update_layout(height=340)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_sim2:
        st.markdown("#### Bảng Chỉ Số Đánh Giá")
        st.markdown(
            f"""
            <div class="card-kpi">
                <h4>Độ Nhạy (Recall - Tỷ lệ Bắt trúng Người mua)</h4>
                <div class="value" style="color:#10b981;">{rec*100:.2f}%</div>
                <div class="subtext">Phát hiện {cm[1,1]:,} / {cm[1,0]+cm[1,1]:,} người mua thực tế</div>
            </div>
            <div class="card-kpi">
                <h4>Độ Chính Xác (Precision)</h4>
                <div class="value" style="color:#0284c7;">{prec*100:.2f}%</div>
                <div class="subtext">Trong số người được đoán mua, có {prec*100:.2f}% mua thật</div>
            </div>
            <div class="card-kpi">
                <h4>F1-Score (Trung hòa Precision & Recall)</h4>
                <div class="value" style="color:#f59e0b;">{f1*100:.2f}%</div>
                <div class="subtext">Độ chính xác tổng quát (Accuracy): {acc*100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =====================================================================
# TAB 5: GIỚI THIỆU ĐỀ TÀI & TÀI LIỆU (ABOUT & DOCS)
# =====================================================================
with tab_about:
    st.markdown("### 📑 Giới thiệu Đề tài Nghiên cứu & Tài liệu Hướng dẫn")
    
    st.markdown(
        """
        #### 1. Tổng quan Đề tài
        - **Tên đề tài**: Dự đoán khả năng phát sinh giao dịch mua hàng của khách truy cập website thương mại điện tử bằng Machine Learning.
        - **Đơn vị thực hiện**: Sinh viên / Nhóm nghiên cứu - Trường Đại học Kinh tế - Luật (UEL), ĐHQG TP.HCM.
        - **Nguồn dữ liệu**: *Online Shoppers Purchasing Intention Dataset* công bố bởi Sakar & Kastro (2018) trên UCI Machine Learning Repository.
        - **Quy mô dữ liệu**: 12.330 phiên ban đầu, sau làm sạch loại bỏ 125 dòng trùng còn **12.205 phiên truy cập**.

        #### 2. Định nghĩa các Biến Đặc trưng (Feature Dictionary)
        """
    )

    feat_dict = [
        {"Tên biến": "Administrative", "Loại": "Số nguyên", "Ý nghĩa": "Số lượng trang thuộc nhóm quản trị/tài khoản/giỏ hàng được xem trong phiên."},
        {"Tên biến": "Administrative_Duration", "Loại": "Số thực", "Ý nghĩa": "Tổng thời gian (giây) khách hàng dành cho các trang Administrative."},
        {"Tên biến": "Informational", "Loại": "Số nguyên", "Ý nghĩa": "Số lượng trang thông tin chính sách, liên hệ, giới thiệu."},
        {"Tên biến": "Informational_Duration", "Loại": "Số thực", "Ý nghĩa": "Tổng thời gian (giây) khách hàng đọc các trang thông tin."},
        {"Tên biến": "ProductRelated", "Loại": "Số nguyên", "Ý nghĩa": "Số lượng trang sản phẩm hoặc danh mục sản phẩm được truy cập."},
        {"Tên biến": "ProductRelated_Duration", "Loại": "Số thực", "Ý nghĩa": "Tổng thời gian (giây) khách hàng xem trang sản phẩm."},
        {"Tên biến": "BounceRates", "Loại": "Số thực (0-0.2)", "Ý nghĩa": "Tỷ lệ phần trăm phiên chỉ xem duy nhất 1 trang rồi rời đi."},
        {"Tên biến": "ExitRates", "Loại": "Số thực (0-0.2)", "Ý nghĩa": "Tỷ lệ phần trăm lượt rời khỏi website tại các trang được xem."},
        {"Tên biến": "PageValues", "Loại": "Số thực", "Ý nghĩa": "Giá trị trung bình của các trang dựa trên việc người dùng ghé thăm trước khi mua."},
        {"Tên biến": "SpecialDay", "Loại": "Số thực (0-1)", "Ý nghĩa": "Mức độ gần với các ngày lễ hoặc dịp đặc biệt."},
        {"Tên biến": "Month", "Loại": "Phân loại", "Ý nghĩa": "Tháng diễn ra phiên truy cập (Feb, Mar, May, June, Jul, Aug, Sep, Oct, Nov, Dec)."},
        {"Tên biến": "OperatingSystems", "Loại": "Phân loại (Số)", "Ý nghĩa": "Mã hệ điều hành thiết bị người dùng sử dụng."},
        {"Tên biến": "Browser", "Loại": "Phân loại (Số)", "Ý nghĩa": "Mã trình duyệt web (Chrome, Safari, Firefox,...)."},
        {"Tên biến": "Region", "Loại": "Phân loại (Số)", "Ý nghĩa": "Mã vùng địa lý nơi người dùng truy cập."},
        {"Tên biến": "TrafficType", "Loại": "Phân loại (Số)", "Ý nghĩa": "Mã kênh nguồn truy cập (Direct, Search, Ads, Social,...)."},
        {"Tên biến": "VisitorType", "Loại": "Phân loại", "Ý nghĩa": "Khách hàng mới (New_Visitor), Khách quay lại (Returning_Visitor), Khác (Other)."},
        {"Tên biến": "Weekend", "Loại": "Boolean", "Ý nghĩa": "Phiên truy cập diễn ra vào cuối tuần (Thứ 7, Chủ Nhật) hay không."},
        {"Tên biến": "Revenue (Target)", "Loại": "Nhị phân (0/1)", "Ý nghĩa": "Biến mục tiêu: 1 - Phát sinh giao dịch (Purchase), 0 - Không mua (No Purchase)."}
    ]
    st.table(pd.DataFrame(feat_dict))

    st.markdown(
        """
        #### 3. Quy trình Xử lý & Mô hình hóa
        1. **Làm sạch**: Kiểm tra thiếu khuyết (0 missing values), phát hiện và loại bỏ 125 bản ghi trùng lặp.
        2. **Pipeline Preprocessing**:
           - 10 biến định lượng: `Passthrough` (giữ nguyên tỷ lệ tự nhiên để mô hình cây học tốt nhất).
           - 7 biến phân loại: `OneHotEncoder(handle_unknown='ignore')` nhằm tránh lỗi khi gặp danh mục mới.
        3. **Thuật toán Phân loại**: `RandomForestClassifier` với siêu tham số tinh chỉnh (`max_depth=30, min_samples_split=5, random_state=42`).
        4. **Tối ưu hóa Ngưỡng**: Thiết lập ngưỡng quyết định tại `0.30` giúp đạt `Recall = 95.1%`, ngăn chặn việc bỏ sót khách hàng tiềm năng.
        """
    )
