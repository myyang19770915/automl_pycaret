import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import *
import plotly.express as px
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import io

# 設置頁面配置
st.set_page_config(page_title="AutoML Analysis App", layout="wide")

# 設置標題
st.title("AutoML Analysis App")

# 初始化 session state
if 'setup_done' not in st.session_state:
    st.session_state.setup_done = False
if 'best_model' not in st.session_state:
    st.session_state.best_model = None

def get_feature_importance(model):
    """
    獲取特徵重要性
    
    Parameters:
    model: 訓練好的模型
    
    Returns:
    pandas.DataFrame: 包含特徵重要性的數據框
    """
    try:
        # 使用 plot_model 獲取特徵重要性
        feature_importance_plot = plot_model(model, plot='feature')
        
        return feature_importance_plot
    
    except Exception as e:
        st.error(f"獲取特徵重要性時發生錯誤: {str(e)}")
        return None

# 檔案上傳
uploaded_file = st.file_uploader("請上傳CSV檔案", type=['csv'])

# 在選擇目標變數之後，加入選擇特徵的部分
if uploaded_file is not None:
    # 讀取數據
    df = pd.read_csv(uploaded_file)
    
    # 顯示原始數據
    st.subheader("原始數據預覽")
    st.write(df.head())
    
    # 基本數據信息
    st.subheader("數據基本信息")
    col1, col2 = st.columns(2)
    with col1:
        st.write("數據維度:", df.shape)
    with col2:
        st.write("缺失值數量:", df.isnull().sum().sum())
    
    # 選擇目標變數
    target_column = st.selectbox("請選擇目標變數", df.columns)
    
    # 顯示目標變數的唯一值和分布
    unique_values = df[target_column].value_counts()
    st.write("目標變數分布:")
    st.write(unique_values)
    
    # 選擇特徵變數
    feature_columns = [col for col in df.columns if col != target_column]
    selected_features = st.multiselect(
        "請選擇要使用的特徵變數（不選擇則使用所有特徵）",
        feature_columns,
        default=feature_columns
    )
    
    # 如果沒有選擇特徵，使用所有特徵
    if not selected_features:
        selected_features = feature_columns
    
    # 顯示選擇的特徵數量
    st.write(f"已選擇 {len(selected_features)} 個特徵")
    
    # 設置參數
    with st.expander("高級設置"):
        test_size = st.slider("測試集比例", 0.1, 0.4, 0.2, 0.05)
        normalize = st.checkbox("是否標準化數據", value=True)
        remove_outliers = st.checkbox("是否移除異常值", value=False)
    
    # 開始分析按鈕
    if st.button("開始分析"):
        try:
            with st.spinner('正在初始化模型...'):
                # 只使用選擇的特徵創建新的數據框
                selected_df = df[selected_features + [target_column]]
                
                # 設置進度條
                progress_bar = st.progress(0)
                
                # 初始化 PyCaret
                clf = setup(data=selected_df, 
                          target=target_column,
                          train_size=(1-test_size),
                          normalize=normalize,
                          remove_outliers=remove_outliers,
                          verbose=False,
                          session_id=42)
                
                
                # 獲取目標變數信息
                target_param = get_config('target_param')
                y = get_config('y')
                
                # 顯示目標變數信息
                st.subheader("目標變數信息")
                st.write(f"目標變數名稱: {target_param}")
                if y is not None:
                    unique_labels = np.unique(y)
                    st.write("目標變數的唯一標籤:", unique_labels)
                
                st.session_state.setup_done = True
                progress_bar.progress(30)
                
                # 訓練和比較模型
                st.write("正在訓練模型...")
                best_model = compare_models(n_select=1)
                st.session_state.best_model = best_model
                progress_bar.progress(60)
                
                # 獲取特徵重要性
                # feature_importance = get_feature_importance(best_model)
                progress_bar.progress(80)
                
                # 顯示結果
                st.subheader("分析結果")
                
                # 分成兩列顯示結果
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("特徵重要性")
                    # plot_model(best_model, plot='feature', display_format='streamlit')
                    feature_importance_fig = plot_model(best_model, plot='feature', save=True) # 輸出是個string 圖片的名稱
                    
                    st.image(feature_importance_fig)
                                    
                with col2:
                    st.write("模型效能指標")
                    model_metrics = pull()
                    st.dataframe(model_metrics)
                    
                    
                
                # 獲取預測結果
                # predictions = predict_model(best_model)
                
                # 在顯示預測結果時加入標籤說明
                st.subheader("預測結果說明")
                predictions = predict_model(best_model)
                
                
                # 顯示混淆矩陣
                st.subheader("混淆矩陣")
                plot_model(best_model, plot='confusion_matrix', display_format='streamlit')
                
                # ROC 曲線
                st.subheader("ROC 曲線")
                plot_model(best_model, plot='auc', display_format='streamlit')
                
                progress_bar.progress(100)
                
                # 提供模型下載
                model_name = 'best_model'
                save_model(best_model, model_name)
                
                with open(f'{model_name}.pkl', 'rb') as f:
                    model_bytes = f.read()
                
                st.download_button(
                    label="下載模型",
                    data=model_bytes,
                    file_name=f"{model_name}.pkl",
                    mime="application/octet-stream"
                )
                
        except Exception as e:
            st.error(f"發生錯誤: {str(e)}")
            st.error("請確保數據格式正確，且目標變數適合分類任務")

with st.expander("使用說明"):
    st.write("""
    ### 使用步驟：
    1. 上傳CSV格式的數據檔案
    2. 選擇要預測的目標變數
    3. 選擇要使用的特徵變數（可選）
    4. 調整高級設置（可選）
    5. 點擊「開始分析」按鈕
    6. 等待分析完成，查看結果
    7. 下載訓練好的模型
    
    ### 注意事項：
    - 請確保數據已經過基本清理
    - 目標變數應為分類變數
    - 可以選擇部分特徵進行訓練，不選擇則使用所有特徵
    - 建議數據量不要太大，否則處理時間會較長
    - 可以在高級設置中調整參數以優化結果
    """)

# 添加頁尾
st.markdown("---")
st.markdown("Powered by PyCaret & Streamlit")