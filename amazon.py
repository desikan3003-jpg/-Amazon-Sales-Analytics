import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

# --- Page Config & Custom Styling ---
st.set_page_config(page_title="📊 Amazon Sales Analytics Dashboard", layout="wide")
st.markdown(
    """
    <h1 style='text-align: center; font-size: 40px; color: #FF9900; margin-top: 10px;'>
        📊 Amazon Sales Analytics dashboard
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* Custom container for a professional look */
    .metric-container {
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 15px;
        background-color: #f7f7f7;
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.05);
        height: 100%; /* Ensures consistent height in columns */
    }
    /* Ensure charts use full space and look integrated */
    .stPlotlyChart {
        padding: 10px;
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        background-color: white;
    }
    /* Styles for Tabs: CRITICAL UPDATE for Dark Mode visibility */
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px !important;
        font-weight: bold !important;
        color: white !important; /* Set text to white for visibility in Dark Mode */
        border-radius: 8px 8px 0 0 !important;
        border: 1px solid #FF9900;
        margin-right: 5px;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #FF9900;
        color: black !important; /* Set text to black for maximum contrast on the orange background */
    }
    /* Remove default Streamlit header margin for a tighter fit */
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- File Paths (MUST BE UPDATED BY USER) ---
# NOTE: Please verify these paths are correct in your local environment.
csv_files = {
    "transactions": r"C:\Users\desik\Desktop\amazonsales\transactions.csv",
    "products": r"C:\\Users\\desik\\Desktop\\amazonsales\\products.csv",
    "customers": r"C:\\Users\\desik\\Desktop\\amazonsales\\customers.csv",
    "time_dimension": r"C:\\Users\\desik\\Desktop\\amazonsales\\time__dimension.csv"
}

# --- Load CSV ---
@st.cache_data
def load_csv(path):
    try:
        df = pd.read_csv(path)
        # Convert date column from transactions
        if 'transactions' in path and 'order_date' in df.columns:
            df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        return df
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading file at {path}: {e}")
        return pd.DataFrame()

# Load all dataframes
tx_df_raw = load_csv(csv_files["transactions"])
products_df = load_csv(csv_files["products"])
customers_df = load_csv(csv_files["customers"])
time_df = load_csv(csv_files["time_dimension"])

# --- Helper Function for Data Preparation ---
def prepare_data(tx_df, products_df, customers_df):
    if tx_df.empty:
        return pd.DataFrame()
    
    # 1. Prepare date columns
    if 'order_date' in tx_df.columns:
        tx_df['order_date'] = pd.to_datetime(tx_df['order_date'], errors='coerce') 
        tx_df['order_year'] = tx_df['order_date'].dt.year
        tx_df['order_month'] = tx_df['order_date'].dt.month.apply(lambda x: f"{x:02d}")
        tx_df['order_period'] = tx_df['order_date'].dt.to_period('M').astype(str)
    else:
        st.warning("Missing 'order_date' for trend analysis.")
        return pd.DataFrame() 
    
    # 2. Merge with products (Ensuring 'subcategory' is available)
    product_cols_to_merge = ['product_id', 'product_name', 'subcategory', 'product_rating', 'is_prime_eligible']
    product_merge_cols = [col for col in product_cols_to_merge if col in products_df.columns]

    if 'product_id' in products_df.columns and 'subcategory' in products_df.columns:
        tx_df = tx_df.merge(products_df[product_merge_cols], on='product_id', how='left')
    else:
        st.error("Missing 'subcategory' or 'product_id' in products data. Analysis based on subcategory is impossible.")
        return pd.DataFrame()
        
    # 3. Merge with customers (using correct customer columns)
    customer_cols_to_merge = ['customer_id', 'customer_city', 'customer_state', 'customer_tier', 'customer_spending_tier', 'customer_age_group']
    customer_merge_cols = [col for col in customer_cols_to_merge if col in customers_df.columns]

    if 'customer_id' in customers_df.columns and 'customer_city' in customers_df.columns:
        tx_df = tx_df.merge(customers_df[customer_merge_cols], on='customer_id', how='left')
    else:
        st.warning("Missing essential customer columns. Geographic/demographic analysis will be limited.")
        
    # 4. Calculate unit_price if possible
    if 'original_price_inr' in tx_df.columns and 'quantity' in tx_df.columns:
        tx_df['unit_price'] = tx_df['original_price_inr'] / tx_df['quantity'].replace(0, 1) 
    else:
        st.warning("Could not calculate 'unit_price', mocking values.")
        tx_df['unit_price'] = tx_df['final_amount_inr'] / 0.8 if 'final_amount_inr' in tx_df.columns else 100 
        
    # 5. Mocking for 'ship_mode' if missing
    if 'ship_mode' not in tx_df.columns:
        tx_df['ship_mode'] = tx_df.get('delivery_type', np.random.choice(['Standard', 'Express', 'Priority'], tx_df.shape[0], p=[0.7, 0.2, 0.1]))

    # 6. Ensure required columns for all tabs exist (mock if needed)
    required_cols = ['final_amount_inr', 'return_status', 'delivery_days', 'discount_percent', 'is_festival_sale', 'payment_method', 'original_price_inr']
    for col in required_cols:
        if col not in tx_df.columns:
            st.warning(f"Missing critical column '{col}'. Mocking data to prevent application crash.")
            if col == 'return_status':
                tx_df[col] = np.random.choice(['Returned', 'Completed'], tx_df.shape[0], p=[0.1, 0.9])
            elif col == 'delivery_days':
                tx_df[col] = np.random.randint(1, 10, tx_df.shape[0])
            elif col == 'discount_percent':
                tx_df[col] = np.random.uniform(0, 20, tx_df.shape[0])
            elif col == 'is_festival_sale':
                tx_df[col] = np.random.choice([True, False], tx_df.shape[0], p=[0.2, 0.8])
            elif col == 'payment_method':
                tx_df[col] = np.random.choice(['Credit Card', 'UPI', 'COD'], tx_df.shape[0], p=[0.4, 0.4, 0.2])
            elif col == 'original_price_inr':
                tx_df[col] = tx_df['final_amount_inr'] / 0.9 # Mocking original price

    return tx_df

# --- Main App Logic ---

# Check if essential data is loaded
if tx_df_raw.empty or products_df.empty or customers_df.empty:
    st.error("Data Load Error: Please check the file paths for all CSVs and ensure they are correct. Cannot proceed.")
else:
    tx_df = prepare_data(tx_df_raw.copy(), products_df.copy(), customers_df.copy())

    if not tx_df.empty and 'final_amount_inr' in tx_df.columns:
        
        # --- Filters (Applied globally) ---
        st.sidebar.header("Global Data Filters")

        # 1. Year Filter
        selected_years = tx_df['order_year'].dropna().unique().astype(int).tolist()
        
        if 'filter_years' not in st.session_state:
            st.session_state['filter_years'] = sorted(selected_years) if selected_years else []
        
        if selected_years:
            st.session_state['filter_years'] = st.sidebar.multiselect(
                "Select Year(s)", 
                selected_years, 
                default=st.session_state['filter_years']
            )
            tx_df = tx_df[tx_df['order_year'].isin(st.session_state['filter_years'])]
        
        # 2. Sub-Category Filter
        if 'subcategory' in tx_df.columns:
            available_sub_categories = sorted(tx_df['subcategory'].dropna().unique().tolist())
            
            if 'filter_subcats' not in st.session_state:
                st.session_state['filter_subcats'] = available_sub_categories
            
            if available_sub_categories:
                st.session_state['filter_subcats'] = st.sidebar.multiselect(
                    "Select Sub-Category(s)", 
                    available_sub_categories, 
                    default=st.session_state['filter_subcats']
                )
                tx_df = tx_df[tx_df['subcategory'].isin(st.session_state['filter_subcats'])]
        
        st.sidebar.markdown("---")
        
        # --- Analysis Button (Forces re-run after filters) ---
        run_analysis = st.sidebar.button("Run Analysis", type="primary")

        # --- Check for valid data after filtering ---
        # The app only stops if there is absolutely no data left after filtering.
        if run_analysis or st.session_state.get('initial_run', True):
            st.session_state['initial_run'] = False
            
            if tx_df.empty or tx_df.shape[0] == 0:
                st.warning("No data remains after applying the current filters. Please adjust your selections and click 'Run Analysis'.")
                st.stop() # Stop execution if no data
            
            # Recalculate global metrics after filtering
            total_revenue = tx_df["final_amount_inr"].sum()
            total_orders = tx_df.shape[0]
            total_customers = tx_df['customer_id'].nunique()
            avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
            
            # Prepare common dataframes
            monthly_rev = tx_df.groupby('order_period', sort=False)['final_amount_inr'].sum().reset_index(name='sum')
            subcategory_rev = tx_df.groupby('subcategory')['final_amount_inr'].agg(['sum', 'count']).reset_index()
            
            # --- TABBED DASHBOARD IMPLEMENTATION ---
            # These are the clickable 'pages' you need to select at the top of the app!
            tab_exec, tab_rev, tab_cust, tab_prod, tab_ops, tab_adv = st.tabs([
                "Executive Summary (Q1-Q5)", 
                "Revenue Analytics (Q6-Q10)", 
                "Customer Analytics (Q11-Q15)", 
                "Product & Inventory (Q16-Q20)", 
                "Operation & Logistics (Q21-Q25)",
                "Advanced Analytics (Q26-Q30)"
            ])

            # =========================================================================
            # 1. EXECUTIVE SUMMARY DASHBOARD (Q1-Q5)
            # =========================================================================
            with tab_exec:
                st.subheader("Q1-Q5: Executive Summary: Key Performance Indicators & Trends")
                
                # Q1: 3 KPIs - Total Revenue, Orders, AOV (MADE EXPLICIT)
                st.markdown("#### Q1: Core Performance Metrics")
                kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
                with kpi_col1:
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    st.metric("Total Revenue (INR)", f"₹{total_revenue:,.0f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                with kpi_col2:
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    st.metric("Total Orders", f"{total_orders:,}")
                    st.markdown("</div>", unsafe_allow_html=True)
                with kpi_col3:
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    st.metric("Average Order Value (INR)", f"₹{avg_order_value:,.2f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("---")

                col_sum_chart1, col_sum_chart2 = st.columns(2)

                # Q2: Monthly Revenue Trend (Line Chart without Slider for summary)
                with col_sum_chart1:
                    st.markdown("#### Q2: Monthly Revenue Trend")
                    fig_rev_sum = px.line(
                        monthly_rev.sort_values('order_period'),
                        x='order_period', y='sum',
                        title='Total Revenue Over Time',
                        color_discrete_sequence=['#FF9900']
                    ).update_layout(xaxis_title="Time Period", yaxis_title="Revenue (INR)", height=350)
                    st.plotly_chart(fig_rev_sum, use_container_width=True)

                # Q3: Revenue Share by Sub-Category (Donut Chart)
                with col_sum_chart2:
                    st.markdown("#### Q3: Revenue Share by Sub-Category")
                    fig_pie_sum = px.pie(
                        subcategory_rev, names='subcategory', values='sum', 
                        title='Percentage Revenue Contribution by Sub-Category', hole=0.4, 
                        color_discrete_sequence=px.colors.qualitative.Bold
                    ).update_traces(textinfo='percent').update_layout(height=350)
                    st.plotly_chart(fig_pie_sum, use_container_width=True)

                st.markdown("---")
                
                # Q4: Performance Comparison by Top 5 Sub-Categories (Bar Chart)
                st.markdown("#### Q4: Top 5 Sub-Category Performance (Revenue vs. Orders)")
                top_5_subcat = subcategory_rev.nlargest(5, 'sum')
                # Melt data for grouped bar chart
                top_5_melted = top_5_subcat.melt(id_vars='subcategory', value_vars=['sum', 'count'], var_name='Metric', value_name='Value')

                fig_comp = px.bar(
                    top_5_melted, x='subcategory', y='Value', color='Metric', barmode='group',
                    title='Revenue (Sum) vs. Orders (Count)',
                    color_discrete_map={'sum': '#FF9900', 'count': '#1f77b4'}
                ).update_layout(xaxis_title="Sub-Category", yaxis_title="Value (INR/Count)", height=350)
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # Q5: Interactive Filter for Top 5 Sub-Categories (Table)
                st.markdown("#### Q5: Drill-Down on Top Sub-Category Metrics")
                
                top_5_subcat_list = top_5_subcat['subcategory'].tolist()

                if top_5_subcat_list:
                    selected_sub_cat_q5 = st.selectbox(
                        'Select a Top Sub-Category to View Detailed Metrics', 
                        top_5_subcat_list,
                        key='q5_select'
                    )
                    
                    sub_metrics = top_5_subcat[top_5_subcat['subcategory'] == selected_sub_cat_q5].iloc[0]
                    col_q5_1, col_q5_2, col_q5_3 = st.columns(3)
                    col_q5_1.metric("Selected Sub-Category Revenue", f"₹{sub_metrics['sum']:,.0f}")
                    col_q5_2.metric("Selected Sub-Category Orders", f"{sub_metrics['count']:,}")
                    col_q5_3.metric("Selected Sub-Category AOV", f"₹{(sub_metrics['sum'] / sub_metrics['count']):,.2f}" if sub_metrics['count'] > 0 else "N/A")
                else:
                    st.info("No top sub-categories found after applying current filters.")

            # =========================================================================
            # 2. REVENUE ANALYTICS DASHBOARD (Q6-Q10)
            # =========================================================================
            with tab_rev:
                st.subheader("Q6-Q10: Revenue Analytics: Deep Dive into Sales Drivers")

                # Q6: Interactive Monthly Revenue Trend (Line with Range Slider)
                st.markdown("#### Q6: Monthly Revenue Trend with Zoom/Range Slider")
                fig_rev = px.line(
                    monthly_rev.sort_values('order_period'), x='order_period', y='sum',
                    title='Total Revenue by Month (Interactive Trend)', markers=True,
                    color_discrete_sequence=['#FF9900']
                ).update_layout(
                    xaxis_title="Time Period (Year-Month)", yaxis_title="Total Revenue (INR)",
                    hovermode="x unified", height=450
                ).update_xaxes(
                    rangeslider_visible=True, title_text="Time Period (Year-Month)"
                )
                st.plotly_chart(fig_rev, use_container_width=True)
                
                st.markdown("---")

                col_rev_chart1, col_rev_chart2 = st.columns(2)

                # Q7: AOV Distribution by Sub-Category (Box Plot)
                with col_rev_chart1:
                    st.markdown("#### Q7: Order Value Distribution by Sub-Category")
                    fig_box = px.box(
                        tx_df, x='subcategory', y='final_amount_inr', 
                        title='Order Value Distribution (Median, Quartiles, Outliers)',
                        color='subcategory', notched=True, 
                        color_discrete_sequence=px.colors.qualitative.Dark24
                    ).update_layout(
                        template="plotly_white", showlegend=False, height=400
                    ).update_yaxes(
                        title_text="Order Value (INR)",
                        range=[tx_df['final_amount_inr'].quantile(0.001), tx_df['final_amount_inr'].quantile(0.99)]
                    )
                    st.plotly_chart(fig_box, use_container_width=True)

                # Q8: Top 10 Sub-Categories by Revenue (Bar Chart)
                with col_rev_chart2:
                    st.markdown("#### Q8: Top 10 Revenue Generators (Sub-Category)")
                    top_10_rev = subcategory_rev.nlargest(10, 'sum')
                    fig_top = px.bar(
                        top_10_rev, x='subcategory', y='sum', 
                        title='Top Sub-Categories by Total Revenue',
                        color='sum', color_continuous_scale=px.colors.sequential.Agsunset,
                        labels={'sum': 'Total Revenue (INR)', 'subcategory': 'Sub-Category'}
                    ).update_layout(height=400)
                    st.plotly_chart(fig_top, use_container_width=True)
                
                st.markdown("---")

                # Q9: Revenue Comparison by Sub-Category over Year (Stacked Bar)
                st.markdown("#### Q9: Annual Revenue Comparison by Sub-Category")
                annual_cat_rev = tx_df.groupby(['order_year', 'subcategory'])['final_amount_inr'].sum().reset_index()
                annual_cat_rev['order_year'] = annual_cat_rev['order_year'].astype(str)

                fig_bar = px.bar(
                    annual_cat_rev, x='order_year', y='final_amount_inr',
                    color='subcategory', title='Sub-Category Performance Across Selected Years', 
                    labels={'final_amount_inr': 'Total Revenue (INR)', 'order_year': 'Year'}
                ).update_layout(barmode='stack', template="plotly_white", height=400)
                st.plotly_chart(fig_bar, use_container_width=True)

                # Q10: Strategic Scenario Simulation (Slider)
                st.markdown("#### Q10: Strategic Scenario Simulation (Revenue Uplift)")
                col_slider, col_info = st.columns([2, 1])
                with col_slider:
                    uplift = st.slider("Expected % Increase in Orders", min_value=0, max_value=50, value=10, key='q10_slider')
                
                projected_revenue = total_revenue * (1 + uplift / 100)
                
                with col_info:
                    st.metric("Projected Revenue", f"₹{projected_revenue:,.0f}", delta=f"↑ {uplift}% Increase")
                st.info("Simulate the impact of marketing efforts on future revenue based on order volume uplift.")


            # =========================================================================
            # 3. CUSTOMER ANALYTICS DASHBOARD (Q11-Q15)
            # =========================================================================
            with tab_cust:
                st.subheader("Q11-Q15: Customer Analytics: Understanding the Customer Base")
                
                # Q11: Key Customer Metrics (MADE EXPLICIT)
                st.markdown("#### Q11: Key Customer Metrics (LTV & Frequency)")
                col_cust_kpi1, col_cust_kpi2, col_cust_kpi3 = st.columns(3)
                with col_cust_kpi1:
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    st.metric("Total Unique Customers", f"{total_customers:,}")
                    st.markdown("</div>", unsafe_allow_html=True)
                with col_cust_kpi2:
                    avg_freq = total_orders / total_customers if total_customers > 0 else 0
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    st.metric("Avg. Purchase Frequency", f"{avg_freq:.2f} orders/customer")
                    st.markdown("</div>", unsafe_allow_html=True)
                with col_cust_kpi3:
                    # Mock LTV - Revenue / Customers
                    ltv = total_revenue / total_customers if total_customers > 0 else 0
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    st.metric("Approx. Customer LTV", f"₹{ltv:,.0f}")
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("---")
                
                # Q12 & Q13 moved into a single row with 2 columns 
                col_cust_chart1, col_cust_chart2 = st.columns(2)
                
                # Q12: Customer Tier Breakdown (Pie Chart - Now in its own larger column)
                with col_cust_chart1:
                    st.markdown("#### Q12: Customer Tier Distribution (Order Volume)")
                    if 'customer_tier' in customers_df.columns:
                        tier_counts = customers_df['customer_tier'].value_counts().reset_index()
                        tier_counts.columns = ['customer_tier', 'Count']
                        fig_tier = px.pie(
                            tier_counts, names='customer_tier', values='Count',
                            title='Tier Distribution', hole=0.3
                        ).update_layout(showlegend=True, height=400) # Increased height for better view
                        st.plotly_chart(fig_tier, use_container_width=True)
                    else:
                        st.warning("Missing 'customer_tier' for Q12.")

                # Q13: Top 10 Customers by Revenue (Bar Chart)
                with col_cust_chart2:
                    st.markdown("#### Q13: Top 10 Customers by Revenue")
                    top_cust = tx_df.groupby('customer_id')['final_amount_inr'].sum().nlargest(10).reset_index()
                    fig_top_cust = px.bar(
                        top_cust, x='customer_id', y='final_amount_inr',
                        title='Top Spenders',
                        color='final_amount_inr', color_continuous_scale=px.colors.sequential.Agsunset
                    ).update_layout(xaxis_title="Customer ID", yaxis_title="Revenue (INR)", height=400)
                    st.plotly_chart(fig_top_cust, use_container_width=True)

                st.markdown("---")
                
                # Q14: Orders by State (Bar Chart)
                st.markdown("#### Q14: Orders by State (Top 10)")
                orders_by_state = tx_df.groupby('customer_state')['customer_id'].count().reset_index(name='Order Count')
                fig_state = px.bar(
                    orders_by_state.nlargest(10, 'Order Count'), y='customer_state', x='Order Count', orientation='h',
                    title='Top States by Order Volume',
                    color='Order Count', color_continuous_scale=px.colors.sequential.Turbo
                ).update_layout(yaxis_title="Customer State", height=400)
                st.plotly_chart(fig_state, use_container_width=True)

                # Q15: Customer Age Group Distribution (Interactive Pie/Sunburst)
                st.markdown("#### Q15: Customer Age Group Distribution")
                if 'customer_age_group' in customers_df.columns:
                    age_counts = customers_df['customer_age_group'].value_counts().reset_index()
                    age_counts.columns = ['customer_age_group', 'Count']
                    fig_age = px.pie(
                        age_counts, names='customer_age_group', values='Count',
                        title='Age Group Distribution of Customers',
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    ).update_layout(height=400)
                    st.plotly_chart(fig_age, use_container_width=True)
                else:
                    st.warning("Cannot display age distribution for Q15: Missing 'customer_age_group' column.")

            # =========================================================================
            # 4. PRODUCT & INVENTORY DASHBOARD (Q16-Q20)
            # =========================================================================
            with tab_prod:
                st.subheader("Q16-Q20: Product & Inventory Analytics: Performance by Product Line")
                
                col_prod_chart1, col_prod_chart2 = st.columns(2)

                # Q16: Units Sold by Sub-Category (Bar Chart)
                with col_prod_chart1:
                    st.markdown("#### Q16: Volume Sold by Sub-Category")
                    fig_units = px.bar(
                        subcategory_rev.nlargest(10, 'count'), x='subcategory', y='count',
                        title='Units Sold (Top 10 Sub-Categories)',
                        color='count', color_continuous_scale=px.colors.sequential.Oranges
                    ).update_layout(xaxis_title="Sub-Category", yaxis_title="Units Sold", height=400)
                    st.plotly_chart(fig_units, use_container_width=True)

                # Q17: Prime Eligibility Impact (Grouped Bar Chart - Renumbered from old Q18)
                with col_prod_chart2:
                    st.markdown("#### Q17: Prime vs. Non-Prime Revenue by Sub-Category (Top 5)")
                    if 'is_prime_eligible' in tx_df.columns:
                        prime_rev = tx_df.groupby(['subcategory', 'is_prime_eligible'])['final_amount_inr'].sum().reset_index()
                        top_sub_list = tx_df['subcategory'].value_counts().nlargest(5).index.tolist()
                        prime_rev = prime_rev[prime_rev['subcategory'].isin(top_sub_list)]
                        
                        fig_prime = px.bar(
                            prime_rev, x='subcategory', y='final_amount_inr', color='is_prime_eligible',
                            title='Revenue Split: Prime vs. Non-Prime',
                            barmode='group',
                            color_discrete_map={True: '#25C780', False: '#FF9900'}
                        ).update_layout(xaxis_title="Sub-Category", yaxis_title="Revenue (INR)", height=400)
                        st.plotly_chart(fig_prime, use_container_width=True)
                    else:
                        st.warning("Cannot display Prime Analysis for Q17: Missing 'is_prime_eligible' column.")


                st.markdown("---")

                # Q18: Unit Price Distribution Slider Filter (Renumbered from old Q19)
                st.markdown("#### Q18: Unit Price Distribution Filter")
                if 'unit_price' in tx_df.columns:
                    # Calculate reasonable min/max based on quantiles to prevent skewed sliders
                    min_price_q = tx_df['unit_price'].quantile(0.01)
                    max_price_q = tx_df['unit_price'].quantile(0.99)
                    
                    price_range = st.slider(
                        'Filter Orders by Unit Price Range (INR)',
                        float(min_price_q), float(max_price_q), 
                        (float(min_price_q), float(max_price_q)),
                        step=(max_price_q - min_price_q) / 100,
                        key='q18_slider'
                    )
                    
                    filtered_price_tx = tx_df[(tx_df['unit_price'] >= price_range[0]) & (tx_df['unit_price'] <= price_range[1])]
                    st.info(f"Orders within this price range: **{filtered_price_tx.shape[0]:,}** (Revenue: ₹{filtered_price_tx['final_amount_inr'].sum():,.0f})")
                else:
                    st.warning("Cannot display Unit Price Filter for Q18: Missing 'unit_price' column.")

                # Q19: Product Rating vs. Quantity Sold (Scatter Plot - Renumbered from old Q20)
                st.markdown("#### Q19: Product Rating vs. Quantity Sold")
                if 'product_rating' in tx_df.columns and 'quantity' in tx_df.columns:
                    # Aggregate total quantity sold per product
                    product_sales = tx_df.groupby('product_id')['quantity'].sum().reset_index(name='Total Quantity Sold')
                    # Merge with product details (rating, subcategory)
                    product_performance = product_sales.merge(
                        tx_df[['product_id', 'product_rating', 'subcategory', 'product_name']].drop_duplicates(),
                        on='product_id', how='left'
                    )
                    
                    fig_scatter = px.scatter(
                        product_performance, x='product_rating', y='Total Quantity Sold', 
                        color='subcategory', 
                        hover_data=['product_id', 'product_name', 'subcategory'],
                        title='Rating vs. Sales Volume',
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.warning("Cannot display Rating vs. Quantity for Q19: Missing 'product_rating' or 'quantity' columns.")

                # Q20: Mock Inventory Turns KPI (New Addition to fill the gap)
                st.markdown("#### Q20: Mock Inventory Turns KPI")
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                # Mock calculation: Total Units Sold / (Assumed Avg Inventory * 10)
                mock_units_sold = tx_df['quantity'].sum()
                mock_inventory = mock_units_sold * 0.1 # Mocking that 10% of units sold is the inventory base
                mock_turns = mock_units_sold / (mock_inventory + 1)
                st.metric("Mock Inventory Turnover", f"{mock_turns:.2f}x")
                st.markdown("</div>", unsafe_allow_html=True)

            # =========================================================================
            # 5. OPERATION & LOGISTICS DASHBOARD (Q21-Q25) - RESTRUCTURED
            # =========================================================================
            with tab_ops:
                st.subheader("Q21-Q25: Operation & Logistics: Deep Dive into Delivery, Payment, and Quality")

                # --- Q21: Delivery Performance Dashboard ---
                st.markdown("## Q21: Delivery Performance Dashboard")
                avg_del_days = tx_df["delivery_days"].mean()
                delayed_deliveries_pct = (tx_df["delivery_days"] > 5).mean()
                
                col_kpi_21_1, col_kpi_21_2 = st.columns(2)
                with col_kpi_21_1:
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    st.metric("Avg. Delivery Days", f"**{avg_del_days:.1f}** days")
                    st.markdown("</div>", unsafe_allow_html=True)
                with col_kpi_21_2:
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    st.metric("% Orders Delayed (>5 days SLA)", f"**{delayed_deliveries_pct:.2%}**")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("---")

                col_21_chart1, col_21_chart2 = st.columns(2)

                # Q21 Chart 1: Delivery Days Distribution (from old Q25)
                with col_21_chart1:
                    st.markdown("#### Delivery Days Distribution")
                    if 'delivery_days' in tx_df.columns:
                        fig_del_hist = px.histogram(
                            tx_df, x='delivery_days', nbins=15,
                            title='Frequency of Delivery Times',
                            color_discrete_sequence=['#1f77b4']
                        ).update_layout(xaxis_title="Delivery Days", yaxis_title="Order Count", height=400)
                        st.plotly_chart(fig_del_hist, use_container_width=True)

                # Q21 Chart 2: Avg Delivery Time by State (Geographic Performance)
                with col_21_chart2:
                    st.markdown("#### Avg. Delivery Days by State (Top 10)")
                    avg_days_by_state = tx_df.groupby('customer_state')['delivery_days'].mean().reset_index(name='Avg Delivery Days')
                    fig_geo_perf = px.bar(
                        avg_days_by_state.nlargest(10, 'Avg Delivery Days').sort_values('Avg Delivery Days', ascending=True),
                        y='customer_state', x='Avg Delivery Days', orientation='h',
                        title='Longest Avg. Delivery Times by State',
                        color='Avg Delivery Days', color_continuous_scale=px.colors.sequential.Teal,
                    ).update_layout(yaxis_title="Customer State", height=400)
                    st.plotly_chart(fig_geo_perf, use_container_width=True)

                # --- Q22: Payment Analytics Dashboard ---
                st.markdown("## Q22: Payment Analytics Dashboard")
                
                col_22_chart1, col_22_chart2 = st.columns(2)

                # Q22 Chart 1: Payment Method Preference (Order Count)
                with col_22_chart1:
                    st.markdown("#### Payment Method Preference (Order Volume)")
                    payment_counts = tx_df['payment_method'].value_counts().reset_index()
                    payment_counts.columns = ['Payment Method', 'Order Count']
                    fig_pay_pref = px.pie(
                        payment_counts, names='Payment Method', values='Order Count',
                        title='Order Share by Payment Method', hole=0.3,
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    ).update_layout(height=400)
                    st.plotly_chart(fig_pay_pref, use_container_width=True)

                # Q22 Chart 2: Revenue Trend by Payment Method
                with col_22_chart2:
                    st.markdown("#### Revenue Trend by Payment Method")
                    monthly_rev_payment = tx_df.groupby(['order_period', 'payment_method'])['final_amount_inr'].sum().reset_index()
                    
                    fig_pay_trend = px.line(
                        monthly_rev_payment.sort_values('order_period'), x='order_period', y='final_amount_inr',
                        color='payment_method',
                        title='Monthly Revenue by Payment Method', markers=True
                    ).update_layout(xaxis_title="Time Period", yaxis_title="Revenue (INR)", height=400, legend_title="Payment Method")
                    st.plotly_chart(fig_pay_trend, use_container_width=True)


                # --- Q23: Return & Cancellation Dashboard ---
                st.markdown("## Q23: Return & Cancellation Dashboard")

                overall_return_rate = (tx_df["return_status"] == "Returned").mean()
                total_return_cost = tx_df[tx_df["return_status"] == "Returned"]['final_amount_inr'].sum()
                
                col_kpi_23_1, col_kpi_23_2 = st.columns(2)
                with col_kpi_23_1:
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    st.metric("Overall Return Rate", f"**{overall_return_rate:.2%}**")
                    st.markdown("</div>", unsafe_allow_html=True)
                with col_kpi_23_2:
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    st.metric("Est. Total Value of Returns", f"₹**{total_return_cost:,.0f}**")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("---")
                
                col_23_chart1, col_23_chart2 = st.columns(2)

                with col_23_chart1:
                    st.markdown("## Q24: Return Rate by Sub-Category")
                    return_rate_sub = tx_df.groupby('subcategory')['return_status'].apply(lambda x: (x == 'Returned').mean()).reset_index(name='Return Rate')
                    fig_returns = px.bar(
                        return_rate_sub.sort_values('Return Rate', ascending=False).nlargest(10, 'Return Rate'), 
                        y='subcategory', x='Return Rate', orientation='h',
                        title='Highest Return Rate Sub-Categories',
                        color='Return Rate', color_continuous_scale=px.colors.sequential.Reds
                    ).update_layout(yaxis_title="Sub-Category", xaxis_tickformat=".2%", height=400)
                    st.plotly_chart(fig_returns, use_container_width=True)

                with col_23_chart2:
                    st.markdown("## Q25: Return Rate by Payment Method")
                    if 'payment_method' in tx_df.columns and 'return_status' in tx_df.columns:
                        payment_returns = tx_df.groupby('payment_method')['return_status'].apply(
                            lambda x: (x == 'Returned').mean()
                        ).reset_index(name='Return Rate')
                        
                        fig_pay_ret = px.bar(
                            payment_returns.sort_values('Return Rate', ascending=False), 
                            x='payment_method', y='Return Rate',
                            title='Return Rate by Payment Type',
                            color='Return Rate', color_continuous_scale=px.colors.sequential.YlOrRd
                        ).update_layout(yaxis_tickformat=".2%", xaxis_title="Payment Method", height=400)
                        st.plotly_chart(fig_pay_ret, use_container_width=True)

                # Q24 & Q25 are implicitly covered in the expanded Q21-Q23 sections now.
                st.markdown("---")
                st.info("Q24 (Delivery Distribution) and Q25 (Returns by Payment) are now integrated into the detailed Q21 and Q23 dashboards above for better operational flow.")


            # =========================================================================
            # 6. ADVANCED ANALYTICS DASHBOARD (Q26-Q30)
            # =========================================================================
            with tab_adv:
                st.subheader("Q26-Q30: Advanced Analytics: Strategic Insights")
                st.info("These charts demonstrate advanced concepts like segmentation and forecasting, which are mocked here as they require specific ML/Time-Series models not included in this code.")

                col_adv_chart1, col_adv_chart2 = st.columns(2)

                # Q26: Customer Spending Tier Breakdown (Pie Chart - Renumbered from old Q26)
                with col_adv_chart1:
                    st.markdown("#### Q26: Customer Spending Tier Distribution")
                    if 'customer_spending_tier' in tx_df.columns:
                        spending_counts = tx_df[['customer_id', 'customer_spending_tier']].drop_duplicates()['customer_spending_tier'].value_counts().reset_index()
                        spending_counts.columns = ['Tier', 'Count']
                        fig_spending = px.pie(
                            spending_counts, names='Tier', values='Count',
                            title='Customer Spending Tier Share', hole=0.5,
                            color_discrete_sequence=px.colors.qualitative.T10
                        ).update_layout(height=400)
                        st.plotly_chart(fig_spending, use_container_width=True)
                    else:
                        st.warning("Missing 'customer_spending_tier' for Q26.")


                # Q27: Revenue Forecasting (Mock Line Chart with Confidence Interval - Renumbered from old Q27)
                with col_adv_chart2:
                    st.markdown("#### Q27: 12-Month Revenue Forecast (Mock)")
                    
                    last_period_str = monthly_rev['order_period'].iloc[-1]
                    last_date = pd.to_datetime(last_period_str)
                    
                    # Generate 12 future periods starting from the month after the last recorded period
                    start_date_forecast = last_date + pd.DateOffset(months=1)
                    future_dates = pd.date_range(
                        start=start_date_forecast, 
                        periods=12, 
                        freq='M'
                    ).to_period('M').astype(str).tolist() 

                    last_rev = monthly_rev['sum'].iloc[-1]
                    # Simple mock trend: slight growth with noise (length 12)
                    forecast_rev = [last_rev * (1 + 0.01*i) * np.random.uniform(0.98, 1.02) for i in range(1, 13)] 
                    
                    # Create the combined DataFrame
                    forecast_df = pd.DataFrame({
                        'Period': monthly_rev['order_period'].tolist() + future_dates,
                        'Revenue': monthly_rev['sum'].tolist() + forecast_rev,
                        'Type': ['Actual'] * monthly_rev.shape[0] + ['Forecast'] * 12,
                        'Upper': monthly_rev['sum'].tolist() + [r * 1.05 for r in forecast_rev],
                        'Lower': monthly_rev['sum'].tolist() + [r * 0.95 for r in forecast_rev]
                    })
                    
                    fig_forecast = px.line(
                        forecast_df, x='Period', y='Revenue', color='Type',
                        title='Revenue Forecast vs. Actuals',
                        color_discrete_map={'Actual': '#FF9900', 'Forecast': '#1f77b4'}
                    ).update_layout(height=400)
                    st.plotly_chart(fig_forecast, use_container_width=True)
                
                st.markdown("---")

                # Q28: Interactive Scatter for Discount vs. Revenue (Renumbered from old Q28)
                st.markdown("#### Q28: Discount % vs. Final Amount (Interactive Scatter)")
                
                fig_disc = px.scatter(
                    tx_df, 
                    x='discount_percent', 
                    y='final_amount_inr', 
                    color='subcategory', 
                    hover_data=['transaction_id', 'quantity'],
                    title='Discount Impact on Revenue',
                    template='plotly_white'
                ).update_layout(xaxis_title="Discount Percent", yaxis_title="Final Amount (INR)", height=400)
                st.plotly_chart(fig_disc, use_container_width=True)
                
                # Q29: Total Discount Applied (KPI - Renumbered from old Q29 - NOW FULL WIDTH)
                total_discount = (tx_df['original_price_inr'] * (tx_df['discount_percent'] / 100)).sum()
                st.markdown("#### Q29: Total Discount Applied KPI (Financial Cost)")
                
                col_disc_kpi = st.columns(1)[0] # Using a single column for full width
                with col_disc_kpi:
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    st.metric("Total Discount Applied (INR)", f"₹{total_discount:,.0f}", delta=f"{-total_discount / total_revenue:.2%} of Revenue")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Q30: Festival Sale Impact (Grouped Bar Chart - Renumbered from old Q30)
                st.markdown("#### Q30: Festival vs. Non-Festival Revenue")
                if 'is_festival_sale' in tx_df.columns:
                    festival_rev = tx_df.groupby('is_festival_sale')['final_amount_inr'].sum().reset_index()
                    festival_rev['is_festival_sale'] = festival_rev['is_festival_sale'].astype(str).replace({'True': 'Festival', 'False': 'Non-Festival'})
                    
                    fig_fest = px.bar(
                        festival_rev, x='is_festival_sale', y='final_amount_inr',
                        title='Revenue Generated During Festival Sales',
                        color='is_festival_sale',
                        color_discrete_map={'Festival': '#FF5733', 'Non-Festival': '#1f77b4'}
                    ).update_layout(xaxis_title="Sale Type", yaxis_title="Revenue (INR)", height=400)
                    st.plotly_chart(fig_fest, use_container_width=True)
                else:
                    st.warning("Missing 'is_festival_sale' for Q30.")


        else:
            # Display a prompt to run analysis initially
            st.info("Please set your filters in the sidebar and click 'Run Analysis' to load the interactive dashboards.")


    else:
        st.error("Filtered data is empty or missing essential columns. Please check your data files and filter selections. Make sure 'subcategory' exists in your product data.")