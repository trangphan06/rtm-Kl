import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from k_means_constrained import KMeansConstrained
import numpy as np
import io
import folium
from streamlit_folium import st_folium
import math
import warnings
import time
import gc
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import base64
import os

warnings.filterwarnings('ignore')

# ==========================================
# 0. CẤU HÌNH CHUNG & CSS
# ==========================================
st.set_page_config(layout="wide", page_title="RTM Route Planner", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        .block-container { 
            padding-top: 3rem !important;
            padding-bottom: 2rem; 
            padding-left: 5rem; 
            padding-right: 5rem;
            max_width: 1400px;
            margin: auto;
        }
        header[data-testid="stHeader"] {
            height: 2rem !important;
        }
        iframe { width: 100% !important; }
        
        div[data-testid="stFormSubmitButton"] > button {
            width: auto !important;
        }
        
        div[data-testid="column"]:nth-of-type(3) button {
            float: right;
        }

        .main-title {
            color: #8B0000 !important; 
            font-weight: bold !important;
            font-size: 2.5rem !important;
            margin-bottom: 1rem !important;
            text-align: center;
        }

        [data-testid="stDataFrame"] td {
            text-align: center !important;
            font-size: 11px !important;
            padding: 0px !important;
            white-space: nowrap !important;
        }
        [data-testid="stDataFrame"] th {
            text-align: center !important;
            font-size: 11px !important;
            padding: 2px !important;
        }
        [data-testid="stDataFrame"] th button { display: none !important; }
        
        .warning-box {
            background-color: #fffae5;
            color: #856404;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            border: 1px solid #ffeeba;
            margin-bottom: 10px;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)

def render_main_title():
    c_spacer, c_info = st.columns([6, 4])
    st.markdown('<h1 class="main-title">Công cụ xếp tuyến - RTM Route Planner</h1>', unsafe_allow_html=True)

ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
ESRI_ATTR = "Tiles &copy; Esri" 

# ==========================================
# 1. QUẢN LÝ STATE
# ==========================================
if 'global_state' not in st.session_state:
    st.session_state.global_state = {
        'has_started': False,
        'step': 'welcome', 
        'config': {'is_tp': False, 'is_vp': False, 'is_integrated': False, 'tp_mode': "Chế độ 1"}
    }

if 'show_reset_warning' not in st.session_state: st.session_state.show_reset_warning = False
if 'page' not in st.session_state: st.session_state.page = 'setup'
if 'df' not in st.session_state: st.session_state.df = None
if 'df_edited' not in st.session_state: st.session_state.df_edited = None
if 'col_mapping' not in st.session_state: st.session_state.col_mapping = {}
if 'mapping_confirmed' not in st.session_state: st.session_state.mapping_confirmed = False
if 'time_matrix' not in st.session_state: 
    st.session_state.time_matrix = {
        'MT': 20.0, 'Cooler': 18.0, 'Gold': 15.0, 'Silver': 10.0, 
        'Bronze': 5.0, 'Mặc định/Trống': 10.0
    }
if 'upload_msg' not in st.session_state: st.session_state.upload_msg = None
if 'upload_msg_type' not in st.session_state: st.session_state.upload_msg_type = "success"
if 'v1_df_edited' not in st.session_state: st.session_state.v1_df_edited = None
if 'v1_report' not in st.session_state: st.session_state.v1_report = None
if 'v1_df_original' not in st.session_state: st.session_state.v1_df_original = None
if 'v1_map_snapshot' not in st.session_state: st.session_state.v1_map_snapshot = None
if 'v2_df_edited' not in st.session_state: st.session_state.v2_df_edited = None
if 'v2_report' not in st.session_state: st.session_state.v2_report = None
if 'v2_df_original' not in st.session_state: st.session_state.v2_df_original = None
if 'v2_map_snapshot' not in st.session_state: st.session_state.v2_map_snapshot = None
if 'map_version' not in st.session_state: st.session_state.map_version = 0
if 'col_widths' not in st.session_state: st.session_state.col_widths = {}
if 'last_mode' not in st.session_state: st.session_state.last_mode = "Chế độ 1"
if 'stage' not in st.session_state: st.session_state.stage = '1_upload'
if 'df_cust' not in st.session_state: st.session_state.df_cust = None
if 'df_dist' not in st.session_state: st.session_state.df_dist = None
if 'df_editing' not in st.session_state: st.session_state.df_editing = None 
if 'df_final' not in st.session_state: st.session_state.df_final = None
if 'map_clicked_code' not in st.session_state: st.session_state.map_clicked_code = None
if 'editor_filter_mode' not in st.session_state: st.session_state.editor_filter_mode = 'all'
if 'col_map_main' not in st.session_state: st.session_state.col_map_main = {}
if 'has_changes' not in st.session_state: st.session_state.has_changes = False
if 'confirm_reset' not in st.session_state: st.session_state.confirm_reset = False
if 'depot_coords' not in st.session_state: st.session_state.depot_coords = None
if 'vp_msg' not in st.session_state: st.session_state.vp_msg = None
if 'vp_msg_type' not in st.session_state: st.session_state.vp_msg_type = None
if 'editor_filter_key' not in st.session_state: st.session_state.editor_filter_key = None
if 'show_download_options' not in st.session_state: st.session_state.show_download_options = False
if 'tp_confirm_clear' not in st.session_state: st.session_state.tp_confirm_clear = False
if 'vp_confirm_clear' not in st.session_state: st.session_state.vp_confirm_clear = False
if 'map_needs_refresh' not in st.session_state: st.session_state.map_needs_refresh = False
if 'just_reset' not in st.session_state: st.session_state.just_reset = False

# ==========================================
# 2. LOGIC FUNCTIONS
# ==========================================
WORKING_DAYS = 20

@st.cache_data
def load_excel_file(file):
    return pd.read_excel(file, dtype=str)

@st.cache_data(show_spinner=False)
def run_territory_planning_v1(df, lat_col, lon_col, n_clusters, min_size, max_size, n_init=50):
    df_run = df.copy()
    df_run[lat_col] = pd.to_numeric(df_run[lat_col], errors='coerce')
    df_run[lon_col] = pd.to_numeric(df_run[lon_col], errors='coerce')
    df_run = df_run.dropna(subset=[lat_col, lon_col])
    
    coords = df_run[[lat_col, lon_col]]
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    if min_size * n_clusters > len(df): return None, "Lỗi: Số lượng tối thiểu quá lớn."
    if max_size * n_clusters < len(df): return None, "Lỗi: Số lượng tối đa quá nhỏ."

    best_clf = None
    best_inertia = float('inf')

    try:
        for i in range(n_init):
            clf = KMeansConstrained(
                n_clusters=n_clusters, size_min=min_size, size_max=max_size, 
                random_state=42 + i, n_init=1
            )
            clf.fit(coords_scaled)
            if clf.inertia_ < best_inertia:
                best_inertia = clf.inertia_
                best_clf = clf
            
        df_run['territory_id'] = best_clf.labels_ + 1
        stats = df_run['territory_id'].value_counts().sort_index().reset_index()
        stats.columns = ['Tuyến (RouteID)', 'Số lượng KH']
        return df_run, stats
    except Exception as e:
        return None, str(e)

@st.cache_data(show_spinner=False)
def run_territory_planning_v2(df, lat_col, lon_col, freq_col, type_col, time_matrix, n_clusters, min_capacity_total, max_capacity_total):
    df_run = df.copy()
    df_run[lat_col] = pd.to_numeric(df_run[lat_col], errors='coerce')
    df_run[lon_col] = pd.to_numeric(df_run[lon_col], errors='coerce')
    df_run = df_run.dropna(subset=[lat_col, lon_col])

    def calc_load(row):
        try: freq = float(row[freq_col])
        except: freq = 1.0
        c_type = str(row[type_col]).strip()
        key = c_type if c_type in time_matrix else 'Mặc định/Trống'
        time_val = time_matrix.get(key, 10.0)
        return freq * time_val

    df_run['workload_min'] = df_run.apply(calc_load, axis=1)
    total_minutes = df_run['workload_min'].sum()
    
    TARGET_POINTS = 50000 
    raw_quantum = total_minutes / TARGET_POINTS
    QUANTUM = max(1, math.ceil(raw_quantum)) 
    
    df_run['weight_points'] = np.ceil(df_run['workload_min'] / QUANTUM).astype(int)
    df_exploded = df_run.loc[df_run.index.repeat(df_run['weight_points'])].copy()
    df_exploded['original_index'] = df_exploded.index
    df_exploded = df_exploded.reset_index(drop=True)
    
    size_min = int(min_capacity_total / QUANTUM)
    size_max = int(max_capacity_total / QUANTUM)
    
    scaler = StandardScaler()
    coords = df_exploded[[lat_col, lon_col]]
    coords_scaled = scaler.fit_transform(coords)
    
    n_init = 5
    best_clf = None
    best_inertia = float('inf')

    try:
        for i in range(n_init):
            clf = KMeansConstrained(
                n_clusters=n_clusters, size_min=size_min, size_max=size_max,
                random_state=42 + i, n_init=1
            )
            clf.fit(coords_scaled)
            if clf.inertia_ < best_inertia:
                best_inertia = clf.inertia_
                best_clf = clf
        
        df_exploded['territory_id'] = best_clf.labels_ + 1
        final_labels = df_exploded.groupby('original_index')['territory_id'].agg(lambda x: x.mode()[0])
        df_run['territory_id'] = final_labels
        
        stats = df_run.groupby('territory_id').agg(
            count_kh=('territory_id', 'count'),
            sum_min=('workload_min', 'sum')
        ).reset_index()
        stats.columns = ['Tuyến (RouteID)', 'Số lượng KH', 'Workload_Total_Min']
        stats['Workload_Day'] = (stats['Workload_Total_Min'] / 60 / WORKING_DAYS).round(2)
        return df_run, stats
    except Exception as e:
        return None, str(e)

def generate_folium_map_tp(df, mapping, time_matrix, mode="Chế độ 1"):
    if df.empty: return None, None
    df_plot = df.copy()
    lat_col, lon_col = mapping['lat'], mapping['lon']
    df_plot[lat_col] = pd.to_numeric(df_plot[lat_col], errors='coerce')
    df_plot[lon_col] = pd.to_numeric(df_plot[lon_col], errors='coerce')
    df_plot = df_plot.dropna(subset=[lat_col, lon_col])
    
    if df_plot.empty: return None, None
    map_center = [df_plot[lat_col].mean(), df_plot[lon_col].mean()]
    m = folium.Map(location=map_center, zoom_start=11, prefer_canvas=True, tiles=ESRI_URL, attr=ESRI_ATTR)
    colors = ["#FF0000", "#0000FF", "#00FF00", "#FFFF00", "#FF00FF", "#00FFFF", "#800000", "#008000", "#000080", "#FFA500"]
    unique_ids = sorted(df_plot['territory_id'].unique())
    color_map = {int(id): colors[(int(id) - 1) % len(colors)] for id in unique_ids}

    legend_html = ''' <div style="position: fixed; bottom: 30px; left: 30px; width: 120px; height: auto; 
                    border:2px solid grey; z-index:9999; font-size:12px; 
                    background-color:white; padding: 10px; opacity: 0.9;">
                    <b>Chú giải:</b><br>'''
    for rid in unique_ids:
        c = color_map.get(int(rid), 'gray')
        legend_html += f'<i style="background:{c}; width:10px; height:10px; display:inline-block; margin-right: 5px;"></i> Tuyến {rid}<br>'
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))
    
    for _, row in df_plot.iterrows():
        tooltip_parts = [f"<b>Tuyến: {row['territory_id']}</b>"]
        if mapping['customer_code'] in row: tooltip_parts.append(f"Mã KH: {row[mapping['customer_code']]}")
        if mapping.get('customer_name') and mapping['customer_name'] in row: tooltip_parts.append(f"Tên: {row[mapping['customer_name']]}")
        tooltip_txt = "<br>".join(tooltip_parts)
        c = color_map.get(int(row['territory_id']), 'gray')
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]], radius=4, color=c, fill=True, fill_color=c, fill_opacity=0.7, tooltip=tooltip_txt
        ).add_to(m)
    return m, map_center

REQUIRED_COLS_CUST = {
    'RouteID': ':red[RouteID (*)]', 'Customer code': ':red[Mã KH (*)]', 'Customer Name': 'Tên KH', 
    'Latitude': ':red[Latitude (Vĩ độ) (*)]', 'Longitude': ':red[Longitude (Kinh độ) (*)]', 
    'Frequency': ':red[Tần suất (*)]', 'Segment': ':red[Phân loại Segment (*)]'
}
REQUIRED_COLS_DIST = {
    'Distributor Code': ':red[Mã NPP (*)]', 'Distributor Name': 'Tên NPP', 
    'Latitude': ':red[Latitude (Vĩ độ) (*)]', 'Longitude': ':red[Longitude (Kinh độ) (*)]'
}

@st.cache_data
def calculate_haversine_distance_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_dynamic_travel_time(dist_km, speed_slow, speed_fast):
    speed_kmh = speed_slow if dist_km < 2.0 else speed_fast
    return (dist_km / speed_kmh) * 60

def calculate_dynamic_quantum(df_route, target_points=1000):
    total_time = (df_route['Visit Time (min)'] * df_route['Total_Visits_Count']).sum()
    raw_quantum = total_time / target_points
    return max(raw_quantum, 0.5)

def explode_data_by_quantum(df_week, quantum):
    df_process = df_week.copy()
    weighted_time = df_process['Visit Time (min)'] * df_process['Weight_Factor']
    df_process['quantum_points'] = np.ceil(weighted_time / quantum).fillna(1).astype(int)
    df_exploded = df_process.loc[df_process.index.repeat(df_process['quantum_points'])].copy()
    df_exploded['original_index'] = df_exploded.index
    df_exploded = df_exploded.reset_index(drop=True)
    return df_exploded, df_process['quantum_points'].sum()

def solve_saturday_strategy(df_exploded, total_points):
    coords = df_exploded[['Latitude', 'Longitude']].values.astype(np.float32)
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    n_chunks = 5
    avg_chunk_size = total_points / n_chunks
    min_size = max(1, int(avg_chunk_size * 0.90))
    max_size = int(avg_chunk_size * 1.10)
    if max_size * n_chunks < total_points: max_size = int(total_points / n_chunks) + 2
    try:
        kmeans = KMeansConstrained(n_clusters=n_chunks, size_min=min_size, size_max=max_size, random_state=42, n_init=10)
        chunk_labels = kmeans.fit_predict(coords_scaled)
    except:
        from sklearn.cluster import KMeans
        kmeans_fallback = KMeans(n_clusters=n_chunks, random_state=42, n_init=10)
        chunk_labels = kmeans_fallback.fit_predict(coords_scaled)
    days_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
    df_exploded['Day'] = [days_map[i] for i in chunk_labels]
    return df_exploded

def collapse_to_original(df_exploded, original_df):
    final_assignments = df_exploded.groupby('original_index')['Day'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Mon')
    df_result = original_df.copy()
    df_result['Assigned_Day'] = final_assignments
    df_result['Assigned_Day'].fillna('Mon', inplace=True)
    return df_result

def build_time_matrix_haversine(locations, speed_slow, speed_fast, is_ghost_mode=False):
    size = len(locations)
    matrix = np.zeros((size, size), dtype=int)
    lats = np.array([loc[0] for loc in locations])
    lons = np.array([loc[1] for loc in locations])
    for i in range(size):
        for j in range(size):
            if i == j: continue
            # Ghost mode logic: If i is Ghost Depot (index 0), dist is 0
            if is_ghost_mode and i == 0:
                matrix[i][j] = 0
            else:
                dist_km = calculate_haversine_distance_km(lats[i], lons[i], lats[j], lons[j])
                speed = speed_slow if dist_km < 2.0 else speed_fast
                matrix[i][j] = int((dist_km / speed) * 3600)
    return matrix.tolist()

def solve_tsp_final(visits, depot_coords, speed_slow, speed_fast, mode='closed', end_coords=None):
    if not visits: return []
    is_ghost = (depot_coords == (0, 0))
    locations = [depot_coords] + [v['coords'] for v in visits]
    has_end_point = (mode == 'open' and end_coords is not None)
    if has_end_point: locations.append(end_coords) 
    num_locations = len(locations)
    time_matrix = build_time_matrix_haversine(locations, speed_slow, speed_fast, is_ghost_mode=is_ghost)
    manager = pywrapcp.RoutingIndexManager(num_locations, 1, [0], [num_locations - 1] if has_end_point else [0])
    routing = pywrapcp.RoutingModel(manager)
    def time_callback(from_index, to_index):
        return time_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 2 
    solution = routing.SolveWithParameters(search_parameters)
    ordered_visits = []
    if solution:
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            if node_index != 0:
                if has_end_point and node_index == num_locations - 1: pass 
                else: ordered_visits.append(visits[node_index - 1])
            index = solution.Value(routing.NextVar(index))
    return ordered_visits

@st.cache_data(show_spinner=False)
def run_master_scheduler(df_cust, depot_coords, selected_route_ids, route_config_dict, visit_time_config, speed_config):
    SPEED_SLOW, SPEED_FAST = speed_config['slow'], speed_config['fast']
    df_cust = df_cust.copy()
    if 'RouteID' in df_cust.columns:
        df_cust['RouteID'] = df_cust['RouteID'].astype(str).str.strip()
    df_cust['Frequency'] = pd.to_numeric(df_cust['Frequency'], errors='coerce').fillna(1).round(0).astype(int)
    df_cust['Customer code'] = df_cust['Customer code'].astype(str).str.strip()
    selected_route_ids = [str(x) for x in selected_route_ids]
    df_cust_filtered = df_cust[df_cust['RouteID'].isin(selected_route_ids)].copy()
    if df_cust_filtered.empty: return pd.DataFrame() 

    cust_week_map = {} 
    f2_counter, f1_counter = 0, 0
    WEEKS = ['W1', 'W2', 'W3', 'W4']
    DAY_ORDER = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    SPACING_MAP_F8 = {'Mon': 'Thu', 'Tue': 'Fri', 'Wed': 'Mon', 'Thu': 'Tue', 'Fri': 'Wed'}
    
    for _, row in df_cust_filtered.iterrows():
        code, freq = row['Customer code'], row['Frequency']
        if freq == 2:
            cust_week_map[code] = ['W1', 'W3'] if f2_counter % 2 == 0 else ['W2', 'W4']
            f2_counter += 1
        elif freq == 1:
            cust_week_map[code] = [WEEKS[f1_counter % 4]]
            f1_counter += 1
            
    final_output_rows = []
    grouped = df_cust_filtered.groupby('RouteID')
    for route_id, route_df in grouped:
        route_df['Visit Time (min)'] = route_df['Segment'].map(visit_time_config).fillna(visit_time_config.get('default', 10.0))
        route_df['Weight_Factor'] = 1.0
        for week in WEEKS:
            week_visits_all = [] 
            for _, row in route_df.iterrows():
                freq, code = row['Frequency'], row['Customer code']
                is_in_week = False
                num_visits = 0
                if freq >= 4: num_visits, is_in_week = int(freq // 4), True
                else:
                    if week in cust_week_map.get(code, []): is_in_week, num_visits = True, 1
                if is_in_week:
                    for v_i in range(int(num_visits)):
                        r = row.copy() 
                        r['Visit_ID_Internal'] = f"{code}_{week}_{v_i}" 
                        r['Visit_Order'] = v_i
                        r['Total_Visits_Count'] = num_visits
                        week_visits_all.append(r)
            if not week_visits_all: continue
            best_df, best_score = None, float('inf')
            for iteration in range(3):
                full_df = pd.DataFrame(week_visits_all)
                df_core = full_df[full_df['Visit_Order'] == 0].copy()
                quantum = calculate_dynamic_quantum(df_core, 1200)
                df_exploded, total_pts = explode_data_by_quantum(df_core, quantum)
                df_labeled = solve_saturday_strategy(df_exploded, total_pts)
                df_core_res = collapse_to_original(df_labeled, df_core)
                anchor_map = df_core_res.set_index('Customer code')['Assigned_Day'].to_dict()
                df_dependent = full_df[full_df['Visit_Order'] > 0].copy()
                if not df_dependent.empty:
                    dep_days = []
                    for _, r_d in df_dependent.iterrows():
                        anchor = anchor_map.get(r_d['Customer code'], 'Mon')
                        day = SPACING_MAP_F8.get(anchor, 'Mon')
                        dep_days.append(day)
                    df_dependent['Assigned_Day'] = dep_days
                df_combined = pd.concat([df_core_res, df_dependent])
                day_stats, total_work = {}, 0
                for day in DAY_ORDER:
                    d_visits = df_combined[df_combined['Assigned_Day'] == day]
                    if d_visits.empty: day_stats[day] = 0; continue
                    work = d_visits['Visit Time (min)'].sum()
                    day_stats[day] = work
                    total_work += work
                unit_work = total_work / 5
                max_dev = 0
                weights = {}
                for day, act in day_stats.items():
                    tgt = unit_work
                    if tgt == 0: continue
                    ratio = act / tgt
                    max_dev = max(max_dev, abs(1 - ratio))
                    weights[day] = max(0.5, min(1 + (ratio - 1) * 0.7, 2.0))
                if max_dev < best_score: best_score, best_df = max_dev, df_combined.copy()
                if max_dev <= 1.10: break 
            
            for day in DAY_ORDER:
                d_visits = best_df[best_df['Assigned_Day'] == day]
                if d_visits.empty: continue
                tsp_in = []
                for _, row in d_visits.iterrows():
                    d = row.to_dict()
                    d['coords'] = (row['Latitude'], row['Longitude'])
                    tsp_in.append(d)
                end_cfg = route_config_dict.get(route_id)
                mode, end_c = ('open', end_cfg) if end_cfg else ('closed', None)
                ordered = solve_tsp_final(tsp_in, depot_coords, SPEED_SLOW, SPEED_FAST, mode, end_c)
                
                # Dynamic start logic: If Ghost Depot, start sequence travel time at 0
                prev, seq, agg_time = depot_coords, 1, 0
                for item in ordered:
                    curr = item['coords']
                    if depot_coords == (0, 0) and seq == 1:
                        travel = 0
                        dist = 0
                    else:
                        dist = calculate_haversine_distance_km(prev[0], prev[1], curr[0], curr[1])
                        travel = get_dynamic_travel_time(dist, SPEED_SLOW, SPEED_FAST)
                    agg_time += travel + item['Visit Time (min)']
                    res = item.copy()
                    res.update({'RouteID': route_id, 'Week': week, 'Day': day, 'Week&Day': f"{week}-{day}",
                                'Sequence': seq, 'Travel Time (min)': round(travel, 2),
                                'Distance (km)': round(dist, 2) if 'dist' in locals() else 0, 'Total Workload (min)': round(agg_time, 2)})
                    for k in ['coords', 'angle', 'Weight_Factor', 'quantum_points']: 
                        if k in res: del res[k]
                    final_output_rows.append(res)
                    prev, seq = curr, seq+1
                    
    df_final = pd.DataFrame(final_output_rows)
    df_final = df_final.merge(df_cust, on='Customer code', how='left', suffixes=('', '_dup'))
    df_final = df_final.drop(columns=[c for c in df_final.columns if c.endswith('_dup')])
    df_final['Day'] = pd.Categorical(df_final['Day'], categories=DAY_ORDER, ordered=True)
    return df_final.sort_values(by=['RouteID', 'Week', 'Day', 'Sequence'])

def recalculate_routes(df_edited, depot_coords, route_config, speed_config, impacted_groups=None):
    SPEED_SLOW, SPEED_FAST = speed_config['slow'], speed_config['fast']
    new_rows = []
    for (r_id, week, day), group in df_edited.groupby(['RouteID', 'Week', 'Day']):
        should_optimize = True
        if impacted_groups is not None: should_optimize = (r_id, week, day) in impacted_groups
        if should_optimize:
            tsp_input = []
            for _, row in group.iterrows():
                d = row.to_dict()
                d['coords'] = (row['Latitude'], row['Longitude'])
                tsp_input.append(d)
            end_cfg = route_config.get(r_id)
            mode, end_c = ('open', end_cfg) if end_cfg else ('closed', None)
            ordered = solve_tsp_final(tsp_input, depot_coords, SPEED_SLOW, SPEED_FAST, mode, end_c)
        else:
            ordered = [row.to_dict() for _, row in group.sort_values('Sequence').iterrows()]
            for item in ordered: item['coords'] = (item['Latitude'], item['Longitude'])
        prev, seq, agg_time = depot_coords, 1, 0
        for item in ordered:
            curr = item['coords']
            if depot_coords == (0, 0) and seq == 1:
                travel, dist = 0, 0
            else:
                dist = calculate_haversine_distance_km(prev[0], prev[1], curr[0], curr[1])
                travel = get_dynamic_travel_time(dist, SPEED_SLOW, SPEED_FAST)
            agg_time += travel + item['Visit Time (min)']
            res = item.copy()
            res.update({'Sequence': seq, 'Travel Time (min)': round(travel, 2),
                        'Distance (km)': round(dist, 2), 'Total Workload (min)': round(agg_time, 2)})
            if 'coords' in res: del res['coords']
            new_rows.append(res)
            prev, seq = curr, seq+1
    return pd.DataFrame(new_rows)

def get_changed_visits(df_orig, df_curr):
    if df_orig is None or df_curr is None: return []
    df1 = df_orig.set_index('Visit_ID_Internal')[['Week', 'Day']].sort_index()
    df2 = df_curr.set_index('Visit_ID_Internal')[['Week', 'Day']].sort_index()
    common = df1.index.intersection(df2.index)
    diff = (df1.loc[common] != df2.loc[common]).any(axis=1)
    return diff[diff].index.tolist()

@st.cache_data
def create_template_excel(is_dist=False):
    if is_dist:
        data = { 'Mã NPP': ['12345678'], 'Tên NPP': ['NPP Thành Phát'], 'Vĩ độ (Latitude)': [0.0], 'Kinh độ (Longitude)': [0.0] }
    else:
        data = { 'RouteID': ['VN123456'], 'Mã KH': ['12345678'], 'Tên KH': ['Tạp Hóa A'], 'Vĩ độ (Latitude)': [10.77], 'Kinh độ (Longitude)': [106.70], 'Tần suất': [4], 'Phân loại Segment': ['Gold'], 'Thêm các cột khác tùy ý': [''] }
    df_sample = pd.DataFrame(data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer: df_sample.to_excel(writer, sheet_name='Template', index=False)
    return output.getvalue()

@st.cache_data
def create_template_v1():
    df = pd.DataFrame({ "Mã KH (*)": ["KH01", "KH02"], "Vĩ độ (Latitude) (*)": [10.7, 10.8], "Kinh độ (Longitude) (*)": [106.6, 106.7], "Tên KH": ["A", "B"], "Địa chỉ": ["HCM", "HCM"], "VolEC": [100, 200], "Tần suất": [4, 8], "Phân loại Segment": ["MT", "Cooler"] })
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer: df.to_excel(writer, index=False)
    return output.getvalue()

@st.cache_data
def create_template_v2():
    df = pd.DataFrame({ "Mã KH (*)": ["KH01", "KH02"], "Vĩ độ (Latitude) (*)": [10.7, 10.8], "Kinh độ (Longitude) (*)": [106.6, 106.7], "Tên KH": ["A", "B"], "Địa chỉ": ["HCM", "HCM"], "VolEC": [100, 200], "Tần suất (*)": [4, 8], "Phân loại Segment (*)": ["MT", "Cooler"] })
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer: df.to_excel(writer, index=False)
    return output.getvalue()

@st.cache_data
def to_excel_tp(df_export):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_export_clean = df_export.copy()
        df_export_clean = df_export_clean.drop(columns=[c for c in ['workload_min', 'weight_points', 'Trạng thái', 'Bỏ chọn'] if c in df_export_clean.columns])
        df_export_clean.to_excel(writer, sheet_name='Details', index=False)
    return output.getvalue()

@st.cache_data
def to_excel_output(df_master):
    output = io.BytesIO()
    df_export = df_master.drop(columns=['Visit_ID_Internal'], errors='ignore').copy()
    for c in ['Bỏ chọn', 'Đã sửa', 'Chọn', 'Trạng thái']:
        if c in df_export.columns: df_export = df_export.drop(columns=[c])
    df_export = df_export.sort_values(by=['RouteID', 'Week', 'Day', 'Sequence'])
    df_export['Agg_Dist'] = df_export.groupby(['RouteID', 'Week', 'Day'])['Distance (km)'].cumsum()
    df_export['Agg_Travel'] = df_export.groupby(['RouteID', 'Week', 'Day'])['Travel Time (min)'].cumsum()
    rename_map = { 'Day': 'Ngày', 'Week': 'Tuần', 'Week&Day': 'Ngày & Tuần', 'Sequence': 'Thứ tự', 'Distance (km)': 'Khoảng cách từ KH trước', 'Travel Time (min)': 'Thời gian di chuyển từ KH trước', 'Agg_Dist': 'Khoảng cách từ đầu ngày', 'Agg_Travel': 'Thời gian di chuyển từ đầu ngày', 'Visit Time (min)': 'Thời gian viếng thăm điểm bán', 'Total Workload (min)': 'Tổng thời gian làm việc từ đầu ngày' }
    df_export_final = df_export.rename(columns=rename_map)
    df_sum = df_master.groupby(['RouteID', 'Week', 'Day']).agg( Total_TIO_min=('Visit Time (min)', 'sum'), Total_TBO_min=('Travel Time (min)', 'sum'), Num_Customers=('Customer code', 'count') ).reset_index()
    df_sum['Total_Workload_min'] = df_sum['Total_TIO_min'] + df_sum['Total_TBO_min']
    df_sum['Total_TIO_h'] = (df_sum['Total_TIO_min'] / 60).round(2)
    df_sum['Total_TBO_h'] = (df_sum['Total_TBO_min'] / 60).round(2)
    df_sum['Total_Workload_h'] = (df_sum['Total_Workload_min'] / 60).round(2)
    sum_rename = { 'Week': 'Tuần', 'Day': 'Ngày', 'Total_TIO_h': 'Tổng thời gian viếng thăm (Giờ)', 'Total_TBO_h': 'Tổng thời gian di chuyển (Giờ)', 'Num_Customers': 'Số KH', 'Total_Workload_h': 'Tổng thời gian làm việc (Giờ)' }
    df_sum_final = df_sum.rename(columns=sum_rename)
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_export_final.to_excel(writer, sheet_name='Lịch viếng thăm', index=False)
        df_sum_final.to_excel(writer, sheet_name='Tổng quan', index=False)
    return output.getvalue()

def create_folium_map(df_filtered_dict, col_mapping):
    df_filtered = pd.DataFrame.from_dict(df_filtered_dict)
    if df_filtered.empty: return None
    center = [df_filtered['Latitude'].mean(), df_filtered['Longitude'].mean()]
    m = folium.Map(location=center, zoom_start=13, tiles=ESRI_URL, attr=ESRI_ATTR)
    legend_html = ''' <div style="position: fixed; bottom: 30px; left: 30px; width: 80px; height: 100px; border:2px solid grey; z-index:9999; font-size:12px; background-color:white; padding: 10px; opacity: 0.9;"> <b>Chú giải:</b><br> <i style="background:red; width:10px; height:10px; display:inline-block;"></i> T2<br> <i style="background:green; width:10px; height:10px; display:inline-block;"></i> T3<br> <i style="background:blue; width:10px; height:10px; display:inline-block;"></i> T4<br> <i style="background:orange; width:10px; height:10px; display:inline-block;"></i> T5<br> <i style="background:purple; width:10px; height:10px; display:inline-block;"></i> T6<br> </div> '''
    m.get_root().html.add_child(folium.Element(legend_html))
    color_map = {'T2': 'red', 'T3': 'green', 'T4': 'blue', 'T5': 'orange', 'T6': 'purple'}
    for (r, w, d), group in df_filtered.groupby(['RouteID', 'Week', 'Day']):
        color = color_map.get(d, 'gray')
        grp = group.sort_values('Sequence')
        folium.PolyLine(grp[['Latitude', 'Longitude']].values.tolist(), color=color, weight=3, opacity=0.7).add_to(m)
        for _, row in grp.iterrows():
            tooltip_parts = [f"<b>Mã KH:</b> {row['Customer code']}", f"<b>Tên:</b> {row['Customer Name']}", f"<b>Thứ tự:</b> {row['Sequence']}", f"<b>Tuần:</b> {row['Week']}", f"<b>Ngày:</b> {row['Day']}"]
            folium.Marker( location=(row['Latitude'], row['Longitude']), icon=folium.DivIcon(html=f"""<div style="background:{color};color:white;border-radius:50%;width:20px;height:20px;text-align:center;font-size:12px;font-weight:bold;line-height:20px;border:1px solid white;">{row['Sequence']}</div>"""), tooltip="<br>".join(tooltip_parts) ).add_to(m)
    return m

@st.cache_data
def create_heatmap(df_dict, value_col, agg_mode, fmt="{:.1f}", title="Heatmap"):
    df_data = pd.DataFrame.from_dict(df_dict)
    weeks, days = ['W1', 'W2', 'W3', 'W4'], ['T2', 'T3', 'T4', 'T5', 'T6']
    if agg_mode == 'count': pivot = df_data.pivot_table(index='Week', columns='Day', values=value_col, aggfunc='count')
    elif agg_mode == 'sum_time': pivot = df_data.pivot_table(index='Week', columns='Day', values=value_col, aggfunc=lambda x: x.sum()/60)
    elif agg_mode == 'mean_time': pivot = df_data.pivot_table(index='Week', columns='Day', values=value_col, aggfunc=lambda x: x.mean()/60)
    elif agg_mode == 'mean_qty': pivot = df_data.pivot_table(index='Week', columns='Day', values=value_col, aggfunc='mean')
    pivot = pivot.reindex(index=weeks, columns=days).fillna(0)
    pivot.index.name = None
    st.markdown(f"**{title}**")
    try:
        styled_df = pivot.style.format(fmt).background_gradient(cmap='RdYlGn_r', axis=None)
        st.dataframe(styled_df, height=140, width="stretch", column_config={col: st.column_config.Column(width="small") for col in days})
    except: st.dataframe(pivot.style.format(fmt), height=140, width="stretch")

def render_sidebar():
    st.sidebar.markdown("### Thực hiện tác vụ:")
    check_tp = st.sidebar.checkbox("1. Chia địa bàn (Territory Planner)", value=st.session_state.global_state['config']['is_tp'])
    tp_mode_sel = "Chế độ 1"
    if check_tp:
        with st.sidebar.columns([0.15, 0.85])[1]:
            mode_val = st.radio("Mode", ["Chế độ 1: Cân bằng Số lượng KH", "Chế độ 2: Cân bằng Workload"], label_visibility="collapsed")
            tp_mode_sel = "Chế độ 1" if "Chế độ 1" in mode_val else "Chế độ 2"
    check_vp = st.sidebar.checkbox("2. Xếp lịch viếng thăm (Visit Planner)", value=st.session_state.global_state['config']['is_vp'])
    is_integrated = (check_tp and check_vp)
    if is_integrated: st.sidebar.info("💡 Tự động xếp lịch sau chia địa bàn.")
    st.sidebar.divider()
    
    if st.sidebar.button("Bắt đầu", type="primary"):
        if not check_tp and not check_vp: st.sidebar.error("Vui lòng chọn tác vụ!")
        else:
            if not st.session_state.global_state['has_started']:
                st.session_state.global_state['config'].update({'is_tp': check_tp, 'is_vp': check_vp, 'is_integrated': is_integrated, 'tp_mode': tp_mode_sel})
                st.session_state.global_state['has_started'] = True
                st.rerun()
            else: st.session_state.show_reset_warning = True

    if st.session_state.show_reset_warning:
        st.markdown('<div class="warning-box">⚠️ Cảnh báo: Việc nhấn "Bắt đầu" sẽ xóa toàn bộ dữ liệu. Bạn chắc chắn?</div>', unsafe_allow_html=True)
        c_can, c_con = st.columns(2)
        if c_can.button("Hủy"): st.session_state.show_reset_warning = False; st.rerun()
        if c_con.button("Tiếp tục"): st.session_state.global_state['has_started'] = False; st.session_state.show_reset_warning = False; st.rerun()

def main():
    render_sidebar()
    step = st.session_state.global_state['step']
    config = st.session_state.global_state['config']
    
    if not st.session_state.global_state['has_started']:
        render_welcome_screen()
    else:
        render_main_title()
        
        # --- LUỒNG XỬ LÝ ĐẦY ĐỦ ---
        # 1. Territory Planner Only
        if config['is_tp'] and not config['is_integrated']:
            render_tp_ui(is_integrated=False)
        
        # 2. Visit Planner Only
        elif config['is_vp'] and not config['is_integrated']:
            render_vp_ui(is_integrated=False)
                
        # 3. Integrated (Cả hai)
        elif config['is_integrated']:
            if step == 'input_integrated':
                render_vp_ui(is_integrated=True)
            elif step in ['tp_setup', 'tp_result_integrated']:
                render_tp_ui(is_integrated=True)
            elif step in ['vp_process', 'vp_result']:
                render_vp_ui(is_integrated=True)
if __name__ == "__main__":
    main()
