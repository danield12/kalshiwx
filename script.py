import pandas as pd
import numpy as np
import requests
import pickle
import os
import re
import time
import json
import warnings
from datetime import datetime, timedelta
from io import StringIO
from sklearn.ensemble import HistGradientBoostingRegressor
import pytz

# Suppress warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
WUNDERGROUND_API_KEY = 'e1f10a1e78da46f5b10a1e78da96f525' 

LOCATIONS = {
    "NYC": {"lat": 40.7851, "lon": -73.9683, "tz": "America/New_York", "station_id": "KNYC", "kalshi": "KXHIGHNY", "name": "New York (Central Park)"},
    "MDW": {"lat": 41.7868, "lon": -87.7522, "tz": "America/Chicago", "station_id": "KMDW", "kalshi": "KXHIGHCHI", "name": "Chicago Midway"},
    "LAX": {"lat": 33.9380, "lon": -118.4085, "tz": "America/Los_Angeles", "station_id": "KLAX", "kalshi": "KXHIGHLAX", "name": "Los Angeles"},
    "PHL": {"lat": 39.8729, "lon": -75.2437, "tz": "America/New_York", "station_id": "KPHL", "kalshi": "KXHIGHPHIL", "name": "Philadelphia"},
    "AUS": {"lat": 30.1975, "lon": -97.6664, "tz": "America/Chicago", "station_id": "KAUS", "kalshi": "KXHIGHAUS", "name": "Austin"},
    "MIA": {"lat": 25.7959, "lon": -80.2870, "tz": "America/New_York", "station_id": "KMIA", "kalshi": "KXHIGHMIA", "name": "Miami"},
    "DEN": {"lat": 39.8561, "lon": -104.6737, "tz": "America/Denver", "station_id": "KDEN", "kalshi": "KXHIGHDEN", "name": "Denver"},
    "CLT": {"lat": 35.2144, "lon": -80.9473, "tz": "America/New_York", "station_id": "KCLT", "kalshi": "KXHIGHCLT", "name": "Charlotte"} 
}

HUB_CONFIG = {
    "KNYC": ['KNYNEWYO1115', 'KNYNEWYO270', 'KNYNEWYO1596', 'KNYNEWYO1686', 'KNYNEWYO1796'],
    "KMDW": ['KILCHICA37', 'KILCHICA1000', 'KILCHICA711', 'KILCHICA954', 'KILCHICA1223'],
    "KLAX": ['KCAELSEG28', 'KCAELSEG33', 'KCAELSEG31', 'KCAELSEG23', 'KCALOSAN958'],
    "KAUS": ['KTXAUSTI3939', 'KTXAUSTI2283', 'KTXDELVA37', 'KTXAUSTI1864', 'KTXAUSTI2523'],
    "KDEN": ['KCOAUROR983', 'KCOAUROR870', 'KCOAUROR879', 'KCOCOMME103', 'KCOAUROR940'],
    "KMIA": ['KFLMIAMI922', 'KFLMIAMI661', 'KFLMIAMI69', 'KFLMIAMI448', 'KFLMIAMI706'],
    "KPHL": ['KNJWESTD8', 'KNJNATIO5', 'KPAPROSP26', 'KPAPROSP27', 'KPAPENNS14'],
    "KCLT": ['KNCCHARL604', 'KNCCHARL1304', 'KNCCHARL1214', 'KNCCHARL1218', 'KNCCATAW11']
}

GLOBAL_MODELS = {
    "HRRR": "gfs_hrrr",
    "ECMWF": "ecmwf_ifs025",
    "GFS": "gfs_global",
    "ICON": "icon_global",
    "GEM": "gem_global"
}

AIRPORT_WEATHER_FILES = ["airportswxpart1.txt", "airportswxpart2.txt"] 
INPUT_FILE_FORMAT = "{station_id}_2025_Full_Year.csv" 

# ==============================================================================
# 2. HGBR TRAINING & LIVE INFERENCE
# ==============================================================================
def train_model_if_missing(hub_id):
    model_path = f"{hub_id}_HGBR_Residual_Model.pkl"
    feat_path = f"{hub_id}_HGBR_Features.pkl"
    
    if os.path.exists(model_path) and os.path.exists(feat_path): return True 
    if not any(os.path.exists(f) for f in AIRPORT_WEATHER_FILES): return False

    print(f"   [ML] Training new model for {hub_id}...")
    master_content = ""
    for path in AIRPORT_WEATHER_FILES:
        if os.path.exists(path):
            with open(path, 'r') as f: master_content += f.read()
    
    df_targets = pd.read_csv(StringIO(master_content))
    df_targets.columns = [c.strip() for c in df_targets.columns]
    
    if 'tmpf' in df_targets.columns:
        df_targets['Target_Temp'] = pd.to_numeric(df_targets['tmpf'], errors='coerce')
        df_targets['Timestamp'] = pd.to_datetime(df_targets['valid(UTC)'], utc=True)
    
    iata_code = next((k for k, v in LOCATIONS.items() if v['station_id'] == hub_id), None)
    df_target_hub = df_targets[df_targets['station'] == iata_code].sort_values('Timestamp')
    if df_target_hub.empty: return False

    pws_dfs, valid_pws = [], []
    for pws_id in HUB_CONFIG.get(hub_id, []):
        p_file = INPUT_FILE_FORMAT.format(station_id=pws_id)
        if os.path.exists(p_file):
            try:
                df_p = pd.read_csv(p_file)
                df_p['Timestamp'] = pd.to_datetime(df_p['Timestamp_UTC'], utc=True)
                df_p = df_p.set_index('Timestamp')[['Temp_F']].rename(columns={'Temp_F': pws_id})
                if len(df_p) > 50: 
                    pws_dfs.append(df_p)
                    valid_pws.append(pws_id)
            except: pass
    
    if not valid_pws: return False

    df_feats = pd.concat(pws_dfs, axis=1).sort_index().ffill().bfill().reset_index()
    df_final = pd.merge_asof(df_target_hub, df_feats, on='Timestamp', direction='nearest', tolerance=pd.Timedelta('30min')).dropna(subset=valid_pws + ['Target_Temp'])
    
    df_final['PWS_Mean'] = df_final[valid_pws].mean(axis=1)
    df_final['Target_Residual'] = df_final['Target_Temp'] - df_final['PWS_Mean']
    df_final['Hour_Sin'] = np.sin(2 * np.pi * df_final['Timestamp'].dt.hour / 24)
    df_final['Hour_Cos'] = np.cos(2 * np.pi * df_final['Timestamp'].dt.hour / 24)
    df_final['Month'] = df_final['Timestamp'].dt.month
    
    features = valid_pws + ['PWS_Mean', 'Hour_Sin', 'Hour_Cos', 'Month']
    
    model = HistGradientBoostingRegressor(max_iter=150, max_depth=6, random_state=42)
    model.fit(df_final[features], df_final['Target_Residual'])
    
    with open(model_path, 'wb') as f: pickle.dump(model, f)
    with open(feat_path, 'wb') as f: pickle.dump(features, f)
    return True

def get_live_ml_prediction(hub_id):
    model_exists = train_model_if_missing(hub_id)
    model_path = f"{hub_id}_HGBR_Residual_Model.pkl"
    feat_path = f"{hub_id}_HGBR_Features.pkl"
    
    pws_vals, valid_vals = {}, []
    for pws_id in HUB_CONFIG.get(hub_id, []):
        url = f"https://api.weather.com/v2/pws/observations/current?stationId={pws_id}&format=json&units=e&apiKey={WUNDERGROUND_API_KEY}"
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                data = r.json()
                if 'observations' in data and len(data['observations']) > 0:
                    val = data['observations'][0].get('imperial', {}).get('temp')
                    if val is not None:
                        pws_vals[pws_id] = val
                        valid_vals.append(val)
        except: pass
        time.sleep(0.02) 

    if not valid_vals: return "No Data", "No Data"
    live_mean = np.mean(valid_vals)
    
    if not model_exists: return f"{live_mean:.1f}", "Avg (No Model)"
    
    try:
        with open(model_path, 'rb') as f: model = pickle.load(f)
        with open(feat_path, 'rb') as f: features = pickle.load(f)
        
        input_data = {}
        now = datetime.now(pytz.utc)
        for f in features:
            if f in pws_vals: input_data[f] = pws_vals[f]
            elif f == 'PWS_Mean': input_data[f] = live_mean
            elif f == 'Hour_Sin': input_data[f] = np.sin(2 * np.pi * now.hour / 24)
            elif f == 'Hour_Cos': input_data[f] = np.cos(2 * np.pi * now.hour / 24)
            elif f == 'Month': input_data[f] = now.month
            else: input_data[f] = live_mean 
            
        df_in = pd.DataFrame([input_data])[features]
        pred_residual = model.predict(df_in)[0]
        final_pred = live_mean + pred_residual
        return f"{final_pred:.1f}", f"HGBR ({pred_residual:+.1f})"
    except: return f"{live_mean:.1f}", "Avg (Error)"

# ==============================================================================
# 3. FORECASTING ENGINES
# ==============================================================================
def track_model_history(station_code, model_name, today_val, tmw_val):
    """
    Saves history for ANY model to a JSON file. 
    Only saves a new entry if the values differ from the last saved entry 
    (to avoid duplicates on frequent script runs).
    """
    if not os.path.exists("Kalshi"): os.makedirs("Kalshi")
    file_path = f"Kalshi/{station_code}_{model_name}_log.json"
    history = []
    
    # 1. Load existing
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f: history = json.load(f)
        except: pass

    # 2. Check if new data is different from the absolute latest entry
    is_new = True
    if history:
        last = history[0]
        # If values match exactly, assume it's the same run cycle (no update)
        if last['today'] == today_val and last['tmw'] == tmw_val:
            is_new = False
            # Update timestamp to show it was verified recently? 
            # No, keep original timestamp to show when that run FIRST appeared.

    if is_new:
        now_utc = datetime.now(pytz.utc)
        new_entry = {
            'ts': now_utc.strftime('%Y-%m-%d %H:%M'), # For debugging
            'label': now_utc.strftime('%H:%M'),       # For display
            'today': today_val, 
            'tmw': tmw_val
        }
        history.insert(0, new_entry)
        
    # 3. Prune (Keep last 10, we only display top 2 but keep a buffer)
    history = history[:10]
    
    # 4. Save
    with open(file_path, 'w') as f: json.dump(history, f)
    
    return history

def get_lamp_data(station, tz_str):
    history = []
    now_utc = datetime.now(pytz.utc)
    
    tz = pytz.timezone(tz_str)
    local_now = datetime.now(tz)
    local_today = local_now.date()
    local_tomorrow = local_today + timedelta(days=1)
    best_today, best_tmw = None, None
    
    def parse(text):
        lines = text.split('\n')
        utc, tmp = [], []
        for line in lines:
            parts = line.split()
            if parts and parts[0] == 'UTC': utc = parts[1:]
            elif parts and parts[0] == 'TMP': tmp = parts[1:]
        if not utc or not tmp: return None
        return pd.DataFrame({'UTC': utc[:len(tmp)], 'TMP': tmp})

    def extract_header_date(text):
        match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})\s+\d{4}\s+UTC', text)
        if match:
            try: return datetime.strptime(match.group(1), '%m/%d/%Y').date()
            except: return None
        return None

    found_anchor = False
    for i in range(18):
        check_hr = (now_utc.hour - i) % 24
        url = f"https://lamp.mdl.nws.noaa.gov/lamp/meteo/bullpop.php?sta={station}&forecast_time={check_hr:02d}"
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200 and "LAMP" in r.text:
                text = r.text
                header_date = extract_header_date(text)
                if not found_anchor:
                    if header_date and header_date != now_utc.date(): continue 
                    found_anchor = True
                if len(history) >= 12: break
                df = parse(text)
                if df is not None:
                    vals = pd.to_numeric(df['TMP'], errors='coerce')
                    offset = 1 if (int(df['UTC'].iloc[0]) < check_hr) else 0
                    prev = -1
                    target_today, target_tmw = [], []
                    base_date = header_date if header_date else now_utc.date()
                    for h_str, t_val in zip(df['UTC'], vals):
                        h = int(h_str)
                        if prev == 23 and h == 0: offset += 1
                        dt_utc = datetime(base_date.year, base_date.month, base_date.day, h, 0, 0, tzinfo=pytz.utc) + timedelta(days=offset)
                        dt_local = dt_utc.astimezone(tz)
                        prev = h
                        if dt_local.hour in [13,14,15,16,17]: 
                            if dt_local.date() == local_today: target_today.append(t_val)
                            if dt_local.date() == local_tomorrow: target_tmw.append(t_val)
                    ht = max(target_today) if target_today else None
                    htm = max(target_tmw) if target_tmw else None
                    history.append({'Run': f"{check_hr:02d}z", 'Today': ht, 'Tmw': htm})
                    if best_today is None: best_today = ht
                    if best_tmw is None: best_tmw = htm
        except: pass
    return best_today, best_tmw, history

_MOS_CACHE = {}
def get_mos(station, tz_str):
    runs = ['18', '12', '06', '00']
    valid_runs = []
    
    for r in runs:
        url = f"https://www.weather.gov/source/mdl/MOS/GFSMAV.t{r}z"
        if url not in _MOS_CACHE:
            try: 
                resp = requests.get(url, timeout=2)
                if resp.status_code == 200: _MOS_CACHE[url] = resp.text
                else: _MOS_CACHE[url] = ""
            except: continue

        text = _MOS_CACHE.get(url, "")
        if not text: continue
        match = re.search(rf"({station}\s+GFS.*?)(?=\n[A-Z]{{4}}|\Z)", text, re.DOTALL)
        if not match: continue
        block = match.group(1)
        
        date_match = re.search(r"(\d{1,2}/\d{1,2}/\d{4})\s+(\d{4})\s+UTC", block)
        if not date_match: continue
        
        try:
            dt_str = f"{date_match.group(1)} {date_match.group(2)}"
            run_dt = datetime.strptime(dt_str, "%m/%d/%Y %H%M").replace(tzinfo=pytz.utc)
            
            lines = block.split('\n')
            hr_line = next((l for l in lines if l.strip().startswith('HR')), None)
            tmp_line = next((l for l in lines if l.strip().startswith('TMP')), None)
            
            if hr_line and tmp_line:
                hours = [int(x) for x in hr_line.split()[1:]]
                temps = [int(x) for x in tmp_line.split()[1:]]
                
                curr_date = run_dt.date()
                run_hour = run_dt.hour
                if hours[0] < run_hour: curr_date += timedelta(days=1)
                
                data_points = []
                prev_h = -1
                for h, t in zip(hours, temps):
                    if h < prev_h: curr_date += timedelta(days=1)
                    dt_utc = datetime(curr_date.year, curr_date.month, curr_date.day, h, 0, 0, tzinfo=pytz.utc)
                    dt_local = dt_utc.astimezone(pytz.timezone(tz_str))
                    data_points.append({'date': dt_local.date(), 'hour': dt_local.hour, 'temp': t})
                    prev_h = h

                local_today = datetime.now(pytz.timezone(tz_str)).date()
                local_tmw = local_today + timedelta(days=1)
                today_temps = [d for d in data_points if d['date'] == local_today]
                tmw_temps = [d for d in data_points if d['date'] == local_tmw]

                t_high = None
                if today_temps:
                    if today_temps[0]['hour'] <= 14: 
                        t_high = max(d['temp'] for d in today_temps)
                
                tm_high = max(d['temp'] for d in tmw_temps) if tmw_temps else None
                
                if t_high is not None or tm_high is not None:
                    valid_runs.append({'run_str': f"{r}z", 'dt': run_dt, 'today': t_high, 'tmw': tm_high})
        except: continue
    valid_runs.sort(key=lambda x: x['dt'], reverse=True)
    return valid_runs

def get_nws_official(lat, lon, tz_str):
    try:
        r = requests.get(f"https://api.weather.gov/points/{lat},{lon}", headers={'User-Agent':'myapp'}, timeout=3)
        forecast_url = r.json()['properties']['forecast']
        r = requests.get(forecast_url, headers={'User-Agent':'myapp'}, timeout=3)
        periods = r.json()['properties']['periods']
        
        tz = pytz.timezone(tz_str)
        today_date = datetime.now(tz).date()
        tmw_date = today_date + timedelta(days=1)
        t_high, tm_high = None, None
        
        for p in periods:
            if p['isDaytime']:
                p_dt = datetime.fromisoformat(p['startTime']).astimezone(tz).date()
                if p_dt == today_date and t_high is None: t_high = p['temperature']
                if p_dt == tmw_date and tm_high is None: tm_high = p['temperature']
        return t_high, tm_high
    except: return None, None

def get_global_model(lat, lon, tz, model_code):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": lat, "longitude": lon, "hourly": "temperature_2m", "timezone": tz, "forecast_days": 3, "models": model_code}
        r = requests.get(url, params=params, timeout=3)
        data = r.json()['hourly']
        df = pd.DataFrame({'time': data['time'], 'temp': data['temperature_2m']})
        df['dt'] = pd.to_datetime(df['time'])
        df['date'] = df['dt'].dt.date
        today = datetime.now(pytz.timezone(tz)).date()
        tmw = today + timedelta(days=1)
        t1 = df[df['date'] == today]['temp'].max()
        t2 = df[df['date'] == tmw]['temp'].max()
        if not pd.isna(t1): t1 = (t1 * 9/5) + 32
        if not pd.isna(t2): t2 = (t2 * 9/5) + 32
        return round(t1, 1), round(t2, 1)
    except: return None, None

def get_kalshi():
    print("\n--- Fetching Kalshi Markets ---")
    url = "https://api.elections.kalshi.com/trade-api/v2/markets"
    today_code = datetime.now().strftime("%y%b%d").upper()
    tmw_code = (datetime.now() + timedelta(days=1)).strftime("%y%b%d").upper()
    data = []
    kalshi_map = {v['kalshi']: k for k, v in LOCATIONS.items()}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    for kt, city in kalshi_map.items():
        try:
            r = requests.get(url, params={"series_ticker": kt, "status": "open"}, headers=headers, timeout=5)
            for m in r.json().get('markets', []):
                try:
                    day_type = None
                    if today_code in m['ticker']: day_type = 'Today'
                    elif tmw_code in m['ticker']: day_type = 'Tomorrow'
                    
                    if day_type:
                        sub = m.get('subtitle','')
                        last_part = m['ticker'].split('-')[-1]
                        s_val_match = re.search(r'([\d.]+)', last_part)
                        if s_val_match:
                            s_val = float(s_val_match.group(1))
                            low, high = s_val, s_val + 0.9
                            if not sub:
                                s_type = 'B' if 'B' in last_part else 'T'
                                if s_type == 'B': sub = f"{int(s_val)}° to {int(s_val)+1}°"
                                else: sub = f"Threshold {s_val}°"

                            if 'below' in sub or 'less' in sub: high = s_val; low = -999
                            elif 'above' in sub or 'greater' in sub: low = s_val; high = 999
                            elif 'to' in sub:
                                 try:
                                     parts = sub.replace('°','').split(' to ')
                                     low, high = float(parts[0]), float(parts[1])
                                 except: pass
                            data.append({'Airport': city, 'Range': sub, 'Price': m['yes_bid'], 'Low': low, 'High': high, 'Day': day_type})
                except: continue
        except: pass
    return pd.DataFrame(data)

# ==============================================================================
# 4. WEIGHTED BLEND LOGIC
# ==============================================================================
def calculate_weighted_blend(city_rows):
    weights = {
        'NWS Official': 5.0, 
        'LAMP': 3.5,         
        'GFS MOS': 2.0,      
        'HRRR': 1.5,         
        'ECMWF': 1.0,        
        'GFS': 0.5,
        'GEM': 0.5,
        'ICON': 0.5
    }

    t1_sum, t1_w_sum = 0, 0
    t2_sum, t2_w_sum = 0, 0

    for r in city_rows:
        model_name = r['Model']
        if "Prev" in model_name: continue # Ignore history in blend

        w = weights.get(model_name, 0.5) 
        
        if r['Today'] is not None:
            t1_sum += r['Today'] * w
            t1_w_sum += w
            
        if r['Tomorrow'] is not None:
            t2_sum += r['Tomorrow'] * w
            t2_w_sum += w
            
    today_blend = t1_sum / t1_w_sum if t1_w_sum > 0 else None
    tmw_blend = t2_sum / t2_w_sum if t2_w_sum > 0 else None
    
    return today_blend, tmw_blend

def build_html(full_data, ml_preds, kalshi_df, history):
    now_utc = datetime.now(pytz.utc)
    now_est = now_utc.astimezone(pytz.timezone('US/Eastern'))
    time_est_str = now_est.strftime('%Y-%m-%d %I:%M:%S %p %Z')
    time_utc_str = now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')

    html = """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Weather Master V17</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script>
    function toggleKalshi(city) {
        var today = document.getElementById(city + '-today');
        var tmw = document.getElementById(city + '-tmw');
        var btn = document.getElementById(city + '-btn');
        if (today.style.display === 'none') {
            today.style.display = 'table-row-group'; 
            tmw.style.display = 'none'; 
            btn.innerText = 'Show Tomorrow';
            btn.classList.remove('btn-cyan');
            btn.classList.add('btn-outline-light');
        } else {
            today.style.display = 'none'; 
            tmw.style.display = 'table-row-group'; 
            btn.innerText = 'Show Today';
            btn.classList.remove('btn-outline-light');
            btn.classList.add('btn-cyan');
        }
    }
    </script>
    <style>
        body { background:#0f172a; color:#f1f5f9; font-family: 'Segoe UI', system-ui, sans-serif; } 
        .card { background:#1e293b; border:1px solid #334155; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border-radius: 8px; } 
        .card-header { background:#334155; border-bottom: 1px solid #475569; font-weight: 600; color: #e2e8f0; }
        .table { color:#e2e8f0; font-size:0.85rem; width: 100%; table-layout: fixed; margin-bottom: 0; } 
        .table thead th { 
            color: #ffffff !important; 
            background-color: #0f172a !important; 
            border-bottom: 2px solid #64748b; 
            vertical-align: middle; 
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.5px;
        }
        .table-striped tbody tr:nth-of-type(odd) { background-color: rgba(255,255,255,0.03); }
        .table td { border-color: #334155; vertical-align: middle; }
        .badge-ml { background:#0ea5e9; color: white; font-size: 0.9rem; }
        .kalshi-btn { font-size: 0.7rem; padding: 2px 10px; margin-left: 10px; }
        .btn-cyan { background-color: #06b6d4; color: white; border: none; }
        .btn-cyan:hover { background-color: #0891b2; color: white; }
        .col-model { width: 40%; }
        .col-temp { width: 30%; text-align: center; }
        .blend-row { background-color: rgba(16, 185, 129, 0.2) !important; font-weight: bold; color: #6ee7b7; border-top: 2px solid #10b981; }
        h4 { color: #38bdf8; font-weight: 700; letter-spacing: -0.5px; }
        hr { border-color: #334155; opacity: 1; }
        .kalshi-table { table-layout: fixed; width: 100%; }
        .kalshi-table td { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    </style>
    </head><body class="p-4">
    
    <div class="d-flex justify-content-between align-items-end mb-2">
        <h3 class="m-0">⚡ Weather Master V17 <small class="text-muted fs-6 ms-2">Weighted Consensus</small></h3>
        <div class="text-end" style="font-size: 0.85rem; color: #94a3b8;">
            <div>Run Time (EST): <span style="color:#e2e8f0; font-weight:600">""" + time_est_str + """</span></div>
            <div>Run Time (UTC): <span style="color:#e2e8f0; font-weight:600">""" + time_utc_str + """</span></div>
        </div>
    </div>
    <hr>"""
    
    for code, cfg in LOCATIONS.items():
        city_rows = [d for d in full_data if d['Airport'] == code]
        city_ml = ml_preds.get(code, ("N/A", ""))
        city_hist = history.get(code, [])
        k_all = kalshi_df[kalshi_df['Airport'] == code].copy() if not kalshi_df.empty else pd.DataFrame()
        
        wb_today, wb_tmw = calculate_weighted_blend(city_rows)
        cons = wb_today if wb_today else 0

        def make_k_table(df_sub):
            if df_sub.empty: return "<tr><td colspan='3' class='text-center text-muted'>No Active Markets</td></tr>"
            df_sub = df_sub.sort_values('Low', ascending=True)
            df_sub['Val'] = df_sub.apply(lambda r: (100-r['Price']) if r['Low'] <= cons <= r['High'] else -r['Price'], axis=1)
            rows = ""
            for _, r in df_sub.iterrows():
                c = "#4ade80" if r['Val']>5 else ("#f87171" if r['Val']<0 else "#94a3b8")
                rows += f"<tr><td>{r['Range']}</td><td>{r['Price']}¢</td><td style='color:{c};font-weight:bold'>{r['Val']:.0f}</td></tr>"
            return rows

        k_today_rows = make_k_table(k_all[k_all['Day'] == 'Today'].copy())
        k_tmw_rows = make_k_table(k_all[k_all['Day'] == 'Tomorrow'].copy())

        html += f"""<div class="row"><div class="col-12"><h4>{cfg['name']} ({code}) <span class="badge badge-ml">Live ML: {city_ml[0]}</span></h4></div>
        
        <div class="col-md-5"><div class="card"><div class="card-header">Forecast Models</div><table class="table table-dark table-sm mb-0">
        <thead><tr><th class="col-model" style="color:#fff !important">Model</th><th class="col-temp" style="color:#fff !important">Today High</th><th class="col-temp" style="color:#fff !important">Tom High</th></tr></thead><tbody>"""
        
        html += f"<tr class='blend-row'><td>WEIGHTED CONSENSUS</td><td class='text-center'>{wb_today:.1f}</td><td class='text-center'>{wb_tmw:.1f}</td></tr>"
        
        for r in city_rows: html += f"<tr><td>{r['Model']}</td><td class='text-center'>{r['Today']}</td><td class='text-center'>{r['Tomorrow']}</td></tr>"
        html += """</tbody></table></div></div>
        
        <div class="col-md-3"><div class="card"><div class="card-header">LAMP History (UTC)</div><div style="max-height:300px;overflow:auto">
        <table class="table table-dark table-sm mb-0"><thead><tr><th style="color:#fff !important">Run</th><th style="color:#fff !important">Today</th><th style="color:#fff !important">Tom</th></tr></thead><tbody>"""
        for h in city_hist: html += f"<tr><td>{h['Run']}</td><td class='text-center'>{h['Today']}</td><td class='text-center'>{h['Tmw']}</td></tr>"
        html += """</tbody></table></div></div></div>
        
        <div class="col-md-4"><div class="card">
            <div class="card-header">Kalshi <button id="{code}-btn" class="btn btn-outline-light kalshi-btn" onclick="toggleKalshi('{code}')">Show Tomorrow</button></div>
            <div style="max-height:300px;overflow:auto">
                <table class="table table-dark table-sm mb-0"><thead><tr><th style="color:#fff !important">Range</th><th style="color:#fff !important">Bid</th><th style="color:#fff !important">Value</th></tr></thead>
                <tbody id="{code}-today">{k_today_rows}</tbody>
                <tbody id="{code}-tmw" style="display:none">{k_tmw_rows}</tbody>
                </table>
            </div></div></div></div><hr class="my-4" style="border-color:#334155">""".replace("{code}", code).replace("{k_today_rows}", k_today_rows).replace("{k_tmw_rows}", k_tmw_rows)
            
    return html + "</body></html>"

if __name__ == "__main__":
    print("STARTING WEATHER MASTER V17...")
    full_data, ml_preds, history = [], {}, {}
    
    # 1. Run Data Collection
    for code, cfg in LOCATIONS.items():
        print(f"Processing {code}...")
        pred, note = get_live_ml_prediction(cfg['station_id'])
        ml_preds[code] = (pred, note)
        
        n_t, n_tm = get_nws_official(cfg['lat'], cfg['lon'], cfg['tz'])
        if n_t is not None or n_tm is not None:
            full_data.append({'Airport': code, 'Model': 'NWS Official', 'Today': n_t, 'Tomorrow': n_tm})
        
        l_t, l_tm, l_h = get_lamp_data(cfg['station_id'], cfg['tz'])
        history[code] = l_h
        if l_t: full_data.append({'Airport': code, 'Model': 'LAMP', 'Today': l_t, 'Tomorrow': l_tm})
        
        # --- GFS MOS (Using Text Parsing History) ---
        mos_list = get_mos(cfg['station_id'], cfg['tz'])
        if mos_list:
            best = mos_list[0]
            full_data.append({'Airport': code, 'Model': 'GFS MOS', 'Today': best['today'], 'Tomorrow': best['tmw']})
            # Show last 2 previous runs
            for old_run in mos_list[1:3]:
                full_data.append({'Airport': code, 'Model': f"GFS MOS (Prev {old_run['run_str']})", 'Today': old_run['today'], 'Tomorrow': old_run['tmw']})
        
        # --- GLOBAL MODELS (Using JSON History) ---
        for name, m_code in GLOBAL_MODELS.items():
            g_t, g_tm = get_global_model(cfg['lat'], cfg['lon'], cfg['tz'], m_code)
            if g_t: 
                # 1. Add Current
                full_data.append({'Airport': code, 'Model': name, 'Today': g_t, 'Tomorrow': g_tm})
                
                # 2. Update History Log
                model_hist = track_model_history(code, name, g_t, g_tm)
                
                # 3. Add Previous 2 distinct runs (skipping index 0 which is current)
                # We skip index 0 because we just added the "Current" row above.
                count = 0
                for item in model_hist[1:]:
                    if count >= 2: break
                    full_data.append({
                        'Airport': code, 
                        'Model': f"{name} (Prev {item['label']})", 
                        'Today': item['today'], 
                        'Tomorrow': item['tmw']
                    })
                    count += 1

    # 2. Get Kalshi Data
    kalshi = get_kalshi()

    # 3. Save to 'Kalshi' Folder
    output_dir = "Kalshi"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, "weather_dashboard.html")
    
    with open(output_path, "w", encoding="utf-8") as f: 
        f.write(build_html(full_data, ml_preds, kalshi, history))
    
    print(f"SUCCESS: Dashboard saved to {output_path}")
