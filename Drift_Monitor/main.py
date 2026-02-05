from fastapi import FastAPI
from datetime import datetime  # <--- FIXED: Added missing import
import pandas as pd
from drift_engine import DriftEngine
from data_simulator import get_reference_data, get_drifted_data

app = FastAPI(title="Sentinel PPE Drift Monitor")

# --- GLOBAL STATE (Memory) ---
SYSTEM_STATE = {
    "is_drifting": False, 
    "drift_severity": "low",
    "persistent_drift_counter": 0,
    "last_risk_level": "Low"  # <--- FIXED: Added for logging logic
}

# --- BLACK BOX AUDIT LOG (The Fix for Yellow Line) ---
# Stores critical events for post-incident analysis
BLACK_BOX_LOGS = []

def log_event(level, action, score, cause):
    """Records critical safety events to the Black Box."""
    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "severity": level,
        "action_taken": action,
        "drift_score": score,
        "root_cause": cause
    }
    # Keep log size manageable (last 50 events)
    BLACK_BOX_LOGS.insert(0, entry)
    if len(BLACK_BOX_LOGS) > 50:
        BLACK_BOX_LOGS.pop()

try:
    reference_data = get_reference_data()
    engine = DriftEngine(reference_data)
    print("✅ PPE System Initialized")
except Exception as e:
    print(f"❌ Init Failed: {e}")

# --- PPE OPERATOR PLAYBOOK ---
PLAYBOOK = {
    "Helmet_Conf": "⚠️ ACTION: Check Camera Height/Angle. Workers may be wearing non-standard helmets.",
    "Harness_Conf": "⚠️ ACTION: Critical Safety Failure. Inspect Camera Lens for dust/fog immediately.",
    "Vest_Conf": "⚠️ ACTION: Lighting issue likely. Toggle IR mode or clean lens.",
    "default": "⚠️ ACTION: Manual Safety Audit Required."
}

# --- ENDPOINT 1: SYSTEM STATUS ---
@app.get("/status")
def get_system_status():
    if SYSTEM_STATE["is_drifting"]:
        current_data = get_drifted_data(severity=SYSTEM_STATE["drift_severity"])
        SYSTEM_STATE["persistent_drift_counter"] += 1
    else:
        current_data = get_reference_data(n=100)
        SYSTEM_STATE["persistent_drift_counter"] = 0

    # Run Engine
    report, score, budget = engine.check_data_drift(current_data)
    entropy_stat = engine.check_confidence_entropy(current_data)
    
    # --- TIERED SAFE MODES ---
    risk_level = "Low"
    action = "✅ Access Granted: Monitoring Active"
    
    if score > 80 or budget <= 0:
        risk_level = "CRITICAL"
        action = "⛔ LOCKDOWN: Turnstiles Locked. Manual Safety Check Required."
    elif score > 60:
        risk_level = "High"
        action = "✋ ALERT: Supervisor Haptic Notification Sent. Pause Work."
    elif score > 30:
        risk_level = "Medium"
        action = "⚠️ WARNING: Audio Broadcast 'Please Ensure PPE is Visible'."

    # --- BLACK BOX LOGGING LOGIC ---
    # Log if risk is high OR if state just changed (e.g. Low -> Medium)
    if risk_level in ["High", "CRITICAL"] or risk_level != SYSTEM_STATE["last_risk_level"]:
        # Find top cause for the log
        exp = engine.check_feature_importance(current_data)
        log_event(risk_level, action, score, exp["top_feature"])
    
    SYSTEM_STATE["last_risk_level"] = risk_level

    return {
        "status": "Active",
        "risk_level": risk_level,
        "action_required": action,
        "global_drift_score": score,
        "risk_budget": budget,
        "model_confidence": entropy_stat["status"],
        "simulation_mode": SYSTEM_STATE["is_drifting"]
    }

# --- ENDPOINT 2: INJECT DRIFT ---
@app.post("/inject-drift")
def inject_drift_scenario(enable: bool = True, severity: str = "high"):
    SYSTEM_STATE["is_drifting"] = enable
    SYSTEM_STATE["drift_severity"] = severity
    # Log the manual trigger
    log_event("INFO", "Drift Simulation Toggled", 0, f"User set to {enable}")
    return {"message": f"Drift {enable} (Severity: {severity})"}

# --- ENDPOINT 3: DRIFT REPORT ---
@app.get("/drift-report")
def get_drift_report():
    if SYSTEM_STATE["is_drifting"]:
        current_data = get_drifted_data(severity=SYSTEM_STATE["drift_severity"])
    else:
        current_data = get_reference_data(n=100)
    report, score, budget = engine.check_data_drift(current_data)
    formatted_report = []
    for feature, stats in report.items():
        formatted_report.append({"feature": feature, "drift_detected": stats["drift_detected"], "p_value": stats["p_value"], "severity": stats["severity"]})
    return {"drift_signature": engine.get_drift_fingerprint(report), "feature_details": formatted_report}

# --- ENDPOINT 4: FORECAST ---
@app.get("/forecast")
def get_retraining_status():
    ready_to_retrain = False
    msg = "System Healthy."
    if SYSTEM_STATE["persistent_drift_counter"] > 5:
        ready_to_retrain = True
        msg = "✅ GATE OPEN: Persistent Confidence Drop. Retraining on new helmet data."
    elif SYSTEM_STATE["persistent_drift_counter"] > 0:
        msg = "⏳ GATE CLOSED: Transient Lighting Shift. Waiting..."
    return {"retraining_gate": "OPEN" if ready_to_retrain else "CLOSED", "message": msg, "persistence_counter": SYSTEM_STATE["persistent_drift_counter"]}

# --- ENDPOINT 5: EXPLAINABILITY ---
@app.get("/explainability")
def get_shap_explanation():
    if SYSTEM_STATE["is_drifting"]:
        current_data = get_drifted_data(severity=SYSTEM_STATE["drift_severity"])
    else:
        current_data = get_reference_data(n=100)
    try:
        exp = engine.check_feature_importance(current_data)
        top_cause = exp["top_feature"]
        operator_msg = PLAYBOOK.get(top_cause, PLAYBOOK["default"])
        return {"status": "Success", "top_driving_feature": top_cause, "operator_message": operator_msg, "attribution_timeline": exp["history"], "all_feature_scores": exp["scores"]}
    except Exception as e:
        return {"status": "Error", "detail": str(e)}

# --- ENDPOINT 6: CALIBRATE ---
@app.post("/calibrate")
def calibrate_baseline():
    if SYSTEM_STATE["is_drifting"]:
        current_data = get_drifted_data(severity=SYSTEM_STATE["drift_severity"])
    else:
        current_data = get_reference_data(n=100)
    
    engine.update_baseline(current_data)
    SYSTEM_STATE["persistent_drift_counter"] = 0
    
    # Log the calibration event
    log_event("INFO", "Manual Calibration Triggered", 0, "Supervisor Action")
    
    return {"message": "✅ Baseline Calibrated. New Normal Established.", "new_risk_budget": 100.0}

# --- ENDPOINT 7: BLACK BOX LOGS (This should work now) ---
@app.get("/logs")
def get_black_box_logs():
    """Returns the history of safety events."""
    return {"logs": BLACK_BOX_LOGS}