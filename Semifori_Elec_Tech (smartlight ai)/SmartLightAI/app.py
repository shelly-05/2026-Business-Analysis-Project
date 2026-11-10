"""

SmartLight AI - municipal dashboard. Run: streamlit run app.py



Tabs: Monitoring | Tickets | Time series | Model | Assistant | Municipality

Data source: a sample of the dataset, or a simulated live IoT stream.

"""



import json

from datetime import datetime

from pathlib import Path



import pandas as pd

import streamlit as st

import streamlit.components.v1 as components



import config as cfg

from anomaly import add_anomaly_columns

from chatbot import ChatState, SmartLightChatbot

from data_prep import clean_readings, load_raw, validate_and_clean

from models import load_bundle, score_readings

from simulator import HeartbeatMonitor, SensorStream

from ticketing import add_priority, build_tickets, dispatch_to_portal, tickets_to_dataframe

from municipal_portal import render_municipal_portal

from notifications import prepare_notifications_for_tickets

from time_series import asset_risk_ranking, choose_rule, hourly_profile, prepare, resample_fleet, usable



st.set_page_config(page_title="SmartLight AI", page_icon="💡", layout="wide")





# ------------------------------------------------------------------ loaders

@st.cache_data(show_spinner=False)

def load_data(path_str: str, mtime: float):

    df, report = validate_and_clean(load_raw(Path(path_str)))

    return df, report





@st.cache_resource(show_spinner=False)

def load_model(path_str: str, mtime: float):

    return load_bundle(Path(path_str))





@st.cache_data(show_spinner=False)

def load_history(path_str: str, mtime: float):

    df, _ = load_data(path_str, mtime)

    d = prepare(df)

    if not usable(d):

        return None, None, None, None

    return d, resample_fleet(d), hourly_profile(d), asset_risk_ranking(d)





def read_metrics() -> dict:

    path = cfg.OUTPUT_DIR / "metrics.json"

    return json.loads(path.read_text()) if path.exists() else {}





def speak(text: str):

    """Read text aloud with the browser's built-in speech synthesis."""

    components.html(

        "<script>const u=new SpeechSynthesisUtterance(" + json.dumps(text) + ");"

        "window.speechSynthesis.cancel();window.speechSynthesis.speak(u);</script>",

        height=0,

    )





# ------------------------------------------------------------- startup checks

if not cfg.MODEL_FILE.exists():

    st.error(f"Trained model not found: `{cfg.MODEL_FILE.name}`. Run `python run_pipeline.py` first.")

    st.stop()

if not cfg.DATA_FILE.exists():

    st.error(f"Dataset not found: `{cfg.DATA_FILE.name}`. Place it in the same folder as `app.py`.")

    st.stop()



bundle = load_model(str(cfg.MODEL_FILE), cfg.MODEL_FILE.stat().st_mtime)

data, quality = load_data(str(cfg.DATA_FILE), cfg.DATA_FILE.stat().st_mtime)

history, fleet, hourly, risk = load_history(str(cfg.DATA_FILE), cfg.DATA_FILE.stat().st_mtime)

metrics = read_metrics()





# ------------------------------------------------------------------- header

st.title("💡 SmartLight AI")

st.caption("Streetlight and electricity fault detection | Semifori Elec Tech")



h1, h2, h3, h4 = st.columns(4)

h1.metric("Model", bundle["model_name"])

h2.metric("Test accuracy", f"{bundle['test_accuracy']:.1%}")

h3.metric("Test macro-F1", f"{bundle['test_macro_f1']:.3f}")

h4.metric("Records in dataset", f"{len(data):,}")

st.caption(

    "Figures are measured on held-out test data, not on the readings shown below. "

    "Demonstration only: not connected to live municipal infrastructure."

)





# ------------------------------------------------------------------ sidebar

st.sidebar.title("Monitoring controls")

mode = st.sidebar.radio("Data source", ["Dataset sample", "Simulated live stream"])

if mode == "Dataset sample":

    sample_size = st.sidebar.slider(

        "Records to analyse", 10, min(500, len(data)), min(100, len(data)), step=10

    )

else:

    n_assets = st.sidebar.slider("Streetlights", 5, 60, 20)

    n_steps = st.sidebar.slider("Time steps (hours)", 1, 12, 6)



faults_only = st.sidebar.checkbox("Show only predicted faults", value=False)

st.sidebar.caption(

    f"Alerts below {bundle.get('confidence_threshold', cfg.CONFIDENCE_THRESHOLD):.0%} "

    "confidence are verified before a crew is dispatched."

)



if "run" not in st.session_state:

    st.session_state.run = None

if "chat" not in st.session_state:

    st.session_state.chat = []





# ------------------------------------------------------------ run monitoring

def run_monitoring():

    offline, rejected = pd.DataFrame(), pd.DataFrame()



    if mode == "Dataset sample":

        raw = data.sample(n=sample_size).reset_index(drop=True)

    else:

        stream = SensorStream(n_assets=n_assets)

        beat = HeartbeatMonitor()

        beat.register(stream.assets, stream.now)

        valid_parts, rejected_parts = [], []



        for _ in range(n_steps):

            batch = stream.next_batch()

            beat.update(batch)

            ok, bad = clean_readings(batch)

            valid_parts.append(ok)

            rejected_parts.append(bad)



        raw = pd.concat(valid_parts, ignore_index=True)

        rejected = pd.concat(rejected_parts, ignore_index=True)

        offline = beat.offline_assets(stream.now)



    results = score_readings(raw, bundle)

    if bundle.get("anomaly"):

        results = add_anomaly_columns(results, raw, bundle["anomaly"])

    results = add_priority(

        results, bundle.get("confidence_threshold", cfg.CONFIDENCE_THRESHOLD)

    )



    # Create maintenance tickets, then prepare a notification for each new ticket.

    tickets = build_tickets(results)

    notifications_created = prepare_notifications_for_tickets(tickets)



    st.session_state.run = {

        "results": results,

        "tickets": tickets,

        "offline": offline,

        "rejected": rejected,

        "mode": mode,

        "dispatched": 0,

        "notifications_created": notifications_created,

    }





if st.button("▶ Run AI monitoring"):

    try:

        run_monitoring()

    except Exception as exc:

        st.error(f"Monitoring failed: {exc}")



run = st.session_state.run

tab_mon, tab_tix, tab_ts, tab_model, tab_chat, tab_municipal,tab_notifications = st.tabs(

    ["Monitoring", "Tickets", "Time series", "Model", "Assistant", "Municipality", "Email Notifications"]

)





# ---------------------------------------------------------------- monitoring

with tab_mon:

    if run is None:

        st.info("Choose a data source in the sidebar, then press **Run AI monitoring**.")

    else:

        res = run["results"]

        faulty = int((res["label"] != cfg.NO_FAULT).sum())

        now_crew = int((res["dispatch_decision"] == "Dispatch crew now").sum())

        verify = int(res["needs_verification"].sum())



        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Readings analysed", len(res))

        c2.metric("Predicted faults", faulty)

        c3.metric("Dispatch crew now", now_crew)

        c4.metric("Verify first", verify)



        if "true_label" in res:

            match = int((res["true_label"] == res["prediction"]).sum())

            st.caption(

                f"{match} of {len(res)} predictions equal the recorded label in this sample "

                "(a quick sanity check, not an accuracy measurement)."

            )



        left, right = st.columns([3, 2])

        with left:

            st.subheader("Predicted fault distribution")

            dist = res["prediction"].value_counts().reindex(

                list(cfg.FAULT_NAMES.values()), fill_value=0

            )

            st.bar_chart(dist)

        with right:

            st.subheader("Priority summary")

            st.dataframe(

                res["dispatch_decision"].value_counts()

                .rename_axis("Decision")

                .reset_index(name="Readings"),

                hide_index=True,

            )



        shown = res[res["label"] != cfg.NO_FAULT] if faults_only else res

        st.subheader(f"Readings ({len(shown)} of {len(res)})")

        cols = [

            "asset_id", "timestamp", "voltage_v", "current_a", "power_w", "weather",

            "prediction", "confidence", "priority", "dispatch_decision",

        ]

        if "true_label" in shown:

            cols.insert(7, "true_label")

        st.dataframe(shown[cols], hide_index=True)



        if "review_anomaly" in res and int(res["review_anomaly"].sum()):

            st.warning(

                f"{int(res['review_anomaly'].sum())} reading(s) look abnormal but match no known "

                "fault. Review them remotely."

            )

        if len(run["offline"]):

            st.subheader("Communication loss")

            st.caption(

                "These nodes stopped reporting. This is treated as a connectivity problem, "

                "not a hardware fault."

            )

            st.dataframe(run["offline"], hide_index=True)

        if len(run["rejected"]):

            st.subheader("Rejected sensor readings")

            st.caption("Corrupted readings are blocked before they reach the model.")

            cols_r = ["bulb_number", "timestamp", "reject_reason"]

            st.dataframe(run["rejected"][cols_r], hide_index=True)





# ------------------------------------------------------------------- tickets

with tab_tix:

    if run is None:

        st.info("Run monitoring to generate tickets.")

    else:

        tix = run["tickets"]

        if not tix:

            st.success("No faults were predicted in this run, so no tickets were raised.")

        else:

            st.write(f"**{len(tix)} ticket(s)** generated. High priority appears first.")

            st.dataframe(

                tickets_to_dataframe(tix)[

                    [

                        "ticket_id", "asset_id", "fault_type", "confidence", "priority",

                        "dispatch_decision", "voltage_v", "current_a", "power_w",

                    ]

                ],

                hide_index=True,

            )



            notifications_created = run.get("notifications_created", 0)

            st.info(

                f"{notifications_created} new email notification(s) prepared. "

                "They are saved in the notification history but have not been sent."

            )



            with st.expander("Plain-language summaries for dispatch"):

                for t in tix[:15]:

                    st.write("• " + t["summary"])



            b1, b2 = st.columns(2)

            if b1.button("📨 Send to maintenance portal (simulated)"):

                paths = dispatch_to_portal(tix)

                st.session_state.run["dispatched"] = len(paths)



            if run["dispatched"]:

                st.success(

                    f"{run['dispatched']} ticket file(s) written to `{cfg.PORTAL_DIR}`."

                )



            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            b2.download_button(

                "📥 Download tickets (CSV)",

                tickets_to_dataframe(tix).to_csv(index=False).encode(),

                file_name=f"smartlight_tickets_{stamp}.csv",

                mime="text/csv",

            )





# --------------------------------------------------------------- time series

with tab_ts:

    if history is None:

        st.info("The dataset has no usable timestamps, so time-series views are unavailable.")

    else:

        st.subheader("Fleet fault rate over time")

        st.caption(f"Resampled with rule '{choose_rule(history)}'.")

        st.line_chart(fleet["fault_rate"] * 100)



        st.subheader("Average voltage and current by hour of day")

        st.line_chart(hourly[[cfg.COL_VOLTAGE, cfg.COL_CURRENT]])



        st.subheader("Assets most at risk (predictive maintenance)")

        st.caption("Ranked by recent fault rate and whether it is rising.")

        st.dataframe(

            risk.head(15)[

                [

                    "asset_id", "risk_level", "risk_score", "fault_rate_recent",

                    "fault_rate_earlier", "most_common_fault",

                ]

            ],

            hide_index=True,

        )



        for name in ("ts_decomposition.png", "ts_asset_anomalies.png"):

            path = cfg.PLOTS_DIR / name

            if path.exists():

                st.image(str(path))





# --------------------------------------------------------------------- model

with tab_model:

    st.subheader("Model performance (held-out test set)")

    pc = bundle["metrics"]["per_class"]

    st.dataframe(

        pd.DataFrame(pc).T.reset_index().rename(columns={"index": "Fault"}),

        hide_index=True,

    )

    st.caption(

        f"Bulb failure is the hardest class (recall {pc['Bulb Failure']['recall']:.0%}): "

        "a failing bulb mainly shows up as lower power draw, which overlaps with healthy lights."

    )



    comp_path = cfg.OUTPUT_DIR / "model_comparison.csv"

    if comp_path.exists():

        st.subheader("Models compared (validation set)")

        st.dataframe(

            pd.read_csv(comp_path).drop(columns=["deployable"], errors="ignore"),

            hide_index=True,

        )



    anom_path = cfg.OUTPUT_DIR / "anomaly_detection_results.csv"

    if anom_path.exists():

        st.subheader("Anomaly detection")

        st.dataframe(pd.read_csv(anom_path), hide_index=True)



    if metrics.get("temporal_experiment"):

        t = metrics["temporal_experiment"]

        st.write(

            f"**Time-series features experiment:** macro-F1 "

            f"{t['val_macro_f1_base']:.3f} → {t['val_macro_f1_with_temporal']:.3f}. "

            f"{t['conclusion']}."

        )



    for name in (

        "model_confusion_matrix.png", "model_feature_importance.png", "model_roc.png"

    ):

        path = cfg.PLOTS_DIR / name

        if path.exists():

            st.image(str(path))





# ----------------------------------------------------------------- assistant

with tab_chat:

    st.subheader("Ask the SmartLight assistant")

    state = ChatState(

        results=run["results"] if run else None,

        risk=risk,

        metrics=bundle["metrics"],

        model_name=bundle["model_name"],

    )



    def ask(question: str):

        reply = SmartLightChatbot(state).reply(question)

        st.session_state.chat.append(("user", question))

        st.session_state.chat.append(("assistant", reply))



    def submit():

        question = st.session_state.get("chat_box", "").strip()

        if question:

            ask(question)

        st.session_state["chat_box"] = ""



    st.text_input(

        "Type a question and press Enter",

        key="chat_box",

        on_change=submit,

        placeholder="e.g. Which streetlights are faulty?",

    )

    q1, q2, q3 = st.columns(3)

    if q1.button("How is the network doing?"):

        ask("How is the network doing?")

    if q2.button("What is most urgent?"):

        ask("What is most urgent?")

    if q3.button("Which assets are at risk?"):

        ask("Which assets are at risk?")



    for role, message in reversed(st.session_state.chat[-12:]):

        with st.chat_message(role):

            st.write(message)



    read_aloud = st.checkbox("🔊 Read the latest answer aloud")

    if (

        read_aloud

        and st.session_state.chat

        and st.session_state.chat[-1][0] == "assistant"

    ):

        speak(st.session_state.chat[-1][1])





# ------------------------------------------------------- municipal portal

with tab_municipal:

    render_municipal_portal()



with tab_notifications:
    st.title("Email Notification Centre")
    st.caption("Semifori Municipal Electricity Department")

    from notifications import load_notification_history

    history = load_notification_history()

    if not history:
        st.info("No email notifications have been prepared yet.")
    else:
        total_notifications = len(history)

        prepared_count = sum(
            1 for item in history
            if item.get("delivery_status") == "Prepared - Not Sent"
        )
        sent_count = sum(
            1 for item in history
            if item.get("delivery_status") == "Sent"
        )
        failed_count = sum(
            1 for item in history
            if item.get("delivery_status") == "Failed"
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Notifications", total_notifications)
        col2.metric("Prepared", prepared_count)
        col3.metric("Sent", sent_count)
        col4.metric("Failed", failed_count)

        st.divider()

        search_query = st.text_input(
            "Search notifications",
            placeholder="Enter a streetlight ID or ticket ID...",
            key="notification_search",
        )

        search_term = search_query.strip().casefold()
        filtered_history = [
            item for item in history
            if search_term in " ".join([
                str(item.get("ticket_id", "")),
                str(item.get("subject", "")),
                str(item.get("message", "")),
            ]).casefold()
        ]

        st.caption(f"Showing {len(filtered_history)} notification(s)")

        if not filtered_history:
            st.warning("No matching notifications found.")

        for notification in reversed(filtered_history):
            with st.expander(
                f"{notification.get('subject', 'Email Notification')} | "
                f"{notification.get('delivery_status', 'Unknown')}"
            ):
                st.write("**Recipient:**", notification.get("recipient", "Unknown"))
                st.write("**Ticket ID:**", notification.get("ticket_id", "Unknown"))
                st.write("**Date prepared:**", notification.get("created_at", "Unknown"))
                st.write("**Delivery status:**", notification.get("delivery_status", "Unknown"))
                st.divider()
                st.write("**Email message:**")
                st.text(notification.get("message", "No message available."))

st.divider()

st.caption("SmartLight AI | Academic prototype | Simulated and historical data only")
