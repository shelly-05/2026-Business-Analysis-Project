"""
SmartLight assistant (NLP softbot) for municipal dispatch operators.

Pipeline: text -> TF-IDF features -> logistic-regression INTENT classifier
          + regex ENTITY extraction (asset number, fault type)
          -> answer built from the live monitoring results.
Runs fully offline; no external API.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import config as cfg

TRAINING_DATA = {
    "greeting": [
        "hi", "hello", "hey there", "good morning", "good afternoon", "hello smartlight",
        "hi assistant", "hey", "good evening"],
    "goodbye": [
        "bye", "goodbye", "see you later", "thanks", "thank you", "thanks a lot",
        "that is all", "cheers", "ok thanks"],
    "help": [
        "help", "what can you do", "how do I use this", "what can I ask you",
        "show me the commands", "what are your features", "guide me", "how does this work"],
    "fleet_status": [
        "how is the network doing", "give me an overall status", "system status",
        "summary of the streetlights", "what is the current state of the grid",
        "how are the lights doing", "overview please", "status report",
        "are the streetlights ok", "show me the summary"],
    "count_faults": [
        "how many faults are there", "number of faults detected", "count the faults",
        "how many short circuits", "how many bulb failures", "total number of faults",
        "how many lights have problems", "how many voltage surges were found",
        "how many flickering lights", "what number of bulbs have failed"],
    "list_faults": [
        "which streetlights are faulty", "list all faults", "show me the faulty lights",
        "which assets have faults", "show faulty streetlights", "give me the list of broken lights",
        "which lights are broken", "list the lights that failed", "show all the problems"],
    "asset_status": [
        "what is the status of bulb 12", "check streetlight 5", "is SL-007 working",
        "how is asset ASSET-SL-2026-003 doing", "status of light 21", "tell me about bulb 8",
        "what is wrong with streetlight 14", "check on pole 33", "is bulb 9 ok"],
    "explain_fault": [
        "what is a short circuit", "explain voltage surge", "what does bulb failure mean",
        "what causes flickering", "tell me about light flickering", "define a short circuit",
        "why do voltage surges happen", "describe bulb failure", "what is flickering"],
    "recommend_action": [
        "what should the crew do", "how do we fix a short circuit", "recommended action for bulb failure",
        "what action is needed", "how should maintenance respond", "what is the repair procedure",
        "what do we do about voltage surges", "how to repair flickering lights",
        "advice for the maintenance team"],
    "urgent": [
        "what is most urgent", "show high priority tickets", "which faults are critical",
        "what needs immediate attention", "urgent faults", "priority list",
        "which lights need a crew now", "what should we fix first"],
    "at_risk": [
        "which assets are at risk", "predict which lights will fail", "predictive maintenance report",
        "which streetlights are deteriorating", "show lights likely to fail",
        "which bulbs need attention soon", "risk ranking", "which lights are trending towards failure"],
    "model_performance": [
        "how accurate is the model", "what is the model accuracy", "show model performance",
        "how good is the AI", "what algorithm is used", "precision and recall",
        "how reliable are the predictions", "tell me about the model",
        "which machine learning method powers this"],
    "sensor_info": [
        "what does the system monitor", "which sensors are used", "what data do you collect",
        "what readings do you track", "what measurements are recorded", "which parameters are monitored"],
}

# Held-out phrasings used only to measure the intent classifier.
TEST_DATA = {
    "greeting": ["hello there", "good day", "hey smartlight"],
    "goodbye": ["thanks, that's everything", "see you", "thank you very much"],
    "help": ["what can I do here", "help me please", "what questions can I ask"],
    "fleet_status": ["how is the whole grid", "give me a status overview", "how are all the streetlights"],
    "count_faults": ["how many faults were found", "count the short circuits", "what is the number of bulb failures"],
    "list_faults": ["which lights have faults", "list the faulty streetlights", "show me every broken light"],
    "asset_status": ["is streetlight 18 working", "status of bulb 4", "check asset ASSET-SL-2026-009"],
    "explain_fault": ["what is a voltage surge", "explain light flickering", "what does short circuit mean"],
    "recommend_action": ["what should we do about bulb failures", "how do I repair a short circuit", "what is the recommended fix"],
    "urgent": ["which faults need attention right now", "show the most urgent tickets", "what is critical"],
    "at_risk": ["which streetlights might fail soon", "show the risk ranking", "predictive maintenance please"],
    "model_performance": ["how accurate are the predictions", "what is the accuracy of the AI", "which algorithm do you use"],
    "sensor_info": ["what sensors do you monitor", "which readings are collected", "what data is measured"],
}

FAULT_KEYWORDS = {
    cfg.SHORT_CIRCUIT: [r"short[\s-]?circuit", r"\bshorted\b", r"\bshorts\b"],
    cfg.VOLTAGE_SURGE: [r"surge", r"over[\s-]?voltage", r"\bspike"],
    cfg.BULB_FAILURE: [r"bulb\s+fail", r"burnt", r"burned", r"blown", r"lamp\s+fail", r"dead\s+bulb"],
    cfg.LIGHT_FLICKERING: [r"flicker", r"blinking"],
}
ASSET_PATTERNS = [
    r"asset[\s-]*sl[\s-]*\d{4}[\s-]*(\d+)",
    r"\bsl[\s-]*(\d+)",
    r"(?:bulb|streetlight|street\s*light|light|asset|pole|node|lamp)\s*(?:number|no\.?|#)?\s*(\d+)",
]


def extract_asset(text: str):
    """Asset number mentioned in the text, or None."""
    for pattern in ASSET_PATTERNS:
        m = re.search(pattern, text, flags=re.I)
        if m:
            return int(m.group(1)), m.span()
    return None, None


def extract_fault(text: str):
    """Fault label mentioned in the text, or None."""
    for label, patterns in FAULT_KEYWORDS.items():
        if any(re.search(p, text, flags=re.I) for p in patterns):
            return label
    return None


class IntentClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", LogisticRegression(C=20, max_iter=2000)),
        ])
        texts = [t for items in TRAINING_DATA.values() for t in items]
        labels = [k for k, items in TRAINING_DATA.items() for _ in items]
        self.pipeline.fit(texts, labels)

    def predict(self, text: str):
        proba = self.pipeline.predict_proba([text])[0]
        i = int(np.argmax(proba))
        return str(self.pipeline.classes_[i]), float(proba[i])

    def evaluate(self) -> dict:
        texts = [t for items in TEST_DATA.values() for t in items]
        truth = [k for k, items in TEST_DATA.items() for _ in items]
        pred = list(self.pipeline.predict(texts))
        wrong = [(t, a, p) for t, a, p in zip(texts, truth, pred) if a != p]
        return {"n_test": len(texts), "n_train": sum(map(len, TRAINING_DATA.values())),
                "accuracy": float(np.mean([a == p for a, p in zip(truth, pred)])),
                "misclassified": [{"text": t, "expected": a, "predicted": p} for t, a, p in wrong]}


@dataclass
class ChatState:
    results: pd.DataFrame | None = None       # scored + prioritised readings
    risk: pd.DataFrame | None = None          # asset risk ranking
    metrics: dict = field(default_factory=dict)
    model_name: str = "the fault model"


class SmartLightChatbot:
    FALLBACK_CONFIDENCE = 0.20

    def __init__(self, state: ChatState):
        self.state = state
        self.intents = IntentClassifier()

    # ------------------------------------------------------------ public
    def reply(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return "Please type a question, for example: 'how is the network doing?'"
        asset, span = extract_asset(text)
        remaining = text if span is None else text[:span[0]] + " " + text[span[1]:]
        fault = extract_fault(remaining)
        intent, confidence = self.intents.predict(text)

        if asset is not None and intent in ("fleet_status", "asset_status", "greeting"):
            intent = "asset_status"
        elif intent == "asset_status" and asset is None and fault is not None:
            intent = "count_faults"       # "number of bulb failures" is a count, not one asset
        elif confidence < self.FALLBACK_CONFIDENCE:
            intent = "fallback"
        return getattr(self, f"_{intent}")(text, asset, fault)

    # ----------------------------------------------------------- helpers
    def _has_data(self) -> bool:
        r = self.state.results
        return r is not None and len(r) > 0

    NO_DATA = "There is no monitoring data yet. Run monitoring first, then ask me again."

    # ----------------------------------------------------------- intents
    def _greeting(self, *_):
        return "Hello! I'm the SmartLight assistant. Ask me about faults, urgent tickets or a specific streetlight."

    def _goodbye(self, *_):
        return "You're welcome. Stay safe and keep the streets lit."

    def _help(self, *_):
        return (
            "You can ask me things like:\n"
            "- How is the network doing?\n- How many short circuits are there?\n"
            "- Which streetlights are faulty?\n- What is the status of bulb 12?\n"
            "- What is most urgent?\n- Which assets are at risk?\n"
            "- What is a voltage surge? / What should the crew do about it?\n"
            "- How accurate is the model? / What sensors do you use?"
        )

    def _fallback(self, *_):
        return "Sorry, I didn't understand that. Type 'help' to see what I can answer."

    def _fleet_status(self, *_):
        if not self._has_data():
            return self.NO_DATA
        r = self.state.results
        faulty = int((r["label"] != cfg.NO_FAULT).sum())
        lines = [f"I analysed {len(r)} readings: {len(r) - faulty} healthy and {faulty} with a predicted fault "
                 f"({faulty / len(r):.0%})."]
        by_type = r[r["label"] != cfg.NO_FAULT]["prediction"].value_counts()
        if len(by_type):
            lines.append("Faults by type: " + ", ".join(f"{k}: {v}" for k, v in by_type.items()) + ".")
        high = int(((r["priority"] == "High") & ~r["needs_verification"]).sum()) if "priority" in r else 0
        lines.append(f"{high} need immediate crew dispatch.")
        return " ".join(lines)

    def _count_faults(self, text, asset, fault):
        if not self._has_data():
            return self.NO_DATA
        r = self.state.results
        if fault is not None:
            n = int((r["label"] == fault).sum())
            return f"{n} reading(s) show {cfg.FAULT_NAMES[fault]}."
        n = int((r["label"] != cfg.NO_FAULT).sum())
        return f"{n} of {len(r)} readings have a predicted fault."

    def _list_faults(self, text, asset, fault):
        if not self._has_data():
            return self.NO_DATA
        r = self.state.results
        sel = r[r["label"] == fault] if fault is not None else r[r["label"] != cfg.NO_FAULT]
        if sel.empty:
            return "No matching faults were predicted in the current run."
        sel = sel.sort_values("confidence", ascending=False).head(10)
        rows = [f"- {x.asset_id}: {x.prediction} ({x.confidence:.0%} confidence)" for x in sel.itertuples()]
        more = "" if len(sel) < 10 else " (showing the 10 most confident)"
        return f"Faulty streetlights{more}:\n" + "\n".join(rows)

    def _asset_status(self, text, asset, fault):
        if not self._has_data():
            return self.NO_DATA
        if asset is None:
            return "Which streetlight? Give me a number, e.g. 'status of bulb 12'."
        r = self.state.results
        sel = r[r["bulb_number"] == asset]
        if sel.empty:
            return f"I have no readings for {cfg.asset_id(asset)} in the current monitoring run."
        last = sel.sort_values("timestamp").iloc[-1]
        msg = (f"{last['asset_id']}: {last['prediction']} (confidence {last['confidence']:.0%}). "
               f"Voltage {last['voltage_v']:.1f} V, current {last['current_a']:.2f} A, "
               f"power {last['power_w']:.1f} W.")
        if last["label"] != cfg.NO_FAULT:
            msg += f" Decision: {last['dispatch_decision']}. {last['recommended_action']}"
        return msg

    def _explain_fault(self, text, asset, fault):
        if fault is None:
            names = ", ".join(cfg.FAULT_NAMES[k] for k in cfg.FAULT_LABELS[1:])
            return f"Which fault? I can explain: {names}."
        info = cfg.FAULT_INFO[fault]
        return f"{cfg.FAULT_NAMES[fault]}: {info['description']} Typical sign: {info['signature']}"

    def _recommend_action(self, text, asset, fault):
        if fault is None:
            return "Short circuit and voltage surge need urgent dispatch; bulb failure and flickering can be scheduled."
        info = cfg.FAULT_INFO[fault]
        return f"{cfg.FAULT_NAMES[fault]} ({info['priority']} priority): {info['action']}"

    def _urgent(self, *_):
        if not self._has_data():
            return self.NO_DATA
        r = self.state.results
        sel = r[(r["priority"] == "High") & ~r["needs_verification"]]
        if sel.empty:
            return "Nothing needs immediate dispatch right now."
        sel = sel.sort_values("confidence", ascending=False).head(8)
        rows = [f"- {x.asset_id}: {x.prediction} ({x.confidence:.0%})" for x in sel.itertuples()]
        return f"{len(sel)} high-priority fault(s) need a crew now:\n" + "\n".join(rows)

    def _at_risk(self, *_):
        risk = self.state.risk
        if risk is None or risk.empty:
            return "I don't have a risk ranking yet (it needs timestamped history)."
        top = risk.head(5)
        rows = [f"- {x.asset_id}: risk {x.risk_score:.2f} ({x.risk_level}), mostly {x.most_common_fault}"
                for x in top.itertuples()]
        return "Assets most at risk, based on recent fault rate and trend:\n" + "\n".join(rows)

    def _model_performance(self, *_):
        m = self.state.metrics or {}
        if not m:
            return f"I use {self.state.model_name}, but no evaluation figures are loaded."
        bulb = m.get("per_class", {}).get("Bulb Failure", {})
        return (f"I use {self.state.model_name}. On unseen test data it scores {m.get('accuracy', 0):.1%} accuracy "
                f"and {m.get('macro_f1', 0):.3f} macro-F1. Bulb Failure is the hardest fault "
                f"(recall {bulb.get('recall', 0):.0%}, precision {bulb.get('precision', 0):.0%}) because a failing "
                "bulb only shows up as lower power draw.")

    def _sensor_info(self, *_):
        return ("Each streetlight reports power consumption (W), line voltage (V), current (A), "
                "temperature (°C), an environmental current channel (A) and the weather condition. "
                "Readings are cleaned, classified, checked for anomalies and turned into tickets.")
