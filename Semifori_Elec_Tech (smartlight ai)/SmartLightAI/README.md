# SmartLight AI (Semifori Elec Tech)

AI streetlight and electricity fault detection. Simulated/historical IoT telemetry goes in,
classified faults, anomaly checks, signed maintenance tickets and an operator assistant come out.

    IoT data stream -> Python preprocessing -> ML evaluation -> fault classification -> maintenance notification

## Run it

1. Put `street_light_fault_prediction_dataset.csv` in this folder.
2. `pip install -r requirements.txt`
3. `python test_all.py`          (checks every module on simulated data; never touches your data)
4. `python run_pipeline.py`      (trains, tunes, evaluates; takes a few minutes. `--fast` for a quick run)
5. `streamlit run app.py`        (dashboard)   |   `python live_monitor.py` (console demo)

`run_pipeline.py` writes everything to `outputs/`: `RESULTS.md` (all key numbers), `metrics.json`,
CSV tables and `plots/*.png`. The trained model is saved as `smartlight_model.joblib`.

## Files

| File | Purpose |
|---|---|
| `config.py` | Column names, fault names, plain-language fault info, thresholds |
| `simulator.py` | IoT sensor simulator, outages/corruption, heartbeat monitor (FR-01) |
| `data_prep.py` | Validation, cleaning, encoding, temporal features, splits, augmentation (FR-02) |
| `time_series.py` | Profiles, decomposition, EWMA anomalies, forecast, pre-failure signatures, asset risk |
| `models.py` | Candidate models, evaluation, tuning, bulb threshold, save/load, scoring (FR-04) |
| `deep_learning.py` | Neural-network classifier (MLP) and autoencoder |
| `anomaly.py` | Baseline-envelope, Isolation Forest and autoencoder detectors (FR-03) |
| `ticketing.py` | Priority rules, signed tickets, simulated portal dispatch (FR-05/06/07) |
| `chatbot.py` | NLP assistant: TF-IDF intent classifier + entity extraction |
| `plots.py` | Charts for the report |
| `run_pipeline.py` | Runs the whole ML pipeline and writes the results |
| `live_monitor.py` | End-to-end console demo of the full pipeline |
| `app.py` | Streamlit dashboard (Monitoring, Tickets, Time series, Model, Assistant) |
| `test_all.py` | One-command test suite |

## Where each assignment topic lives

| Rubric item | Code | Evidence after running |
|---|---|---|
| Machine learning approach | `models.py` (LR, Random Forest, SVM, Gradient Boosting, MLP) | `outputs/model_comparison.csv` |
| Data | `data_prep.py`, `simulator.py` | `outputs/plots/eda_*.png`, `metrics.json -> data_quality` |
| Model evaluation | `models.py` | `RESULTS.md`, `model_confusion_matrix.png`, `cross_validation.csv` |
| Time-series analysis | `time_series.py` | `outputs/plots/ts_*.png`, `ts_*.csv` |
| Solution techniques / improving accuracy | class weights, augmentation, tuning, CV, threshold, importance | `rf_tuning_results.csv`, `feature_importance.csv` |
| NLP / speech | `chatbot.py`; browser text-to-speech in `app.py` | intent accuracy in `RESULTS.md` |
| Deep learning | `deep_learning.py` | `dl_*_loss.png`, MLP row in the comparison |
| Chatbot | `chatbot.py` + Assistant tab | `test_all.py` chatbot checks |
| Practical solution | `app.py`, `live_monitor.py` | dashboard and console demo |

## Honest notes

* The dataset is synthetic. Short circuit, voltage surge and flickering have clear signatures;
  **bulb failure** only appears as lower power and overlaps heavily with healthy lights, so its recall is low
  whatever model is used. Report this as a data limitation rather than tuning further.
* About 70% of rows are "No Fault", so always guessing "No Fault" already scores about 70% accuracy.
  Use macro-F1 and per-class recall next to accuracy.
* Temperature carries almost no signal in this dataset, and the dataset has no light on/off column.
* The portal dispatch writes JSON files to `outputs/portal_outbox/`; connecting to a real municipal
  system is out of scope. Tickets are HMAC-signed; set `SMARTLIGHT_SECRET` to use your own key.
* No LSTM is included (it needs TensorFlow or PyTorch). The MLP and autoencoder cover deep learning.

## Troubleshooting

* `Dataset not found`: the CSV must sit next to `app.py`, or use `python run_pipeline.py --data path.csv`.
* `Trained model not found`: run `python run_pipeline.py` first.
* Model loading errors after upgrading scikit-learn: re-run `run_pipeline.py` to retrain.
* Move the old scripts (`train_model.py`, `save_model.py`, `automated_monitor.py`, `fault_detector.py`,
  `generate_dataset.py`, `sensor_simulator.py`, `smartlight_system.py`, `main.py`, `app_backup.py`,
  `predict_fault.py`, the experiment scripts) into a `legacy/` folder; this project replaces them.
