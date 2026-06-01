"""Alerting fan-out for drift reports.

Two sinks, both optional + env-configured so the pipeline runs in CI/locally
with zero infra (it just logs):
  SLACK_WEBHOOK_URL       -> formatted Slack message on ALERT
  PROMETHEUS_PUSHGATEWAY  -> pushes gauges Alertmanager can page on

Drift detection is worthless without a path to a human. This is that path.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request

from quant_risk.monitoring.drift import PSI_ALERT, DriftReport

log = logging.getLogger("quant_risk.alerts")


def _post_json(url, payload, timeout=5):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status


def _slack_message(report: DriftReport) -> dict:
    feats = ", ".join(report.alerting_features) or "none"
    lines = [":rotating_light: *Data drift detected -- credit_pd*",
             f"> reference n={report.n_reference:,}  current n={report.n_current:,}",
             f"> features over PSI {PSI_ALERT}: *{feats}*"]
    if report.prediction_psi is not None:
        lines.append(f"> prediction PSI: *{report.prediction_psi}*")
    if report.target_rate_current is not None:
        lines.append(f"> default rate {report.target_rate_reference} -> {report.target_rate_current}")
    return {"text": "\n".join(lines)}


def _push_prometheus(gateway, report, job="drift_monitor"):
    lines = [f'drift_detected{{model="credit_pd"}} {int(report.drift_detected)}']
    for f in report.features:
        lines.append(f'feature_psi{{model="credit_pd",feature="{f.feature}"}} {f.psi}')
    if report.prediction_psi is not None:
        lines.append(f'prediction_psi{{model="credit_pd"}} {report.prediction_psi}')
    body = ("\n".join(lines) + "\n").encode()
    req = urllib.request.Request(f"{gateway.rstrip('/')}/metrics/job/{job}", data=body, method="POST")
    urllib.request.urlopen(req, timeout=5)


def dispatch(report: DriftReport) -> dict:
    """Send the report to all configured sinks. Alerting must never crash the job."""
    sent = {"slack": False, "prometheus": False, "drift_detected": report.drift_detected}
    if not report.drift_detected:
        log.info("No material drift. Nothing dispatched.")
        return sent
    log.warning("DRIFT ALERT: %s", report.alerting_features)
    if (gw := os.getenv("PROMETHEUS_PUSHGATEWAY")):
        try:
            _push_prometheus(gw, report); sent["prometheus"] = True
        except Exception as e:  # noqa: BLE001
            log.error("Prometheus push failed: %s", e)
    if (url := os.getenv("SLACK_WEBHOOK_URL")):
        try:
            _post_json(url, _slack_message(report)); sent["slack"] = True
        except Exception as e:  # noqa: BLE001
            log.error("Slack post failed: %s", e)
    else:
        log.warning("SLACK_WEBHOOK_URL unset -- would have sent:\n%s", _slack_message(report)["text"])
    return sent