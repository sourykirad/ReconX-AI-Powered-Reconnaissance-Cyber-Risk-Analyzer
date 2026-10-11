from urllib.parse import urlparse
import socket

import requests

from ml_model import predict_risk, train_or_load_model


def normalize_target(raw):
    raw = raw.strip()
    if not raw:
        raise ValueError("Enter a hostname or URL.")
    if "://" not in raw:
        raw = "https://" + raw
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Enter a valid http:// or https:// URL.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing embedded credentials are not accepted.")
    return raw, parsed


def _collect_dns(host):
    try:
        return sorted({item[4][0] for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)})
    except (socket.gaierror, OSError):
        return []


def _collect_response(url):
    try:
        return requests.get(
            url,
            timeout=6,
            allow_redirects=True,
            headers={"User-Agent": "ReconX-Educational-Audit/1.0"},
        )
    except requests.RequestException:
        return None


def analyze_target(raw):
    url, parsed = normalize_target(raw)
    host = parsed.hostname
    findings, recommendations = [], []
    ips = _collect_dns(host)
    dns_valid = bool(ips)

    if not dns_valid:
        findings.append("DNS resolution was unsuccessful.")
        recommendations.append("Verify the hostname and review DNS configuration.")

    response = _collect_response(url)
    if response is None and "://" not in raw:
        response = _collect_response("http://" + raw)

    https = bool(response and response.url.lower().startswith("https://"))
    status = response.status_code if response is not None else None
    headers = {k.lower(): v for k, v in (response.headers.items() if response is not None else [])}

    missing_headers = []
    for key in [
        "content-security-policy",
        "x-content-type-options",
        "referrer-policy",
        "strict-transport-security",
    ]:
        if key not in headers:
            missing_headers.append(key)

    certificate_invalid = 0
    hsts_missing = int("strict-transport-security" not in headers)
    server_header_present = int("server" in headers)
    missing_count = len(missing_headers)
    dns_invalid = int(not dns_valid)

    score = 0

    if not https:
        score += 30
        findings.append("HTTPS was not confirmed for the final response.")
        recommendations.append("Configure HTTPS with a valid certificate and redirect HTTP to HTTPS.")
    else:
        findings.append("HTTPS is enabled for the final response.")

    if response is None:
        score += 20
        findings.append("The web server did not return a response within the timeout.")
        recommendations.append("Confirm the service is available and review server/network logs.")
    else:
        if missing_headers:
            findings.append("Missing common security headers: " + ", ".join(missing_headers) + ".")
            recommendations.append("Review and configure appropriate security headers: " + ", ".join(missing_headers) + ".")
        else:
            findings.append("The checked security headers are present.")

        if status is not None and status >= 500:
            score += 25
            findings.append(f"Server returned HTTP {status}.")
            recommendations.append("Review application/server logs and resolve server-side errors.")
        elif status is not None and status >= 400:
            score += 10
            findings.append(f"Server returned HTTP {status}; this may be expected for the requested path.")
            recommendations.append("Verify that the requested URL and access rules are intentional.")

        if "server" in headers:
            findings.append("Server header is visible.")
        if "strict-transport-security" not in headers:
            findings.append("HSTS is not present.")

    # Missing headers alone are not treated as automatic critical findings.
    score += min(20, missing_count * 5)
    score = min(100, score)

    model, _ = train_or_load_model()
    features = [[
        int(not https),
        min(4, missing_count),
        int(status is None or (status is not None and status >= 500)),
        min(5, len(ips)),
        certificate_invalid,
        dns_invalid,
        hsts_missing,
        server_header_present,
    ]]
    risk = predict_risk(model, features[0])

    if not recommendations:
        recommendations.append("Keep software updated, monitor logs, and periodically review configuration.")

    return {
        "target": host,
        "ips": ips,
        "https": https,
        "status": status,
        "score": score,
        "risk": risk,
        "findings": findings,
        "recommendations": list(dict.fromkeys(recommendations)),
    }
