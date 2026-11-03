"""The shipped observability config: it parses, it is cached, and it ships tracing OFF.

The default-off assertions deliberately read the REAL configs/observability.yaml rather
than a fixture, so a careless edit flipping the shipped default to "otlp" fails this
suite instead of quietly turning on outbound telemetry from a clinical service.
"""
from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from src.observability.tracing import configure_tracing
from src.utils.config import (
    _OBSERVABILITY_YAML,
    ObservabilityConfig,
    get_observability_config,
)


# --- the shipped file ------------------------------------------------------------------

def test_shipped_config_parses():
    config = get_observability_config()
    assert isinstance(config, ObservabilityConfig)
    assert config.service_name == "skinsure"


def test_shipped_exporter_default_is_none():
    """The repo must ship with tracing off. Asserted against the real file on disk."""
    assert get_observability_config().exporter == "none"


def test_shipped_yaml_literally_says_none():
    """Belt and braces: the YAML value itself, not just the parsed model's default.

    Without this, deleting the `exporter:` key from the file would still pass the test
    above via the model default, hiding the fact that the config no longer states it.
    """
    raw = yaml.safe_load(_OBSERVABILITY_YAML.read_text())
    assert raw["exporter"] == "none"
    assert raw["service_name"] == "skinsure"


def test_shipped_config_names_no_endpoint_or_credential():
    """Endpoint and headers come from OTEL_EXPORTER_OTLP_* env vars only.

    A key here would mean a backend URL or credential had been committed; the only
    http(s) strings in the file are Langfuse's documented endpoints inside comments.
    """
    raw = yaml.safe_load(_OBSERVABILITY_YAML.read_text())
    assert set(raw) == {"exporter", "service_name"}, (
        "configs/observability.yaml gained a key. If it is an endpoint, a header or a "
        "token, it does not belong in the repo - use OTEL_EXPORTER_OTLP_ENDPOINT / "
        "OTEL_EXPORTER_OTLP_HEADERS. If it is genuinely non-secret, add it to "
        "ObservabilityConfig and to this assertion."
    )


# --- validation ------------------------------------------------------------------------

def test_invalid_exporter_is_rejected():
    with pytest.raises(ValidationError) as excinfo:
        ObservabilityConfig(exporter="langfuse")
    message = str(excinfo.value)
    assert "exporter" in message
    # The error must name the allowed values, or the operator has to read source.
    assert "none" in message and "console" in message and "otlp" in message


@pytest.mark.parametrize("exporter", ["none", "console", "otlp"])
def test_the_three_supported_exporters_are_accepted(exporter):
    assert ObservabilityConfig(exporter=exporter).exporter == exporter


def test_defaults_apply_to_an_empty_config():
    config = ObservabilityConfig()
    assert config.exporter == "none"
    assert config.service_name == "skinsure"


# --- caching ---------------------------------------------------------------------------

def test_loader_is_cached():
    """lru_cache(maxsize=1), matching get_settings / get_model_config: same object."""
    assert get_observability_config() is get_observability_config()


# --- the config actually feeds configure_tracing ---------------------------------------

def test_shipped_config_configures_no_tracing():
    """End to end on the shipped default: tracing is OFF, so no provider is built."""
    assert configure_tracing(get_observability_config().exporter) is None
