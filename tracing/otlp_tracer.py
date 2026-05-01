import os
from functools import wraps
from threading import Lock
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_initialized = False
_provider: Optional[TracerProvider] = None
_init_lock = Lock()


def init_tracer(service_name: Optional[str] = None) -> None:
    """Initialize OTLP/HTTP exporter so created spans are sent to Tempo."""
    global _initialized, _provider

    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        resolved_service_name = service_name or os.getenv("OTEL_SERVICE_NAME") or "python-service"
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or "http://localhost:4318/v1/traces"

        resource = Resource.create({"service.name": resolved_service_name})
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        provider.add_span_processor(processor)

        trace.set_tracer_provider(provider)
        _provider = provider
        _initialized = True


def instrument_requests() -> None:
    """Compatibility no-op for existing call sites."""


def instrument_flask_app(flask_app) -> None:
    """Compatibility no-op for existing call sites."""
    _ = flask_app


def instrument_fastapi_app(app) -> None:
    """Compatibility no-op for existing call sites."""
    _ = app


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)


def shutdown() -> None:
    if _provider is None:
        return
    _provider.shutdown()


def trace_decorator(span_name=None):
    """Create a span around function execution."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer(func.__module__)
            name = span_name or func.__name__
            with tracer.start_as_current_span(name):
                return func(*args, **kwargs)

        return wrapper

    return decorator
