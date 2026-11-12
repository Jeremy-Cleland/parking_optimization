"""Traffic fallback tests that never contact a provider."""

from types import SimpleNamespace

from core.traffic_manager import TrafficManager


def test_free_mode(monkeypatch):
    config = SimpleNamespace(
        active_api_key=None,
        has_api_keys=False,
        api=SimpleNamespace(
            map_provider="tomtom", api_cache_ttl_seconds=60, max_api_calls_per_minute=30
        ),
    )
    monkeypatch.setattr("core.traffic_manager.get_config", lambda: config)
    traffic = TrafficManager()
    result = traffic.get_traffic_conditions((42.963, -85.668), (42.964, -85.669))
    assert result.is_real_data is False
    assert result.speed_kmh > 0
    assert result.travel_time_factor > 0
    assert (
        traffic.get_traffic_conditions((42.963, -85.668), (42.964, -85.669)) is result
    )


def test_provider_timeouts_fall_back_without_logging_credentials(monkeypatch, capsys):
    import requests

    from core.map_services.google_maps_service import GoogleMapsService
    from core.map_services.mapquest_service import MapQuestService
    from core.map_services.tomtom_service import TomTomService

    calls = []

    def timed_out(url, **kwargs):
        calls.append(kwargs)
        raise requests.Timeout("sensitive-token-in-request-url")

    monkeypatch.setattr(requests, "get", timed_out)
    for service_type in [GoogleMapsService, MapQuestService, TomTomService]:
        service = service_type("test-key")
        assert service.get_route([42.963, -85.668], [42.964, -85.669]) == {}
        assert service.get_traffic_incidents([42.963, -85.669, 42.964, -85.668]) == {}
    assert len(calls) == 6
    assert all(call["timeout"] == (3.05, 10) for call in calls)
    assert "sensitive-token" not in capsys.readouterr().out
