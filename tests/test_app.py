from streamlit.testing.v1 import AppTest

from hotel_booking.prepare import ROOT


def test_filters_reset_and_empty_selection():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.metric[0].value == "119,390"
    app.selectbox(key="hotel_filter").select("City Hotel").run()
    assert not app.exception
    assert app.metric[0].value == "79,330"
    app.selectbox(key="hotel_filter").select("All hotels").run()
    assert app.metric[0].value == "119,390"
    app.selectbox(key="period_filter").select("2016 (complete year)").run()
    assert app.metric[0].value == "56,707"
    app.selectbox(key="hotel_filter").select("Resort Hotel").run()
    app.multiselect(key="segment_filter").select("Aviation").run()
    assert not app.exception
    assert any("No bookings match" in item.value for item in app.info)
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[0].value == "119,390"


def test_model_workload_scenario():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    app.slider[0].set_value(0.0).run()
    values = {item.label: item.value for item in app.metric}
    assert values["Bookings to review"] == values["Test bookings"]
    assert values["Cancellations missed"] == "0"
    app.slider[0].set_value(20.0).run()
    assert not app.exception
    values = {item.label: item.value for item in app.metric}
    assert int(values["Bookings to review"].replace(",", "")) < 22639
