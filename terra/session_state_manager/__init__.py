import dataclasses

import streamlit as st

import terra.app_options as ao
from terra.culture_fit import dimensions_info


# TODO move to a config file
# harmonize with processing.FieldListBank
APP_OPTIONS_CODES = ["cf", "ql", "hp", "sp", "hf"]

# TODO use a SafeSessionState(SessionState()) if ctx is None from streamlit to enable modification for unit testing
_STATE = st.session_state


def set_(key, value):
    """Set a session state key to a given value."""
    setattr(_STATE, key, value)


def get_(key):
    """Get a session state value from a given key."""
    return getattr(_STATE, key, None)


def get_app_options_from_query_params():
    """Get app options from URL query parameters."""
    query_params = st.experimental_get_query_params()
    app_options = ao.AppOptions()
    for field in dataclasses.fields(app_options):
        if field.name in query_params:
            query_param_val = query_params[field.name]

            # This takes care of extracting singleton lists
            # (required due to experimental_get_query_params implementation)
            if len(query_param_val) == 1:
                query_param_val = query_param_val[0]

            # This takes care of converting from string to the proper data type for the field
            query_param_val = field.type(query_param_val)

            # Finally, overwrite the field in app_options with the query_param_val
            setattr(app_options, field.name, query_param_val)

    return app_options


def set_query_params_from_app_options(app_options: ao.AppOptions):
    """Set the query params with all the app_options."""
    st.experimental_set_query_params(**dataclasses.asdict(app_options))


def set_app_options(app_options: ao.AppOptions):
    """Set session state fields related to app options."""

    # First set the single collection object app_options
    set_("app_options", app_options)

    # Next, set all individual fields that are used by widgets.
    # It is critical to do this before widgets are instantiated for the first time.
    # See https://discuss.streamlit.io/t/why-do-default-values-cause-a-session-state-warning/15485/27

    # For now, these setter ops need to be manually kept in sync with the widgets & keys defined in
    # terra\ui\sections\options.py

    for dimension in dimensions_info.DIMENSIONS:
        set_(dimension, getattr(app_options, f"culture_fit_preference_{dimension}"))

    # TODO move list to config
    for code in APP_OPTIONS_CODES:
        weight_field = f"{code}_score_weight"
        min_field = f"{code}_score_min"
        set_(weight_field, getattr(app_options, weight_field))
        set_(min_field, getattr(app_options, min_field))

    set_("english_ratio_min", getattr(app_options, "english_ratio_min"))

    # Special handling for (min, max) range params
    set_(
        "average_temperature_celsius_range",
        (
            getattr(app_options, "average_temperature_celsius_min"),
            getattr(app_options, "average_temperature_celsius_max"),
        ),
    )
    set_(
        "average_sunshine_hours_per_day_range",
        (
            getattr(app_options, "average_sunshine_hours_per_day_min"),
            getattr(app_options, "average_sunshine_hours_per_day_max"),
        ),
    )
    set_("continents", getattr(app_options, "continents"))


def initialize_session():
    """Perform initialization operations for each session."""

    # Only pull the query_params on the first run e.g. to support deeplinking.
    # Otherwise, when this function is not called, only use the options that have been set in the session.
    # This helps avoid a race condition between getting options via query_params and getting options via the UI.
    set_app_options(get_app_options_from_query_params())
    set_("initialized", True)


def initialize_run():
    """Perform initialization operations for each app run.

    Should be called at the start of each app run.

    Uses the 'initialized' key in session state to determine if the session is fresh or not.
    """

    if get_("initialized"):
        return

    initialize_session()


def finalize_run():
    """Perform finalization operations for each app run.

    Should be called at the end of each app run.
    """
    set_query_params_from_app_options(get_("app_options"))
