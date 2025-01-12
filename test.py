import streamlit as st


if not st.session_state.get("init"):
    qp = st.experimental_get_query_params()

    x = qp.get("x")
    if x is not None:
        x = float(x[0])
    else:
        x = 0.5

    st.session_state.x = x

    st.session_state.init = True


reset = st.button("Reset")
if reset:
    st.session_state.x = 0.5


st.slider("x", min_value=0.0, max_value=1.0, key="x")


st.write(st.session_state.x)

qp = {}
qp["x"] = st.session_state.x

st.experimental_set_query_params(**qp)
