import streamlit as st

from terra.resource_utils import get_help_file_path


def open_and_markdown_in_streamlit(path, encoding="utf8"):
    st.markdown(open(path, encoding=encoding).read())


def open_and_markdown_in_streamlit_help(filename: str):
    path = get_help_file_path(filename)
    open_and_markdown_in_streamlit(path)
