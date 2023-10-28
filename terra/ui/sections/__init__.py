from abc import ABC, abstractmethod
import dataclasses
from contextlib import nullcontext
from typing import Any, ContextManager

import streamlit as st


class UISection(ABC):
    """Generic UI section.

    Concrete child classes should implement a run() method.
    """

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        return


@dataclasses.dataclass(kw_only=True)
class LabeledUISection(UISection):
    name: str
    icon: str | None = None

    @property
    def label(self) -> str:
        return f"{self.icon} {self.name}" if self.icon else self.name


@dataclasses.dataclass(kw_only=True)
class ContainerWrappedUISection(UISection):
    """Run child UI section in a streamlit container."""

    child: UISection
    container: ContextManager = dataclasses.field(default_factory=nullcontext)

    def run(self, *args, **kwargs) -> Any:
        with self.container:
            self.child.run(*args, **kwargs)


@dataclasses.dataclass(kw_only=True)
class LabeledContainerWrappedUISection(LabeledUISection, ContainerWrappedUISection):
    pass


@dataclasses.dataclass
class ExpanderWrappedUISection(LabeledUISection):
    """Run child UI section in a streamlit expander.

    Includes ability to conditionally run if a streamlit toggle is enabled.
    """

    child: UISection
    expanded: bool = False
    conditional: bool = False
    toggle_default: bool = False
    inform_when_toggle_disabled: bool = True
    header: bool = False
    subheader: bool = False

    def run(self, *args, **kwargs) -> Any:
        with st.expander(self.label, self.expanded):
            if self.header:
                st.header(self.label, anchor=False)
            if self.subheader:
                st.subheader(self.label, anchor=False)

            if self.conditional:
                toggle_label = f"Show {self.name}"
                spinner_label = f"Executing *{self.name}*"
                not_shown_str = f'Enable "{toggle_label}" to populate this section. Note this may increase render time.'
                if st.toggle(toggle_label, value=self.toggle_default):
                    with st.spinner(spinner_label):
                        return self.child.run(*args, **kwargs)
                else:
                    if self.inform_when_toggle_disabled:
                        st.info(not_shown_str)
            else:
                return self.child.run(*args, **kwargs)


@dataclasses.dataclass
class SequentialUISection(UISection):
    """A sequential list of UI sections."""

    sections: list[UISection]

    def run(self, *args, **kwargs) -> None:
        for section in self.sections:
            section.run(*args, **kwargs)


@dataclasses.dataclass
class SequentialLabeledUISection(SequentialUISection):
    """A sequential list of labeled UI sections."""

    sections: list[LabeledUISection]

    def asdict(self):
        return {section.name: section for section in self.sections}


@dataclasses.dataclass
class SequentialLabeledContainerWrappedUISection(SequentialLabeledUISection):
    sections: list[LabeledContainerWrappedUISection]

    def create_tabs(self):
        # Create tabs and set each tab as container in sections
        tabs = st.tabs([section.label for section in self.sections])
        for section, tab in zip(self.sections, tabs, strict=True):
            section.container = tab
