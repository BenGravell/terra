from abc import ABC, abstractmethod
import dataclasses
from typing import Any

import streamlit as st


class UISection(ABC):
    """Generic UI section.

    Concrete child classes should implement a run() method.
    """

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        return


@dataclasses.dataclass
class SequentialUISection(UISection):
    """A sequential list of UI sections."""

    sections: list[UISection]

    def run(self, *args, **kwargs) -> None:
        for section in self.sections:
            section.run(*args, **kwargs)


@dataclasses.dataclass
class ExpanderWrappedUISection(UISection):
    """Run child UI section in a streamlit expander.

    Includes ability to conditionally run if an stremalit toggle is enabled.
    """

    child: UISection
    name: str
    icon: str | None = None
    expanded: bool = False
    conditional: bool = False
    toggle_default: bool = False
    inform_when_toggle_disabled: bool = True
    header: bool = False
    subheader: bool = False

    @property
    def label(self) -> str:
        x = self.name
        if self.icon:
            x = f"{self.icon} {x}"
        return x

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
