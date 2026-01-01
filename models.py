"""
Shared data the Lua analyzer.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class Finding:
    """Represents a single issue found during analysis."""
    pattern_name: str
    severity: str  # GREEN, YELLOW, RED, DEBUG
    line_num: int
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    source_line: str = ""

    # aliases for compatibility
    @property
    def description(self) -> str:
        return self.message

    @property
    def line_content(self) -> str:
        return self.source_line
