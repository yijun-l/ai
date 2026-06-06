# core/parser.py
import re
from typing import Tuple, Dict, Optional

def parse_tool_call(response_text: str) -> Tuple[Optional[str], Dict[str, str]]:
    """Parse LLM output and extract tool name and parameters"""
    if "<tool_call>" not in response_text:
        return None, {}

    func_match = re.search(r"<function=(.*?)>", response_text)
    if not func_match:
        return None, {}

    func_name = func_match.group(1).strip()
    param_matches = re.findall(r"<parameter=(.*?)>\s*(.*?)\s*</parameter>", response_text, re.DOTALL)
    arguments = {k.strip(): v.strip() for k, v in param_matches}

    return func_name, arguments