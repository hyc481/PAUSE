import json
from pathlib import Path

from bench.utils.paths import TOOL_SCHEMAS_DIR, TOOL_SCHEMAS_GEMINI_DIR

tool_schemas_path = TOOL_SCHEMAS_DIR
tool_schemas_gemini_path = TOOL_SCHEMAS_GEMINI_DIR
tool_files_to_adapt = [
    "med_tools.json",
    "platform_tools.json",
    "source_tools.json",
]

tool_schemas_gemini_path.mkdir(parents=True, exist_ok=True)

for tool_file in tool_files_to_adapt:
    with open(tool_schemas_path / tool_file, "r") as f_in:
        tool_schemas = json.load(f_in)

    converted_tool_schemas = [tool_schema["function"] for tool_schema in tool_schemas]

    with open(str(tool_schemas_gemini_path / tool_file), "w") as f_out:
        json.dump(converted_tool_schemas, f_out, ensure_ascii=False, indent=2)