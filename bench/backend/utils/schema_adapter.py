import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
tool_schemas_path = root / "backend" / "tool_schemas"
tool_schemas_gemini_path = root / "backend" / "tool_schemas_gemini"
tool_files_to_adapt = [
    "med_tools.json",
    "platform_tools.json",
    "source_tools.json"
]

for tool_file in tool_files_to_adapt:
    with open(tool_schemas_path / tool_file, "r") as f_in:
        tool_schemas = json.load(f_in)

    converted_tool_schemas = [tool_schema["function"] for tool_schema in tool_schemas]

    with open(str(tool_schemas_gemini_path / tool_file), "w") as f_out:
        json.dump(converted_tool_schemas, f_out, ensure_ascii=False, indent=2)