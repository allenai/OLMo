from typing import Any, Dict, List, Optional, Tuple

from efficiency_benchmark.task import InstanceConversion


def t5_prompt_conversion(
    *,
    task_name: str,
    label_field: str = "label",
    label_map: Dict[int, str],
    use_fields: Optional[List[str]] = None,
) -> InstanceConversion:
    def convert(instance: Dict[str, Any]) -> Tuple[str, str]:
        target = label_map[instance[label_field]]
        fields = list(instance.keys()) if use_fields is None else use_fields
        if label_field in fields:
            fields.remove(label_field)

        source = [task_name]
        for field in fields:
            source.append(f"{field}:")
            source.append(instance[field])

        return " ".join(source), target

    return convert
