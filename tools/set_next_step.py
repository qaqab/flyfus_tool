from collections.abc import Generator

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage


class SetNextStepTool(Tool):
    _STATES = {"plan", "act", "observe", "verify", "write_output"}
    _EFFORTS = {"low", "medium", "high", "xhigh"}

    def _invoke(self, tool_parameters: dict) -> Generator[ToolInvokeMessage, None, None]:
        next_state = tool_parameters.get("next_state")
        next_effort = tool_parameters.get("next_effort")
        next_objective = tool_parameters.get("next_objective")
        effort_reason = tool_parameters.get("effort_reason")

        if next_state not in self._STATES or next_effort not in self._EFFORTS:
            yield self.create_text_message("Invalid next step parameters.")
            return
        if not isinstance(next_objective, str) or not next_objective.strip():
            yield self.create_text_message("next_objective is required.")
            return
        if not isinstance(effort_reason, str) or not effort_reason.strip():
            yield self.create_text_message("effort_reason is required.")
            return

        yield self.create_json_message(
            {
                "reasoning_effort": next_effort,
                "next_state": next_state,
                "next_objective": next_objective.strip(),
                "effort_reason": effort_reason.strip(),
            }
        )
