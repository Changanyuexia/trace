import json
import sys
from typing import Dict, Any, Callable
from agent.utils import get_cached_result, cache_tool_result

class ToolRuntime:
    def __init__(self, func_map: Dict[str, Callable[..., Dict[str, Any]]]):
        self.func_map = func_map

    def handle_tool_calls(self, tool_calls):
        tool_messages = []
        for tc in tool_calls:
            name = tc.function.name
            # Parse arguments with error handling
            args_str = tc.function.arguments or "{}"
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError as e:
                print(f"[WARN] Failed to parse tool call arguments for {name}: {e}", file=sys.stderr, flush=True)
                print(f"[WARN] Arguments string (first 200 chars): {args_str[:200]}", file=sys.stderr, flush=True)
                # Try to fix common JSON issues
                # Remove any trailing incomplete JSON
                args_str_clean = args_str.strip()
                # Try to extract valid JSON by finding the last complete object
                if args_str_clean and not args_str_clean.endswith('}'):
                    # Try to find the last complete JSON object
                    last_brace = args_str_clean.rfind('}')
                    if last_brace > 0:
                        args_str_clean = args_str_clean[:last_brace + 1]
                    else:
                        # If no closing brace, try to add one
                        args_str_clean = args_str_clean.rstrip(',') + '}'
                try:
                    args = json.loads(args_str_clean)
                    print(f"[INFO] Successfully fixed JSON for {name}", file=sys.stderr, flush=True)
                except json.JSONDecodeError:
                    # If still fails, use empty dict
                    print(f"[ERROR] Could not fix JSON for {name}, using empty dict", file=sys.stderr, flush=True)
                    args = {}
            
            # Check if tool exists
            if name not in self.func_map:
                # Only expose tools that are in the current phase's schema, not all func_map tools
                # This prevents LLM from discovering tools that shouldn't be available in the current phase
                available_tools = ", ".join(sorted(self.func_map.keys()))
                error_msg = f"Tool '{name}' is not available. Available tools: {available_tools}"
                print(f"[ERROR] {error_msg}", file=sys.stderr, flush=True)
                # Don't expose all func_map tools in the error message to avoid LLM discovering tools
                # that shouldn't be available in the current phase (e.g., verify_red in patch phase)
                result = {
                    "ok": False,
                    "error": f"Tool '{name}' is not available in the current phase.",
                    # Don't include available_tools list to prevent LLM from discovering tools
                    # that aren't in the current phase's schema
                }
            else:
                # Check cache first
                cached_result = get_cached_result(name, args)
                if cached_result is not None:
                    print(f"[CACHE] Using cached result for {name}({json.dumps(args, ensure_ascii=False)[:100]}...)", file=sys.stderr, flush=True)
                    result = cached_result
                else:
                    # Execute tool call
                    result = self.func_map[name](**args)
                    # Cache the result
                    cache_tool_result(name, args, result)
            
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": json.dumps(result, ensure_ascii=False),
            })
        return tool_messages



