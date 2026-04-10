import json
import unittest

from app.core.llm.llm import LLM


class TestLLMMessageNormalization(unittest.TestCase):
    def setUp(self):
        self.llm = LLM(
            api_key="test",
            model="openai/glm-5.1",
            base_url="https://example.com/v1",
            task_id="test-task",
        )

    def test_validate_and_fix_tool_calls_strips_unsupported_fields(self):
        history = [
            {"role": "system", "content": "system prompt"},
            {
                "role": "assistant",
                "content": "先调用工具",
                "reasoning_content": "internal chain",
                "function_call": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "index": 0,
                        "function": {
                            "name": "execute_code",
                            "arguments": {"code": "print(1)"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "name": "execute_code",
                "content": "done",
            },
        ]

        fixed = self.llm._validate_and_fix_tool_calls(history)

        self.assertEqual(len(fixed), 3)
        self.assertEqual(fixed[1]["role"], "assistant")
        self.assertEqual(fixed[1]["content"], "先调用工具")
        self.assertNotIn("reasoning_content", fixed[1])
        self.assertNotIn("function_call", fixed[1])
        self.assertEqual(
            fixed[1]["tool_calls"],
            [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "execute_code",
                        "arguments": json.dumps(
                            {"code": "print(1)"},
                            ensure_ascii=False,
                        ),
                    },
                }
            ],
        )
        self.assertEqual(
            fixed[2],
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "done",
            },
        )

    def test_validate_and_fix_tool_calls_removes_orphan_tool_messages(self):
        history = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hello"},
            {
                "role": "tool",
                "tool_call_id": "call_missing",
                "name": "execute_code",
                "content": "orphan",
            },
        ]

        fixed = self.llm._validate_and_fix_tool_calls(history)

        self.assertEqual(
            fixed,
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "hello"},
            ],
        )

    def test_validate_and_fix_tool_calls_keeps_content_when_invalid_tool_call_is_removed(self):
        history = [
            {"role": "system", "content": "system prompt"},
            {
                "role": "assistant",
                "content": "保留这段文本",
                "tool_calls": [
                    {
                        "id": "call_bad",
                        "type": "function",
                        "index": 0,
                        "function": {
                            "arguments": "{\"code\": \"print(1)\"}",
                        },
                    }
                ],
            },
        ]

        fixed = self.llm._validate_and_fix_tool_calls(history)

        self.assertEqual(
            fixed,
            [
                {"role": "system", "content": "system prompt"},
                {"role": "assistant", "content": "保留这段文本"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
