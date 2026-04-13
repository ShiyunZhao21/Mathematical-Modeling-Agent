import unittest

from app.core.agents.agent import Agent


class _FakeLLM:
    pass


class TestAgentStructuredCompression(unittest.IsolatedAsyncioTestCase):
    async def test_clear_memory_preserves_exact_image_filenames_in_structured_memory(self):
        agent = Agent(task_id="test", model=_FakeLLM(), max_memory=4)
        agent.chat_history = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "当前章节 ques1"},
            {
                "role": "assistant",
                "content": "【章节资产合同】\n当前章节=ques1\n唯一允许引用的图片文件名=ques1_correlation_heatmap.png, ques1_residual_diagnostics.png\n必须插入的图片文件名=ques1_correlation_heatmap.png",
            },
            {
                "role": "user",
                "content": "请只使用 ![图](ques1_correlation_heatmap.png) 和 ![图](ques1_residual_diagnostics.png)",
            },
            {"role": "assistant", "content": "好的，保持真实文件名不变"},
            {"role": "user", "content": "继续完成问题一"},
        ]

        await agent.clear_memory()

        self.assertEqual(agent.chat_history[0]["role"], "system")
        self.assertEqual(agent.chat_history[1]["role"], "assistant")
        self.assertIn("[结构化记忆压缩]", agent.chat_history[1]["content"])
        self.assertIn("ques1_correlation_heatmap.png", agent.chat_history[1]["content"])
        self.assertIn("ques1_residual_diagnostics.png", agent.chat_history[1]["content"])
        self.assertIn(
            "ques1_correlation_heatmap.png",
            agent.compressed_memory.allowed_images,
        )
        self.assertIn(
            "ques1_residual_diagnostics.png",
            agent.compressed_memory.allowed_images,
        )
        self.assertIn(
            "ques1_correlation_heatmap.png",
            agent.compressed_memory.required_images,
        )

    async def test_clear_memory_merges_multiple_rounds_without_losing_locked_identifiers(self):
        agent = Agent(task_id="test", model=_FakeLLM(), max_memory=4)
        agent.chat_history = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "当前章节 ques1"},
            {
                "role": "assistant",
                "content": "唯一允许引用的图片文件名=ques1_model_fit.png\n必须插入的图片文件名=ques1_model_fit.png",
            },
            {"role": "assistant", "content": "生成图片 ques1_model_fit.png"},
            {"role": "user", "content": "继续"},
            {"role": "assistant", "content": "继续处理"},
        ]
        await agent.clear_memory()

        agent.chat_history.extend(
            [
                {"role": "user", "content": "当前章节 ques2"},
                {
                    "role": "assistant",
                    "content": "唯一允许引用的图片文件名=ques2_prediction_curve.png\n必须插入的图片文件名=ques2_prediction_curve.png",
                },
                {"role": "assistant", "content": "生成图片 ques2_prediction_curve.png"},
                {"role": "user", "content": "继续"},
                {"role": "assistant", "content": "继续处理"},
                {"role": "user", "content": "准备收尾"},
            ]
        )
        await agent.clear_memory()

        self.assertIn("ques1_model_fit.png", agent.compressed_memory.locked_identifiers)
        self.assertIn("ques2_prediction_curve.png", agent.compressed_memory.locked_identifiers)
        self.assertIn("ques1_model_fit.png", agent.compressed_memory.generated_images)
        self.assertIn("ques2_prediction_curve.png", agent.compressed_memory.generated_images)
        self.assertIn("ques1", agent.compressed_memory.section_states)
        self.assertIn("ques2", agent.compressed_memory.section_states)

    async def test_clear_memory_filters_generic_and_non_project_image_identifiers(self):
        agent = Agent(task_id="test", model=_FakeLLM(), max_memory=4)
        agent.chat_history = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "当前章节 ques4"},
            {
                "role": "assistant",
                "content": "【章节资产合同】\n当前章节=ques4\n唯一允许引用的图片文件名=ques4_cluster_profiles.png, ques4_feature_importance.png\n必须插入的图片文件名=ques4_cluster_profiles.png, ques4_feature_importance.png",
            },
            {
                "role": "tool",
                "content": "[IMAGE_SAVED] fig1.png\n[IMAGE_SAVED] image.png\n[IMAGE_SAVED] ques4_cluster_profiles.png\n/site-packages/sklearn/utils/validation.py\n[IMAGE_MANIFEST] ques4_cluster_profiles.png, ques4_feature_importance.png, plot.png",
            },
            {"role": "assistant", "content": "继续完成问题四"},
            {"role": "user", "content": "收尾"},
            {"role": "assistant", "content": "结束"},
        ]

        await agent.clear_memory()

        self.assertIn("ques4_cluster_profiles.png", agent.compressed_memory.allowed_images)
        self.assertIn("ques4_feature_importance.png", agent.compressed_memory.generated_images)
        self.assertNotIn("fig1.png", agent.compressed_memory.locked_identifiers)
        self.assertNotIn("image.png", agent.compressed_memory.locked_identifiers)
        self.assertNotIn("plot.png", agent.compressed_memory.locked_identifiers)
        self.assertNotIn(
            "validation.py",
            agent.compressed_memory.locked_identifiers,
        )


if __name__ == "__main__":
    unittest.main()
