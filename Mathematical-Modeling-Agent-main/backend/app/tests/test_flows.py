import unittest

from app.core.flows import Flows
from app.schemas.A2A import GeneratedFigure, ModelerToCoder, QuestionModelPlan, RequiredFigure


class _FakeInterpreter:
    def get_code_output(self, section: str) -> str:
        return "图表输出摘要"


class TestFlowsRequiredFigures(unittest.TestCase):
    def test_solution_flows_include_required_figures_from_question_plan(self):
        flows = Flows({"ques1": "问题一", "ques_count": 1})
        modeler_response = ModelerToCoder(
            questions_solution={"ques1": "旧版方案"},
            question_model_docs={
                "ques1": QuestionModelPlan(
                    question_key="ques1",
                    coder_prompt="请完成问题一代码",
                    required_figures=[
                        RequiredFigure(
                            figure_id="ques1_correlation_heatmap",
                            filename="ques1_correlation_heatmap.png",
                            purpose="展示变量相关性",
                        )
                    ],
                )
            },
        )

        solution_flows = flows.get_solution_flows(
            {"ques1": "问题一", "ques_count": 1},
            modeler_response,
        )

        self.assertIn("ques1", solution_flows)
        self.assertEqual(solution_flows["ques1"]["coder_prompt"], "请完成问题一代码")
        self.assertEqual(
            solution_flows["ques1"]["required_figures"][0].filename,
            "ques1_correlation_heatmap.png",
        )

    def test_get_writer_prompt_includes_asset_contract_and_exact_filenames(self):
        flows = Flows({"ques1": "问题一", "ques_count": 1, "background": "测试背景"})

        prompt = flows.get_writer_prompt(
            key="ques1",
            coder_response="模型结果总结",
            code_interpreter=_FakeInterpreter(),
            config_template={"ques1": "模板内容"},
            question_plan=QuestionModelPlan(question_key="ques1", writer_context="写作上下文"),
            available_images=["ques1_correlation_heatmap.png", "ques1_residual_diagnostics.png"],
            generated_figures=[
                GeneratedFigure(
                    figure_id="fig1",
                    filename="ques1_correlation_heatmap.png",
                    purpose="展示变量相关性",
                    required=True,
                    generated=True,
                ),
                GeneratedFigure(
                    figure_id="fig2",
                    filename="ques1_residual_diagnostics.png",
                    purpose="展示残差诊断",
                    generated=True,
                ),
            ],
        )

        self.assertIn("【章节资产合同】", prompt)
        self.assertIn("当前章节=ques1", prompt)
        self.assertIn(
            "唯一允许引用的图片文件名=ques1_correlation_heatmap.png, ques1_residual_diagnostics.png",
            prompt,
        )
        self.assertIn("必须插入的图片文件名=ques1_correlation_heatmap.png", prompt)
        self.assertIn("文件名=ques1_correlation_heatmap.png；要求=必须插入", prompt)
        self.assertIn("文件名只认本合同，不认上下文中的描述性图名", prompt)


if __name__ == "__main__":
    unittest.main()
