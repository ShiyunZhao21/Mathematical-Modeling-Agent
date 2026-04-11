import unittest

from app.core.flows import Flows
from app.schemas.A2A import ModelerToCoder, QuestionModelPlan, RequiredFigure


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


if __name__ == "__main__":
    unittest.main()
