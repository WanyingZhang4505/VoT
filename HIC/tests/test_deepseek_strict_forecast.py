import math
import unittest

from LLM.deepseek_strict_tool_provider import (
    build_forecast_tool,
    decode_forecast_arguments,
    make_strict_forecast_prompt,
)


class StrictForecastSchemaTests(unittest.TestCase):
    def test_schema_requires_one_numeric_field_per_forecast_step(self):
        tool = build_forecast_tool(192)
        function = tool["function"]
        schema = function["parameters"]
        expected = [f"step_{i:03d}" for i in range(1, 193)]
        self.assertTrue(function["strict"])
        self.assertEqual(schema["required"], expected)
        self.assertEqual(list(schema["properties"]), expected)
        self.assertFalse(schema["additionalProperties"])
        self.assertTrue(all(row == {"type": "number"}
                            for row in schema["properties"].values()))

    def test_decoder_orders_values_by_step_number(self):
        arguments = {f"step_{i:03d}": float(i) for i in range(1, 193)}
        arguments = dict(reversed(list(arguments.items())))
        response = decode_forecast_arguments(arguments, 192)
        self.assertEqual(response["prediction"], [float(i) for i in range(1, 193)])
        self.assertEqual(response["reasoning"], "")

    def test_decoder_rejects_missing_extra_boolean_and_nonfinite_values(self):
        valid = {f"step_{i:03d}": float(i) for i in range(1, 193)}
        for mutate in (
            lambda row: row.pop("step_192"),
            lambda row: row.update(extra=1.0),
            lambda row: row.update(step_001=True),
            lambda row: row.update(step_001=math.inf),
        ):
            candidate = dict(valid)
            mutate(candidate)
            with self.assertRaises(ValueError):
                decode_forecast_arguments(candidate, 192)

    def test_prompt_replaces_array_contract_but_preserves_case_input(self):
        source = (
            "HIC_PAIRED_DIRECT_HIC_FORECAST\n"
            "Return exactly 192 finite numeric\n"
            "predictions and concise reasoning in this JSON schema:\n"
            "{\"prediction\":[0],\"reasoning\":\"...\"}\n\n"
            "FORECAST-TIME INPUT:\n{\"history\":[1,2]}\n"
        )
        prompt = make_strict_forecast_prompt(source, 192)
        self.assertIn("step_001", prompt)
        self.assertIn("step_192", prompt)
        self.assertIn('{"history":[1,2]}', prompt)
        self.assertNotIn('"prediction":[0]', prompt)


if __name__ == "__main__":
    unittest.main()
