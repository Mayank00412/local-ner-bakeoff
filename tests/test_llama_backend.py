import unittest
from src.common import PROMPT
from src.llama_backend import chatml


class PromptTests(unittest.TestCase):
    def test_shared_prompt_and_delimiters(self):
        text = 'Mira visited London.'
        value = chatml(text)
        self.assertTrue(value.startswith('<|im_start|>system\n' + PROMPT))
        self.assertTrue(value.endswith('<|im_start|>assistant\n'))
        self.assertEqual(value.count(PROMPT), 1)
        self.assertIn('<|im_start|>user\n' + text + '<|im_end|>', value)


if __name__ == '__main__':
    unittest.main()
