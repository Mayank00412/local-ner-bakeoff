import unittest
from src.common import clean_entities, parse_output


class ValidationTests(unittest.TestCase):
    def test_fenced_json(self):
        self.assertEqual(parse_output('Here:\n```json\n{"entities": []}\n```'), [])

    def test_ambiguous_and_invalid(self):
        for raw in ['{"entities": []} {"entities": []}', 'not json', '{"entities": {}}']:
            with self.assertRaises(ValueError):
                parse_output(raw)

    def test_validation_without_gold(self):
        entities = [{'text': "Singapore's", 'label': 'location'},
                    {'text': 'Singapore', 'label': 'LOCATION'},
                    {'text': 'Paris', 'label': 'LOCATION'},
                    {'text': 'Singapore', 'label': 'DATE'}, {}, None]
        clean, rejected = clean_entities(entities, "Singapore's harbour")
        self.assertEqual(clean, [{'text': 'Singapore', 'label': 'LOCATION'}])
        self.assertEqual(rejected, 4)

    def test_no_substring_hallucinations(self):
        self.assertEqual(clean_entities([{'text': 'York', 'label': 'LOCATION'}], 'Yorkshire')[0], [])


if __name__ == '__main__':
    unittest.main()
