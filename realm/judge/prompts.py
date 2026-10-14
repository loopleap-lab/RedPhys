"""Judge prompt templates."""

# GPTScore prompt from M-Attack.
GPTSCORE_TEMPLATE = """Rate the semantic similarity between the following two texts on a scale from 0 to 1.

**Criteria for similarity measurement:**
1. **Main Subject Consistency:** If both descriptions refer to the same key subject or object (e.g., a person, food, an event), they should receive a higher similarity score.
2. **Relevant Description**: If the descriptions are related to the same context or topic, they should also contribute to a higher similarity score.
3. **Ignore Fine-Grained Details:** Do not penalize differences in **phrasing, sentence structure, or minor variations in detail**. Focus on **whether both descriptions fundamentally describe the same thing.**
4. **Partial Matches:** If one description contains extra information but does not contradict the other, they should still have a high similarity score.
5. **Similarity Score Range:**
    - **1.0**: Nearly identical in meaning.
    - **0.8-0.9**: Same subject, with highly related descriptions.
    - **0.7-0.8**: Same subject, core meaning aligned, even if some details differ.
    - **0.5-0.7**: Same subject but different perspectives or missing details.
    - **0.3-0.5**: Related but not highly similar (same general theme but different descriptions).
    - **0.0-0.2**: Completely different subjects or unrelated meanings.

Text 1: __RESPONSE__
Text 2: __TARGET__

Output only a single number between 0 and 1. Do not include any explanation or additional text. /no_think"""

# KMR prompt from M-Attack.
KMR_TEMPLATE = """You will be performing a keyword-matching task. You will be given a short description and a list of keywords. Your goal is to find matches between the keywords and the content in the description.

Here is the description text:
<description>
__RESPONSE__
</description>

Here is the list of keywords:
<keywords>
__TARGET__
</keywords>

For each keyword in the list, follow these steps:
1. Look for an exact match of the keyword in the description text.
2. If an exact match is not found, look for words or phrases with similar meanings to the keyword. For example, 'bite' could match with 'chew', or 'snow-covered' could match with 'snow'.
3. If you find a match (either exact or similar), record the keyword and its matched content.

Your output should be in JSON format, where each key is a keyword from the list, and its value is the matched content from the description. Only include keywords that have matches. For example:

{
  "bite": "chew",
  "snow": "snow-covered"
}

Here are some important points to remember:
- Only include keywords that have matches in the description.
- If a keyword doesn't have a match, do not include it in the JSON.
- The matched content should be the exact text from the description, not a paraphrase.
- If there are multiple matches for a keyword, use the most relevant or closest match.

Please provide your answer in the following format:
<answer>
{
  // Your JSON output here
}
</answer>

Remember to only include the JSON in your answer, with no additional explanation or text. /no_think"""

MAIN_OBJECT_TEMPLATE = """A vision-language model was asked "__QUESTION__".
Reference label: "__TARGET__"
Model response: "__RESPONSE__"

Does the response identify the same main object/subject as the reference label?
- Judge the core object only. Ignore extra context in the label (scene, location, action, count, color), e.g. "dog" matches "Dog on leash", "ostrich" matches "Ostrich on road".
- Synonyms, hypernyms/hyponyms of the same thing, and more specific names count as a match (e.g. "bald eagle" matches "Eagle in flight", "springbok" matches "Antelope herd").
- A different object, or only a peripheral part of the scene, is NO.
- Only what the response says is actually shown counts: an object it only attributes to an annotation or text, or denies (e.g. "there is no X"), is NO.
Answer only YES or NO. /no_think"""
