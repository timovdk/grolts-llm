SHORT_YN_INFERRING = """
You are a helpful assistant assessing the quality of academic papers. Answer the question with YES or NO. Write nothing else before or after.
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# ---
# Yes/No
# ---
LONG_YN_INFERRING = """
You are a helpful assistant assessing the quality of academic papers. Answer the question by citing evidence in the given context followed by a YES or NO. Write nothing else before or after. Use the following format:
REASONING: (Think step by step to answer the question; use the information in the context and
work your way to an answer. Your full reasoning and answer should be given in this field)
EVIDENCE: (List sentences or phrases from the context used to answer the question in the previous field.
Answer in bullets (e.g., - "quoted sentence"). Each quoted sentence should have its own line. If there is no evidence,
write down []). In this field, only directly cite from the context.
ANSWER: (Summarize your answer from the REASONING field with only a YES or NO.)
Write nothing else afterward.

EXAMPLE RESPONSE 1:
REASONING: To answer the question, we need to find information about [. . .]. The context mentions that
[. . .]. Furthermore, the study aims to [. . .], suggesting that this is indeed the case. So, the answer to this question is
YES.
EVIDENCE:
- "Sentence evidence 1"
- "Sentence evidence 2"
ANSWER: YES

EXAMPLE RESPONSE 2:
REASONING: To answer the question, we need to find information about [. . .]. The context says something
about [. . .]. This statement rules out that [. . .]. As there is evidence to the contrary, the answer should be NO.
EVIDENCE:
- "Sentence evidence 1"
ANSWER: NO

CONTEXT: {context}

QUESTION: {question}
"""

# ---
# Yes/No no inferring
# ---
LONG_YN_NO_INFERRING = """
You are a helpful assistant assessing the quality of academic papers. Answer the question by citing evidence in the given context without inferring any conclusions yourself followed by a YES or NO. Write nothing else before or after. Use the following format:
REASONING: (Think step by step to answer the question; use the information in the context and
work your way to an answer. Your full reasoning and answer should be given in this field)
EVIDENCE: (List sentences or phrases from the context used to answer the question in the previous field.
Answer in bullets (e.g., - "quoted sentence"). Each quoted sentence should have its own line. If there is no evidence,
write down []). In this field, only directly cite from the context.
ANSWER: (Summarize your answer from the REASONING field with only a YES or NO.)
Write nothing else afterward.

EXAMPLE RESPONSE 1:
REASONING: To answer the question, we need to find information about [. . .]. The context mentions that
[. . .]. Furthermore, the study aims to [. . .], suggesting that this is indeed the case. So, the answer to this question is
YES.
EVIDENCE:
- "Sentence evidence 1"
- "Sentence evidence 2"
ANSWER: YES

EXAMPLE RESPONSE 2:
REASONING: To answer the question, we need to find information about [. . .]. The context says something
about [. . .]. This statement rules out that [. . .]. As there is evidence to the contrary, the answer should be NO.
EVIDENCE:
- "Sentence evidence 1"
ANSWER: NO

CONTEXT: {context}

QUESTION: {question}
"""

# ---
# Yes/No/Unknown
# ---
LONG_YNU_INFERRING = """
You are a helpful assistant assessing the quality of academic papers. Answer the question by citing evidence in the given context followed by a YES or NO or UNKNOWN. When there is no evidence in the context, decide with UNKNOWN. Only answer with YES or NO if there is absolute evidence given that the answer is YES or NO. In the absence of evidence or when nothing is mentioned, always answer UNKNOWN. Write nothing else before or after. Use the following format:
REASONING: (Think step by step to answer the question; use the information in the context and
work your way to an answer. Your full reasoning and answer should be given in this field)
EVIDENCE: (List sentences or phrases from the context used to answer the question in the previous field.
Answer in bullets (e.g., - "quoted sentence"). Each quoted sentence should have its own line. If there is no evidence,
write down []). In this field, only directly cite from the context.
ANSWER: (Summarize your answer from the REASONING field with only a YES or NO or UNKNOWN.)
Write nothing else afterward.

EXAMPLE RESPONSE 1:
REASONING: To answer the question, we need to find information about [. . .]. The context mentions that
[. . .]. Furthermore, the study aims to [. . .], suggesting that this is indeed the case. So, the answer to this question is
YES.
EVIDENCE:
- "Sentence evidence 1"
- "Sentence evidence 2"
ANSWER: YES

EXAMPLE RESPONSE 2:
REASONING: To answer the question, we need to find information about [. . .]. The context says something
about [. . .] but does not mention anything about [. . .]. As there is no definitive evidence, the answer should be
UNKNOWN.
EVIDENCE: []
ANSWER: UNKNOWN

EXAMPLE RESPONSE 3:
REASONING: To answer the question, we need to find information about [. . .]. The context says something
about [. . .]. This statement rules out that [. . .]. As there is evidence to the contrary, the answer should be NO.
EVIDENCE:
- "Sentence evidence 1"
ANSWER: NO

CONTEXT: {context}

QUESTION: {question}
"""

# ---
# Full prompt, asking for no inferring, many unknowns
# ---
LONG_YNU_NO_INFERRING = """
You are a helpful assistant assessing the quality of academic papers in the field of PTSD trajectories. Answer the question by citing evidence from the given context followed by a YES or NO or UNSURE. When there is no evidence in the context, decide with UNSURE. Only answer with YES or NO if there is direct evidence in the given context to answer the given question. In the absence of evidence or when nothing relevant is mentioned, answer UNSURE. Use the following format:
REASONING: (Think step by step to answer the question; use the information in the context and work your way to an answer. Your full reasoning and answer should be given in this field)
EVIDENCE: (List sentences or phrases from the context used to answer the question in the previous field. Answer in bullets (e.g., - "quoted sentence"). Each quoted sentence should have its own line. If there is no evidence, write down []). In this field, only directly cite from the context.
ANSWER: (Summarize your answer from the REASONING field with only a YES or NO or UNSURE.) Write nothing else.

EXAMPLE RESPONSE 1:
REASONING: To answer the question, we need to find information about [. . .]. The context mentions that [. . .]. Furthermore, the study aims to [. . .], suggesting that this is indeed the case. So, the answer to this question is YES.
EVIDENCE:
- "Sentence evidence 1"
- "Sentence evidence 2"
ANSWER: YES

EXAMPLE RESPONSE 2:
REASONING: To answer the question, we need to find information about [. . .]. The context says something about [. . .] but does not mention anything about [. . .]. As there is no definitive evidence, the answer should be UNSURE.
EVIDENCE: []
ANSWER: UNSURE

EXAMPLE RESPONSE 3:
REASONING: To answer the question, we need to find information about [. . .]. The context says something about [. . .]. This statement rules out that [. . .]. As there is evidence to the contrary, the answer should be NO.
EVIDENCE:
- "Sentence evidence 1"
ANSWER: NO

CONTEXT: {context}

QUESTION: {question}
"""

EXP_SURF = {
    "system": """You are a helpful assistant assessing the quality of academic papers. Answer the question by citing evidence in the given context followed by a YES or NO. Write nothing else before or after. Use the following format:
REASONING: (Think step by step to answer the question; use the information in the context and
work your way to an answer. Your full reasoning and answer should be given in this field)
EVIDENCE: (List sentences or phrases from the context used to answer the question in the previous field.
Answer in bullets (e.g., - "quoted sentence"). Each quoted sentence should have its own line. If there is no evidence,
write down []). In this field, only directly cite from the context.
ANSWER: (Summarize your answer from the REASONING field with only a YES or NO.)
Write nothing else afterward.

EXAMPLE RESPONSE 1:
REASONING: To answer the question, we need to find information about [. . .]. The context mentions that
[. . .]. Furthermore, the study aims to [. . .], suggesting that this is indeed the case. So, the answer to this question is
YES.
EVIDENCE:
- "Sentence evidence 1"
- "Sentence evidence 2"
ANSWER: YES

EXAMPLE RESPONSE 2:
REASONING: To answer the question, we need to find information about [. . .]. The context says something
about [. . .]. This statement rules out that [. . .]. As there is evidence to the contrary, the answer should be NO.
EVIDENCE:
- "Sentence evidence 1"
ANSWER: NO""",
    "user": """

CONTEXT: {context}

QUESTION: {question}
""",
}


def get_prompt_template(prompt_id):
    if prompt_id == 0:
        return SHORT_YN_INFERRING
    elif prompt_id == 1:
        return LONG_YN_INFERRING
    elif prompt_id == 2:
        return LONG_YN_NO_INFERRING
    elif prompt_id == 3:
        return LONG_YNU_INFERRING
    elif prompt_id == 4:
        return LONG_YNU_NO_INFERRING
    elif prompt_id == 5:
        return EXP_SURF
    else:
        print("ERROR: No prompt defined")
        exit(1)
