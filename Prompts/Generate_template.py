from langchain_core.prompts import PromptTemplate

template=PromptTemplate(
  template="""
Please summarize the research paper titled {paper_input} with the following specifications:
Explanation Style: {style_input}
Explanation Length: {length_input}

Ensure the summary includes:

Relevant mathematical equations if present.
Explanation of mathematical concepts using simple, intuitive code snippets where applicable.
Relatable analogies.
If any piece of information is not available, state "Insufficient Information Available" instead of hallucinating.
""",
input_variables=["paper_input","style_input","length_input"]
)

template.save("template.json")