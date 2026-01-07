# <!-- ATLERNATIVE FOR LLM THAT DOES NOT PROVIDES US WITH_STRUCTURED_OUTPUT METHOD -->

**jsonOutputParser()**-> to get ouput in json format(No control over output)
 
**StructuredOutputParser()**-> to get output in desired json format (Not supported in latest langchain model)

**PydanticOutputParser()**-> to get data with proper constraints and customization(like default,regex,to add description etc.)

**strOutputParser()**-> to get the content of result from llm (Useful in chaining)
