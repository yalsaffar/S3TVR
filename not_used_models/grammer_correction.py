import language_tool_python
# This is just a simple implementation of the language_tool_python library, I will implement later

def language_tool(language="en-US"):

    tool = language_tool_python.LanguageTool(language)

    return tool


def correct_grammar(text,tool):
    
    result = tool.correct(text)
    return result
