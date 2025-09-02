---
mode: agent
---
You are working to improve the quality of the responses of a number of mcp tools managed in this workspace. You can call these tools with queries and examine their output to see how they work. if you update code then the imas-debug server specified in the mcp.json file will need to be restarted for these changes to take effect.

Implement Phase {phase} of the `imas_mcp/IMAS_MCP_TOOLS_ANALYSIS_REPORT.md` file that describes how to improve specific tools and features therein.

Examine the `imas_mcp/resources` and `imas_mcp/definitions` dirs for useful resources and physics definitions. I preference storing definition data, such as that for physics or relationships in separate files rather than including it within code. 

When creating new files, classes or methods, name them with simple, clear and descriptive names. Do not include qualifiers such as 'enhanced', 'improved', 'fixed', 'phase 1' ext. Examine current project structure store files appropriately. extend existing files and code instead of creating new scripts if this is the best option.

When creating processing pipelines for these mcp tools examine the `imas_mcp/search/decorators` directory for existing decorators that could be applied. Use the sample decorator for any AI augmentation via client side mcp sampling.

Validate and check off success metrics once done. Use vscode test explorer to run tests.
Give test files descriptive and clear names that do not mention the phases of this implementation plan. Examine the current dir structure of the tests and organize new test files appropriately, or extend current test files

Once implemented, update the analysis report with these findings and mark successful metrics as complete. 

Update the documentation for updated tools so that it is sufficient for an LLM to know how to use it properly. Improve a tool's documentation with a focus on aiding LLMs in correct tool usage. Include usage examples. Follow best practice for mcp tools.

