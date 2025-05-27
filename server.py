import pandas as pd
from fastmcp import FastMCP 


class MCPServer:

    def __init__(self):
        self.mcp = FastMCP("Data_loader_server")
        print('MCP_Server_Loaded')
        self._register_tools()
        
    def _register_tools(self):
        # An MCP tool 
        @self.mcp.tool()
        async def load_data(data_path):
            print("Server tool being called")
            try:
                df = pd.read_csv(data_path)
                return df
            except Exception as e:
                print(f"❌ Error loading data: {e}")
                return None

    def runner(self):
        print("MCP Loader")
        # transport: Literal["stdio", "streamable-http", "sse"]
        self.mcp.run(transport="stdio")

if __name__ == '__main__':
    loader_server = MCPServer()
    loader_server.runner()