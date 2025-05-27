import argparse
import asyncio
from contextlib import AsyncExitStack
from io import StringIO
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import pandas as pd
from typing import Optional

# Import your model (assuming model.py is in the same directory)
try:
    from model import Model  # Replace with your actual model class name
except ImportError:
    print("Warning: Could not import model.py. Please ensure model.py exists and contains your AI model.")
    Model = None


class MCPClient:
    def __init__(self, server_command=None):
        """
        Initialize MCP Client
        
        Args:
            server_command: Command to start the MCP server (e.g., ["python", "server.py"])
        """      
        self.server_command = server_command or ["python", "server.py"]
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()      

        if Model:
            self.model = Model()
            # print(self.model.accuracy)
        else:
            self.model = None
            print("Model not loaded. Please check model.py")

    async def connect_to_server(self):
        """Connect to the MCP server"""
        try:
            #StdioServerParameters allows server to be language-neutral and be easily embedded in different environment
            server_params = StdioServerParameters(
                command=self.server_command[0],
                args=self.server_command[1:] if len(self.server_command) > 1 else [],
                env=None
            )
            # This creates MCP transport protocol
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            # creates a session between client and server
            self.session = await self.exit_stack.enter_async_context(ClientSession(stdio_transport[0], stdio_transport[1]))

            # Initialize the session
            await self.session.initialize()

            # List available tools
            print("\n✅ Connected to MCP Server")

            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to server: {e}")
            return False

    async def load_data_from_server(self, data_path):
        """Load data using the MCP server's load_data tool"""
        if not self.session:
            print("❌ Not connected to server")
            return None
            
        try:
            # Call the load_data tool from the server
            result = await self.session.call_tool(
                "load_data",
                arguments={"data_path": data_path}
            )
        
            if not result.content:
                print("❌ No data received from server")
                return None
            """

            Result returns a TextContent
            Converting into json format to convert to pandas dataframe

            """
            if isinstance(result.content, list) and len(result.content) > 0:
                content = result.content[0]
                if hasattr(content, 'text'):
                    json_data = content.text
                else:
                    json_data = str(content)
            else:
                json_data = str(result.content)
            
            # Parse JSON and convert to DataFrame
            try:
                # First check if it's an error response
                parsed_data = json.loads(json_data)
                # print(parsed_data)
                if isinstance(parsed_data, dict) and "error" in parsed_data:
                    return None
            
                # Convert to DataFrame
                data_io = StringIO(parsed_data)
                df = pd.read_csv(data_io, delim_whitespace=True, index_col=0)
                return df
                
            except json.JSONDecodeError as e:
                return None
                
        except Exception as e:
            print(f"❌ Error loading data from server: {e}")
            return None

    def predict_with_model(self, data):
        """Use the AI model to make predictions on the data"""
        if not self.model:
            print("❌ Model not available")
            return None
            
        try:
            # Make predictions using your model
            predictions = self.model.predict(data) 
            print("✅ Predictions generated successfully")

            return predictions
            
        except Exception as e:
            print(f"❌ Error making predictions: {e}")
            return None

    async def process_data_and_predict(self, data_path):
        """Complete workflow: load data from server and make predictions"""
        print(f"🔄 Processing data from: {data_path}")
        
        # Load data from MCP server
        data = await self.load_data_from_server(data_path)
        if data is None:
            return None
        
        # Make predictions with the model
        predictions =  self.predict_with_model(data)
        if predictions is None:
            return None
        
        return predictions

    async def close(self):
        """Close the connection to the server"""
        await self.exit_stack.aclose()
        print("🔌 Disconnected from MCP Server")


class EnhancedMCPClient(MCPClient):
    """Enhanced MCP Client with additional features"""
    
    async def batch_process(self, data_paths):
        """Process multiple data files"""
        results = {}
        for path in data_paths:
            print(f"\n📊 Processing: {path}")
            result = await self.process_data_and_predict(path)
            results[path] = result
        return results
    
    async def get_server_info(self):
        """Get information about available tools from the server"""
        if not self.session:
            print("❌ Not connected to server")
            return None
            
        try:
            print("Get info for tools available")
            tools =  await self.session.list_tools()
            print("🔧 Available tools from server:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")
            return tools
            print("No tools were there")
        except Exception as e:
            print(f"❌ Error getting server info: {e}")
            return None


async def main():
    """Example usage of the MCP Client"""
    # Initialize client
    client = EnhancedMCPClient(["python", "server.py"])  # Adjust server command as needed
    
    try:
        # Connect to server
        connected = await client.connect_to_server()
        if not connected:
            print("Server connection failure")
            return
        
        # Get server information
        await client.get_server_info()
        
        # Process data and get predictions
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_path", type=str)
        args = parser.parse_args()
        data_path = args.data_path  # Replace with your actual data path
        predictions = await client.process_data_and_predict(data_path)
        
        if predictions is not None:
            print(f"🎯 Predictions: {predictions}")
        else:
            print("Predictions are empty")
        
    #     # Example of batch processing
    #     # data_paths = ["data1.csv", "data2.csv", "data3.csv"]
    #     # batch_results = await client.batch_process(data_paths)
    #     # print(f"📈 Batch results: {batch_results}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        # Clean up
        await client.close()


if __name__ == "__main__":
    # Run the client
    print("🚀 Starting MCP Client for AI Model")
    asyncio.run(main())