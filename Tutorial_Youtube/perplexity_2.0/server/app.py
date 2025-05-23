from typing import TypedDict, Annotated, Optional, List
from langgraph.graph import add_messages, StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from uuid import uuid4
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from IPython.display import Image, display
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages.ai import AIMessageChunk
import json
import os


load_dotenv()

llmGemini=ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

search_tool=TavilySearchResults(max_results=3)

tools=[search_tool]

memory=MemorySaver()

model_with_tools=llmGemini.bind_tools(tools=tools)

class State(TypedDict):
    messages: Annotated[List,add_messages]

async def model(state:State):
    result=await model_with_tools.ainvoke(input=state["messages"])
    state['messages']=[result]
    return state

async def tools_router(state: State):
    last_message=state["messages"][-1]
    return "tool_node" if hasattr(last_message,"tool_calls") and len(last_message.tool_calls)>0 else "end"

async def tool_node(state):
    """
        Custom tool node that handles tool calls from the LLM
    """
    tool_calls=state["messages"][-1].tool_calls

    # Initialize list to store tool messages
    tool_messages=[]

    # Process each tool call
    for tool_call in tool_calls:
        tool_name=tool_call["name"]
        tool_query=tool_call["args"]['query']
        tool_id=tool_call["id"]

        # Handle the search tool
        if tool_name=="tavily_search_results_json":
            # Execute the search tool with the provided arguments
            search_results=await search_tool.ainvoke(input=tool_query)

            # Create a ToolMessage for this result
            tool_message=ToolMessage(
                content=str(search_results),
                tool_call_id=tool_id,
                name=tool_name
            )
            tool_messages.append(tool_message)
    # print(tool_messages)
    state['messages']=tool_messages

    return state


graph_builder=StateGraph(state_schema=State)
graph_builder.add_node(node="model",action=model)
graph_builder.add_node(node="tool_node",action=tool_node)
graph_builder.add_edge(start_key="tool_node",end_key="model")
graph_builder.set_entry_point(key="model")
graph_builder.add_conditional_edges("model",path=tools_router,
                                    path_map={"tool_node":"tool_node","end":END}
                                   )
graph=graph_builder.compile(checkpointer=memory)

app=FastAPI()

# Add CORS middleware with settings that match frontend requirements
app.add_middleware(
    middleware_class=CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, adjust as needed
    allow_credentials=True,  # Allows credentials (cookies, authorization headers, etc.)
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
    expose_headers=["*"]  # Expose all headers to the client

)

def serialize_ai_message(chunk):
    if isinstance(chunk, AIMessageChunk):
        return chunk.content
    else:
        raise TypeError(f"Object of type {type(chunk)} is not correctly formatted for serialization.")



async def generate_chat_response(message:str, checkpoint_id:Optional[str]=None):
    is_new_conversation=checkpoint_id is None 

    if is_new_conversation:
        # generate a new checkpoint id for first message in conversation
        new_checkpoint_id=str(uuid4())

        config={"configurable":{"thread_id":new_checkpoint_id}}

        # Initialize with first message
        events=graph.astream_events(
            input={"messages":[HumanMessage(content=message)]},
            version="v2",
            config=config
            )
        
        # First send the checkpoint id to the client
        yield f"data: {{\"type\":\"checkpoint\", \"checkpoint_id\":\"{new_checkpoint_id}\"}}\n\n"
    else:
        config={"configurable":{"thread_id":checkpoint_id}}
        # continue with the existing conversation
        events=graph.astream_events(
            input={"messages":[HumanMessage(content=message)]},
            version="v2",
            config=config
        )
    
    async for event in events:
        event_type=event["event"]
        
        if event_type=="on_chat_model_stream":
            chunk_content=serialize_ai_message(chunk=event['data']['chunk'])
            if chunk_content:
                safe_content=chunk_content.replace("'","\\'").replace("\n","\\n") # escape single quotes and newlines
                # send the chunk to the client
                yield f"data: {{\"type\": \"content\", \"content\":\"{safe_content}\" }}\n\n"
            
        elif event_type=="on_chat_model_end":
            tool_calls=event["data"]["output"].tool_calls if hasattr(event["data"]["output"],"tool_calls") else []
            search_calls=[call for call in tool_calls if call['name']=="tavily_search_results_json"]
            if search_calls:
                # signal that a search is starting
                search_query=search_calls[0]['args'].get('query',"")
                # escape quotes and special characters
                safe_query=search_query.replace('"','\\"').replace("'","\\n").replace("\n","\\n")  # escape single quotes and newlines
                # send the search query to the client
                yield f"data: {{\"type\": \"search_start\", \"query\":\"{safe_query}\" }}\n\n"

        elif event_type=="on_tool_end" and event["name"]=="tavily_search_results_json":
            # search completed, send results or error
            output=event["data"]["output"]

            # check if output is a list
            if isinstance(output,list):
                # Extract URLs from the list of search results
                urls=[]
                for item in output:
                    if isinstance(item,dict) and "url" in item:
                        urls.append(item["url"])
                # Convert the list of URLs to a JSON string
                urls_json=json.dumps(obj=urls)
                # Send the URLs to the client
                yield f"data: {{\"type\": \"search_results\", \"urls\": {urls_json} }}\n\n"
    
    yield f"data: {{\"type\": \"end\"}}\n\n"  # signal end of the conversation

@app.get("/chat_stream/{message}")
async def chat_stream(message:str, checkpoint_id: Optional[str]=Query(default=None)):
    return StreamingResponse(
                            content=generate_chat_response(
                                                        message=message, 
                                                        checkpoint_id=checkpoint_id
                                                        ), 
                            media_type="text/event-stream"
                             )