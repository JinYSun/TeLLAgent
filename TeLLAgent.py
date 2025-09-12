import getpass
import os
from langgraph.prebuilt import ToolNode
from typing import Optional, Literal, Dict, List, Any
from prompt import prompt1, prompt2
import langchain
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain import chains
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import ValidationError
from langchain.agents import AgentExecutor
from langchain.agents.mrkl.base import ZeroShotAgent
from tools import make_tools
 

from langchain_ollama import OllamaLLM
import base64
from io import BytesIO
from PIL import Image
import os
from langchain_openai import ChatOpenAI, OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI
import json
import asyncio
import datetime
import hashlib
from collections import deque
import os

def load_api_keys(file_path='api.txt'):
     
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):  
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
    
                    if value == 'None':
                        continue
                    
                    os.environ[key] = value
                    print(f" {key}")
  
load_api_keys("api.txt")
class ReasoningTracker:
    """Enhanced reasoning tracker with better loop detection and performance optimization."""
    
    def __init__(self, max_history_size: int = 1000):
        self.reasoning_steps = []
        self.tool_calls = []
        self.supervisor_decisions = []
        self.iteration_count = 0
        self.tool_call_history = deque(maxlen=max_history_size)  # Use deque for better performance
        self.task_completion_signals = []
        self.error_count = 0
        self.consecutive_same_decisions = 0
        self.last_decision = None
        
        # Performance optimization: cache for repetitive checks
        self._completion_keywords_cache = {
            "task completed", "finished", "done", "completed successfully", 
            "final result", "final answer", "conclusion", "summary",
            "all requirements met", "objective achieved", "process complete",
            "task finished", "work done", "execution complete", "success"
        }
    
    def reset(self):
        """Reset the tracker for a new query."""
        self.reasoning_steps = []
        self.tool_calls = []
        self.supervisor_decisions = []
        self.iteration_count = 0
        self.tool_call_history.clear()
        self.task_completion_signals = []
        self.error_count = 0
        self.consecutive_same_decisions = 0
        self.last_decision = None
    
    def add_reasoning_step(self, agent_name: str, step_type: str, content: str, timestamp: Optional[datetime.datetime] = None):
        """Add a reasoning step with improved logging."""
        if timestamp is None:
            timestamp = datetime.datetime.now()
        
        step = {
            "agent": agent_name,
            "type": step_type,
            "content": content,
            "timestamp": timestamp,
            "iteration": self.iteration_count
        }
        self.reasoning_steps.append(step)
        
        # Only print if verbose mode is needed
        if step_type in ["error", "force_stop", "completion_signal", "routing_decision"]:
            print(f"[{timestamp.strftime('%H:%M:%S')}] {agent_name} - {step_type}: {content}")
    
    def add_tool_call(self, tool_name: str, input_data: Any, output_data: Any):
        """Add tool call with improved duplicate detection."""
        # Create a more reliable signature
        input_hash = hashlib.md5(str(input_data).encode()).hexdigest()
        call_signature = f"{tool_name}:{input_hash}"
        
        tool_call = {
            "tool": tool_name,
            "input": str(input_data),
            "output": str(output_data)[:500] + "..." if len(str(output_data)) > 500 else str(output_data),
            "timestamp": datetime.datetime.now(),
            "signature": call_signature
        }
        self.tool_calls.append(tool_call)
        self.tool_call_history.append(call_signature)
        
        print(f"[TOOL] {tool_name}: {str(input_data)[:50]}..." if len(str(input_data)) > 50 else f"[TOOL] {tool_name}: {input_data}")
    
    def add_supervisor_decision(self, decision: str, reasoning: str):
        """Add supervisor decision with loop detection."""
        self.iteration_count += 1
        
        # Check for consecutive same decisions (potential loop)
        if decision == self.last_decision:
            self.consecutive_same_decisions += 1
        else:
            self.consecutive_same_decisions = 0
        
        self.last_decision = decision
        
        decision_info = {
            "decision": decision,
            "reasoning": reasoning,
            "timestamp": datetime.datetime.now(),
            "iteration": self.iteration_count,
            "consecutive_count": self.consecutive_same_decisions
        }
        self.supervisor_decisions.append(decision_info)
        print(f"[SUPERVISOR #{self.iteration_count}] {decision} (consecutive: {self.consecutive_same_decisions})")
    
    def is_repetitive_tool_call(self, tool_name: str, input_data: Any, lookback: int = 5) -> bool:
        """Improved repetitive call detection with configurable lookback."""
        input_hash = hashlib.md5(str(input_data).encode()).hexdigest()
        call_signature = f"{tool_name}:{input_hash}"
        
        # Check recent history
        recent_calls = list(self.tool_call_history)[-lookback:]
        return recent_calls.count(call_signature) >= 2
    
    def has_task_completion_signals(self, content: str) -> bool:
        """Optimized completion signal detection."""
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in self._completion_keywords_cache)
    
    def should_stop_iteration(self, max_iterations: int = 10) -> tuple[bool, str]:
        """Enhanced stopping criteria with reason."""
        # Check iteration limit
        if self.iteration_count >= max_iterations:
            return True, f"Reached maximum iterations: {max_iterations}"
        
        # Check consecutive same decisions (loop detection)
        if self.consecutive_same_decisions >= 3:
            return True, f"Detected decision loop: {self.consecutive_same_decisions} consecutive '{self.last_decision}' decisions"
        
        # Check error count
        if self.error_count >= 3:
            return True, f"Too many errors: {self.error_count}"
        
        # Check completion signals
        recent_steps = self.reasoning_steps[-5:] if len(self.reasoning_steps) >= 5 else self.reasoning_steps
        completion_signals = sum(1 for step in recent_steps if self.has_task_completion_signals(step["content"]))
        
        if completion_signals >= 2:
            return True, f"Multiple completion signals detected: {completion_signals}"
        
        # Check tool call repetition patterns
        if len(self.tool_calls) >= 5:
            recent_tools = [call['tool'] for call in self.tool_calls[-5:]]
            if len(set(recent_tools)) <= 2 and len(recent_tools) >= 4:
                return True, "Repetitive tool usage pattern detected"
        
        return False, ""
    
    def add_error(self, error_msg: str):
        """Track errors."""
        self.error_count += 1
        self.add_reasoning_step("system", "error", error_msg)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "total_iterations": self.iteration_count,
            "total_tool_calls": len(self.tool_calls),
            "total_reasoning_steps": len(self.reasoning_steps),
            "error_count": self.error_count,
            "consecutive_same_decisions": self.consecutive_same_decisions,
            "unique_tools_used": len(set(call['tool'] for call in self.tool_calls))
        }
    
    def get_full_reasoning_trace(self) -> Dict[str, Any]:
        """Get complete reasoning trace with performance stats."""
        return {
            "reasoning_steps": self.reasoning_steps,
            "tool_calls": self.tool_calls,
            "supervisor_decisions": self.supervisor_decisions,
            "performance_stats": self.get_performance_stats(),
            "tool_call_history": list(self.tool_call_history)
        }

def convert_to_base64(pil_image):
    """Convert PIL image to base64 string."""
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def _make_llm(model: str, temp: float, api_key: str, streaming: bool = False):
    """Create LLM instance with improved error handling."""
    try:
        if model.startswith("claude"):
            return OpenAI(
                temperature=temp,
                model_name=model,
                max_tokens=5000,
                openai_api_key=api_key,
            )
        elif model.startswith("gpt") or model.startswith("deepseek"):
            return ChatOpenAI(
                model=model,
                temperature=temp,
                timeout=1000,
                base_url=os.getenv("OPENAI_API_BASE"),
                callbacks=[StreamingStdOutCallbackHandler()] if streaming else [],
                openai_api_key=api_key,
                max_tokens=5000,
            )
        elif model.startswith("llama"):
            return OllamaLLM(
                model=model,
                temperature=temp,
            )
        else:
            raise ValueError(f"Invalid model name: {model}")
    except Exception as e:
        raise ValueError(f"Failed to create LLM for model {model}: {str(e)}")

# Environment setup

from typing import Annotated
from pathlib import Path
from tempfile import TemporaryDirectory
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_core.messages import trim_messages
from IPython.display import Image, display
from langchain_core.messages import AnyMessage

# Working directory setup
_TEMP_DIRECTORY = TemporaryDirectory()
WORKING_DIRECTORY = Path(_TEMP_DIRECTORY.name)

class State(MessagesState):
    """Enhanced state with better tracking."""
    next: str
    reasoning_tracker: Optional[ReasoningTracker] = None  
    task_status: str = "in_progress"
    completion_check_count: int = 0
    current_iteration: int = 0
    final_result: Optional[str] = None
    accumulated_results: List[str] = []

class EnhancedReactAgent:
    """Enhanced wrapper for React agent with better error handling and tracking."""
    
    def __init__(self, base_agent, tracker: ReasoningTracker):
        self.base_agent = base_agent
        self.tracker = tracker
    
    def invoke(self, state: State) -> Dict[str, Any]:
        """Invoke agent with comprehensive error handling and tracking."""
        try:
            user_message = state["messages"][-1].content if state["messages"] else "No message"
            self.tracker.add_reasoning_step("tool_agent", "start_processing", 
                                          f"Processing: {user_message[:100]}...")
            
            # Invoke base agent
            result = self.base_agent.invoke(state)
            
            # Process results
            if result and "messages" in result and result["messages"]:
                self._process_agent_messages(result["messages"])
                final_content = result["messages"][-1].content
                self.tracker.add_reasoning_step("tool_agent", "completed", 
                                              f"Completed: {final_content[:100]}...")
            else:
                result = {"messages": [HumanMessage(content="No response from tool agent")]}
                self.tracker.add_reasoning_step("tool_agent", "no_result", "No messages returned")
            
            return result
            
        except Exception as e:
            self.tracker.add_error(f"Tool agent error: {str(e)}")
            return {"messages": [HumanMessage(content=f"Error in tool execution: {str(e)}")]}
    
    def _process_agent_messages(self, messages: List[BaseMessage]):
        """Process agent messages to extract tool calls and completion signals."""
        for msg in messages:
            # Check for tool calls
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get('name', 'unknown_tool')
                    tool_args = tool_call.get('args', {})
                    
                    if not self.tracker.is_repetitive_tool_call(tool_name, tool_args):
                        self.tracker.add_tool_call(tool_name, tool_args, "Executed")
                    else:
                        self.tracker.add_reasoning_step("tool_agent", "skip_repetitive", 
                                                      f"Skipped repetitive call to {tool_name}")
            
            # Check message content
            if hasattr(msg, 'content') and msg.content:
                content = str(msg.content)
                if self.tracker.has_task_completion_signals(content):
                    self.tracker.add_reasoning_step("tool_agent", "completion_signal", 
                                                  f"Completion signal: {content[:100]}...")

def make_supervisor_node(llm: BaseChatModel, members: List[str], tracker: ReasoningTracker, 
                        max_iterations: int = 10):
    """Create enhanced supervisor node with better decision making."""
    
    options = ["FINISH"] + members
    
    enhanced_prompt = f"""{prompt1}

ENHANCED STOPPING CRITERIA (CRITICAL):
- ALWAYS choose FINISH if the task has been completed successfully
- ALWAYS choose FINISH if you have a comprehensive answer to the user's question  
- ALWAYS choose FINISH if the same tools have been called multiple times with diminishing returns
- Current iteration: {tracker.iteration_count if tracker else 0}/{max_iterations}
- If iteration > {max_iterations//2}, be increasingly selective about continuing
- Look for completion signals: "finished", "completed", "done", "final result", etc.

DECISION GUIDELINES:
- Prioritize task completion over additional exploration
- Avoid repetitive tool calls unless they provide genuinely new information
- Consider the quality and completeness of existing results before requesting more work
"""

    class Router(TypedDict):
        """Enhanced router with better reasoning."""
        reasoning: str  
        confidence: float
        task_assessment: str
        completion_indicators: List[str]
        next: Literal[*options]

    def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
        """Enhanced supervisor with better loop prevention."""
        
        # Check stopping conditions
        should_stop, stop_reason = tracker.should_stop_iteration(max_iterations)
        if should_stop:
            tracker.add_reasoning_step("supervisor", "force_stop", f"Force stop: {stop_reason}")
            return Command(goto=END, update={"next": "FINISH", "task_status": "force_completed"})
        
        # Prepare messages with context
        messages = [{"role": "system", "content": enhanced_prompt}] + state["messages"]
        
        # Add iteration context
        iteration_context = f"""
CURRENT CONTEXT:
- Iteration: {tracker.iteration_count}/{max_iterations}
- Tool calls made: {len(tracker.tool_calls)}
- Errors encountered: {tracker.error_count}
- Consecutive same decisions: {tracker.consecutive_same_decisions}
- Recent completion signals: {sum(1 for step in tracker.reasoning_steps[-3:] if tracker.has_task_completion_signals(step['content']))}
"""
        messages.append({"role": "system", "content": iteration_context})
        
        try:
            response = llm.invoke(messages)
            goto = response["next"]
            reasoning = response.get("reasoning", "No reasoning provided")
            confidence = response.get("confidence", 0.5)
            
            # Log decision
            tracker.add_supervisor_decision(goto, reasoning)
            
            # Enhanced stopping logic
            if goto == "FINISH" or confidence < 0.3 or tracker.iteration_count >= max_iterations - 1:
                tracker.add_reasoning_step("supervisor", "task_completion", 
                                         f"Completing task: {goto}, confidence: {confidence}")
                goto = END
            
            task_status = "completed" if goto == END else "in_progress"
            return Command(goto=goto, update={"next": goto, "task_status": task_status})
            
        except Exception as e:
            tracker.add_error(f"Supervisor error: {str(e)}")
            return Command(goto=END, update={"next": "FINISH", "task_status": "error_completed"})

    return supervisor_node

class TeLLAgent:
    """Enhanced TeLLAgent with improved architecture and error handling."""
    
    def __init__(
        self,
        tools=None,
        model1: str = "deepseek-r1-250528",
        model2: str = "deepseek-v3.1-nothinking", 
        tools_model: str = "gpt-5",
        temp: float = 0.1,
        max_iterations: int = 8,
        verbose: bool = True,
        streaming: bool = True,
        openai_api_key: Optional[str] = None,
        api_keys: Dict[str, str] = {},
        file_path: str = r"...",
        image_path: str = r"...",
    ):
        """Initialize enhanced agent with better configuration."""
        self.file_path = file_path
        self.image_path = image_path
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Initialize tracker
        self.reasoning_tracker = ReasoningTracker()
        
        # Load environment
        load_dotenv()
        
        # Initialize LLMs with error handling
        try:
            self.llm1 = _make_llm(model1, temp, openai_api_key or os.getenv("OPENAI_API_KEY"), streaming)
            self.llm2 = _make_llm(model2, temp, openai_api_key or os.getenv("OPENAI_API_KEY"), streaming)
            self.reasoning_tracker.add_reasoning_step("system", "initialization", "LLMs initialized successfully")
        except Exception as e:
            raise ValueError(f"Failed to initialize LLMs: {str(e)}")

        # Initialize tools
        if tools is None:
            api_keys["OPENAI_API_KEY"] = openai_api_key or os.getenv("OPENAI_API_KEY")
            tools_llm = _make_llm(tools_model, temp, openai_api_key or os.getenv("OPENAI_API_KEY"), streaming)
            self.tools = make_tools(tools_llm, api_keys=api_keys, verbose=verbose, 
                                  image_path=image_path, file_path=file_path)
        else:
            self.tools = tools

    def run(self, query: str) -> tuple[Any, Dict[str, Any]]:
        """Run the agent with enhanced error handling and loop prevention."""
        
        # CRITICAL: Reset tracker for new query
        self.reasoning_tracker.reset()
        print(f"\n{'='*60}")
        print(f"STARTING NEW QUERY: {query}")
        print(f"{'='*60}")
        
        # Prepare query
        full_query = f"{query} {self.image_path} {self.file_path}".strip()
        self.reasoning_tracker.add_reasoning_step("system", "query_start", f"Starting: {full_query}")
        
        try:
            # Create enhanced tool agent
            base_tool_agent = create_react_agent(self.llm2, tools=self.tools, prompt=prompt2)
            enhanced_tool_agent = EnhancedReactAgent(base_tool_agent, self.reasoning_tracker)
            
            # Build graph
            graph = self._build_graph(enhanced_tool_agent)
            
            # Execute with comprehensive monitoring
            final_result = self._execute_graph(graph, full_query)
            
            return final_result, self.reasoning_tracker.get_full_reasoning_trace()
            
        except Exception as e:
            self.reasoning_tracker.add_error(f"Agent execution error: {str(e)}")
            return None, self.reasoning_tracker.get_full_reasoning_trace()
    
    def _build_graph(self, enhanced_tool_agent: EnhancedReactAgent) -> Any:
        """Build the agent graph with enhanced components."""
        
        def tool_agent_node(state: State) -> Command[Literal["supervisor"]]:
            should_stop, stop_reason = self.reasoning_tracker.should_stop_iteration(self.max_iterations)
            if should_stop:
                final_message = f"Task stopped: {stop_reason}"
                return Command(
                    update={
                        "messages": [HumanMessage(content=final_message, name="tool_agent")],
                        "task_status": "force_completed",
                        "final_result": final_message,
                        "accumulated_results": state.get("accumulated_results", []) + [final_message]
                    },
                    goto="supervisor",
                )
             
            result = enhanced_tool_agent.invoke(state)
            final_message = result["messages"][-1].content if result.get("messages") else "No result"
            
            
            accumulated_results = state.get("accumulated_results", []) + [final_message]
            
            return Command(
                update={
                    "messages": [HumanMessage(content=final_message, name="tool_agent")],
                    "final_result": final_message,
                    "accumulated_results": accumulated_results
                },
                goto="supervisor",
            )

        # Create supervisor with result preservation
        def enhanced_supervisor_node(state: State) -> Command[Literal["tool_agent", "__end__"]]:
            
            should_stop, stop_reason = self.reasoning_tracker.should_stop_iteration(self.max_iterations)
            if should_stop:
                self.reasoning_tracker.add_reasoning_step("supervisor", "force_stop", f"Force stop: {stop_reason}")
                 
                final_result = state.get("final_result") or state.get("accumulated_results", ["No results"])[-1]
                return Command(
                    goto=END, 
                    update={
                        "next": "FINISH", 
                        "task_status": "force_completed",
                        "final_result": final_result
                    }
                )
            
            
            messages = [{"role": "system", "content": f"""{prompt1}

ENHANCED STOPPING CRITERIA (CRITICAL):
- ALWAYS choose FINISH if the task has been completed successfully
- ALWAYS choose FINISH if you have a comprehensive answer to the user's question  
- Current iteration: {self.reasoning_tracker.iteration_count}/{self.max_iterations}
- Look for completion signals in the recent messages
"""}] + state["messages"]
             
            iteration_context = f"""
CURRENT CONTEXT:
- Iteration: {self.reasoning_tracker.iteration_count}/{self.max_iterations}
- Tool calls made: {len(self.reasoning_tracker.tool_calls)}
- Recent results available: {len(state.get("accumulated_results", []))}
"""
            messages.append({"role": "system", "content": iteration_context})
            
            try:
                from typing_extensions import TypedDict
                from typing import Literal
                
                class Router(TypedDict):
                    reasoning: str  
                    confidence: float
                    task_assessment: str
                    next: Literal["tool_agent", "FINISH"]

                # Try to get structured output, with fallback handling
                try:
                    response = self.llm1.with_structured_output(Router).invoke(messages)
                    goto = response.get("next")
                    
                    # Handle case where 'next' is None or missing
                    if goto is None:
                        # Fallback logic based on current state
                        if (self.reasoning_tracker.iteration_count >= self.max_iterations - 1 or 
                            len(state.get("accumulated_results", [])) > 0):
                            goto = "FINISH"
                        else:
                            goto = "tool_agent"
                    
                    reasoning = response.get("reasoning", "No reasoning provided")
                    confidence = response.get("confidence", 0.5)
                    
                except (KeyError, AttributeError, ValidationError, json.JSONDecodeError) as struct_error:
                    # Fallback to regular LLM call if structured output fails
                    self.reasoning_tracker.add_error(f"Structured output failed: {struct_error}, falling back to regular call")
                    fallback_response = self.llm1.invoke(messages + [{"role": "system", "content": "Respond with either 'tool_agent' or 'FINISH' based on whether more tool usage is needed."}])
                    response_text = fallback_response.content if hasattr(fallback_response, 'content') else str(fallback_response)
                    
                    # Simple text parsing
                    if "FINISH" in response_text.upper() or "finish" in response_text.lower():
                        goto = "FINISH"
                    else:
                        goto = "tool_agent"
                    
                    reasoning = f"Fallback decision based on text parsing: {response_text[:100]}..."
                    confidence = 0.5
                
                self.reasoning_tracker.add_supervisor_decision(goto, reasoning)
                
                if goto == "FINISH" or confidence < 0.3 or self.reasoning_tracker.iteration_count >= self.max_iterations - 1:
                    self.reasoning_tracker.add_reasoning_step("supervisor", "task_completion", 
                                                           f"Completing task: {goto}, confidence: {confidence}")
                     
                    final_result = state.get("final_result") or "Task completed"
                    return Command(
                        goto=END, 
                        update={
                            "next": "FINISH", 
                            "task_status": "completed",
                            "final_result": final_result
                        }
                    )
                
                return Command(
                    goto="tool_agent", 
                    update={
                        "next": "tool_agent", 
                        "task_status": "in_progress"
                    }
                )
                
            except Exception as e:
                self.reasoning_tracker.add_error(f"Supervisor error: {str(e)}")
                final_result = state.get("final_result") or f"Error occurred: {str(e)}"
                return Command(
                    goto=END, 
                    update={
                        "next": "FINISH", 
                        "task_status": "error_completed",
                        "final_result": final_result
                    }
                )

        # Build basic graph
        builder = StateGraph(State)
        builder.add_node("supervisor", enhanced_supervisor_node)
        builder.add_node("tool_agent", tool_agent_node)
        builder.add_edge(START, "supervisor")
        basic_graph = builder.compile()
        
        # Create team wrapper that preserves results
        def call_tool_team(state: State) -> Command[Literal["supervisor"]]:
            self.reasoning_tracker.add_reasoning_step("tool_team", "delegation", 
                                                     f"Tool team called (iteration {self.reasoning_tracker.iteration_count})")
            
             
            response = basic_graph.invoke({
                "messages": state["messages"],
                "accumulated_results": state.get("accumulated_results", [])
            })
            
            
            final_content = response.get("final_result") or (
                response["messages"][-1].content if response.get("messages") else "Tool team completed"
            )
            
            self.reasoning_tracker.add_reasoning_step("tool_team", "completed", 
                                                     f"Tool team completed: {final_content[:100]}...")
            
        
            if self.reasoning_tracker.has_task_completion_signals(final_content):
                self.reasoning_tracker.add_reasoning_step("tool_team", "completion_signal_detected", 
                                                         "Task completion signal detected in tool team result")
            
            return Command(
                update={
                    "messages": [HumanMessage(content=final_content, name="tool_team")],
                    "final_result": final_content,
                    "accumulated_results": state.get("accumulated_results", []) + [final_content]
                },
                goto="supervisor",
            )

       
        def super_supervisor_node(state: State) -> Command[Literal["tool_team", "__end__"]]:
            should_stop, stop_reason = self.reasoning_tracker.should_stop_iteration(self.max_iterations)
            if should_stop:
                self.reasoning_tracker.add_reasoning_step("super_supervisor", "force_stop", f"Force stop: {stop_reason}")
                final_result = state.get("final_result") or "Task completed with limits reached"
                return Command(
                    goto=END, 
                    update={
                        "next": "FINISH", 
                        "task_status": "force_completed",
                        "final_result": final_result
                    }
                )
            
             
            messages = [{"role": "system", "content": f"""{prompt1}

TEAM COORDINATION CRITERIA:
- Delegate to tool_team if the task requires tool usage or research
- Choose FINISH if you have a complete answer to the user's question
- Current iteration: {self.reasoning_tracker.iteration_count}/{self.max_iterations}
- Available results: {len(state.get("accumulated_results", []))}
"""}] + state["messages"]
            
            try:
                from typing_extensions import TypedDict
                from typing import Literal
                
                class TeamRouter(TypedDict):
                    reasoning: str  
                    confidence: float
                    next: Literal["tool_team", "FINISH"]

                # Try to get structured output, with fallback handling
                try:
                    response = self.llm1.with_structured_output(TeamRouter).invoke(messages)
                    goto = response.get("next")  # Use .get() instead of direct access
                    
                    # Handle case where 'next' is None or missing
                    if goto is None:
                        # Fallback logic based on current state
                        if (self.reasoning_tracker.iteration_count >= self.max_iterations - 1 or 
                            len(state.get("accumulated_results", [])) > 0):
                            goto = "FINISH"
                        else:
                            goto = "tool_team"
                    
                    reasoning = response.get("reasoning", "No reasoning provided")
                    confidence = response.get("confidence", 0.5)
                    
                except (KeyError, AttributeError, ValidationError, json.JSONDecodeError) as struct_error:
                    # Fallback to regular LLM call if structured output fails
                    self.reasoning_tracker.add_error(f"Structured output failed: {struct_error}, falling back to regular call")
                    fallback_response = self.llm1.invoke(messages + [{"role": "system", "content": "Respond with either 'tool_team' or 'FINISH' based on whether more tool usage is needed."}])
                    response_text = fallback_response.content if hasattr(fallback_response, 'content') else str(fallback_response)
                    
                    # Simple text parsing
                    if "FINISH" in response_text.upper() or "finish" in response_text.lower():
                        goto = "FINISH"
                    else:
                        goto = "tool_team"
                    
                    reasoning = f"Fallback decision based on text parsing: {response_text[:100]}..."
                    confidence = 0.5
                
                self.reasoning_tracker.add_supervisor_decision(goto, reasoning)
                
                if goto == "FINISH" or confidence < 0.3 or self.reasoning_tracker.iteration_count >= self.max_iterations - 1:
                    final_result = state.get("final_result") or "Task completed by supervisor decision"
                    return Command(
                        goto=END, 
                        update={
                            "next": "FINISH", 
                            "task_status": "completed",
                            "final_result": final_result
                        }
                    )
                
                return Command(
                    goto="tool_team", 
                    update={
                        "next": "tool_team", 
                        "task_status": "in_progress"
                    }
                )
                
            except Exception as e:
                self.reasoning_tracker.add_error(f"Super supervisor error: {str(e)}")
                final_result = state.get("final_result") or f"Error in supervision: {str(e)}"
                return Command(
                    goto=END, 
                    update={
                        "next": "FINISH", 
                        "task_status": "error_completed",
                        "final_result": final_result
                    }
                )
        
        super_builder = StateGraph(State)
        super_builder.add_node("supervisor", super_supervisor_node)
        super_builder.add_node("tool_team", call_tool_team)
        super_builder.add_edge(START, "supervisor")
        
        return super_builder.compile()
    
    def _execute_graph(self, graph: Any, query: str) -> Any:
        """Execute graph with enhanced monitoring and result extraction."""
        
        final_result = None
        final_answer = None
        accumulated_results = []
        recursion_limit = min(200, self.max_iterations * 10)
        
        try:
            
            final_state = None
            for step_result in graph.stream(
                {
                    "messages": [("user", query)],
                    "task_status": "in_progress",
                    "completion_check_count": 0,
                    "accumulated_results": []
                },
                {"recursion_limit": recursion_limit},
            ):
                if self.verbose:
                    print(f"Step: {step_result}")
                    print("---")
                
                # 保存每个步骤的结果
                final_result = step_result
                final_state = step_result
                 
                for node_name, node_data in step_result.items():
                    if isinstance(node_data, dict):
                      
                        if 'final_result' in node_data and node_data['final_result']:
                            final_answer = node_data['final_result']
                            self.reasoning_tracker.add_reasoning_step("system", "result_captured", 
                                                                     f"Captured result from {node_name}: {final_answer[:100]}...")
                         
                        if 'accumulated_results' in node_data and node_data['accumulated_results']:
                            accumulated_results = node_data['accumulated_results']
                        
                       
                        if 'messages' in node_data and node_data['messages']:
                            for msg in node_data['messages']:
                                if hasattr(msg, 'content') and msg.content and msg.name in ['tool_agent', 'tool_team']:
                                     
                                    content = msg.content
                                    if not content.startswith("Task stopped") and not content.startswith("Error"):
                                        final_answer = content
                                        self.reasoning_tracker.add_reasoning_step("system", "tool_result_captured", 
                                                                                 f"Captured tool result: {content[:100]}...")
                
                # 紧急停止检查
                if self.reasoning_tracker.iteration_count >= self.max_iterations:
                    self.reasoning_tracker.add_reasoning_step("system", "emergency_stop", 
                                                            "Emergency stop - max iterations reached")
                    break
            
            # 构建最终结果
            if final_answer:
                 
                result = {
                    "final_answer": final_answer,
                    "execution_status": "completed",
                    "accumulated_results": accumulated_results,
                    "raw_final_state": final_state
                }
            elif accumulated_results:
           
                result = {
                    "final_answer": accumulated_results[-1],
                    "execution_status": "completed", 
                    "accumulated_results": accumulated_results,
                    "raw_final_state": final_state
                }
            else:
                
                result = final_result
                if final_result:
                   
                    extracted_content = self._extract_meaningful_content(final_result)
                    if extracted_content:
                        result = {
                            "final_answer": extracted_content,
                            "execution_status": "completed",
                            "raw_final_state": final_result
                        }
                        
        except Exception as e:
            self.reasoning_tracker.add_error(f"Graph execution error: {str(e)}")
            result = {
                "final_answer": f"Execution error: {str(e)}",
                "execution_status": "error",
                "raw_final_state": final_result
            }
        
        self.reasoning_tracker.add_reasoning_step("system", "execution_complete", 
                                                f"Completed after {self.reasoning_tracker.iteration_count} iterations")
        
        return result
    
    def _extract_meaningful_content(self, result: Any) -> Optional[str]:
 
        try:
            if isinstance(result, dict):
                for node_name, node_data in result.items():
                    if isinstance(node_data, dict) and 'messages' in node_data:
                        messages = node_data['messages']
                        if messages:
                            for msg in messages:
                                if hasattr(msg, 'content') and msg.content:
                                    content = msg.content
                                     
                                    if (not content.startswith("Task stopped") and 
                                        not content.startswith("Error") and
                                        len(content.strip()) > 10):  
                                        return content
            return None
        except:
            return None

if __name__ == '__main__':
    # Example usage with proper reset handling
    agent = TeLLAgent(
        temp=0.1, 
        streaming=True,
        model1="deepseek-r1-250528",
        model2="deepseek-v3.1-nothinking", 
        openai_api_key=os.getenv("OPENAI_API_KEY"), 
        max_iterations=50,
        verbose=True,
        image_path=r"...",
        file_path=r" "
    )
    
    # First query
    print("\n" + "="*80)
    print("FIRST QUERY")
    print("="*80)
    result1, reasoning_trace1 = agent.run("Generate a donor with PCE = 12% and give all its properties. Then, ask human to give you three acceptors give me the best match donor/acceptor pairs")
    
    print("\n" + "="*60)
    print("FIRST QUERY RESULTS:")
    print("="*60)
    if result1 and isinstance(result1, dict) and 'final_answer' in result1:
        print(result1['final_answer'])
    else:
        print(f"Result: {result1}")
   