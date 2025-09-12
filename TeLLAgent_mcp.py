import getpass
import os
import datetime
import json
from typing import Dict, List, Any, Optional
from langchain_ollama import OllamaLLM
from langchain_mcp_adapters.client import MultiServerMCPClient
def load_api_keys(file_path="api.txt"):

    with open(file_path, 'r', encoding='utf-8') as f:
        exec(f.read(), {}, globals())

    for key, value in globals().items():
        if not key.startswith('__') and isinstance(value, str):
            os.environ[key] = value

load_api_keys("api.txt")
import langchain
from typing import Annotated, List
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional
from langchain_openai import ChatOpenAI, OpenAI
import asyncio
from typing_extensions import TypedDict
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.tools import tool
import threading
import queue
import time
import sys
import select
import hashlib

_TEMP_DIRECTORY = TemporaryDirectory()
WORKING_DIRECTORY = Path(_TEMP_DIRECTORY.name)

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from prompt import prompt1, prompt2
from typing import List, Optional, Literal
from langchain_core.language_models.chat_models import BaseChatModel

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, trim_messages
from IPython.display import Image, display
from langchain_core.messages import AnyMessage

def _make_llm(model, temp, api_key, streaming: bool = False):
    if model.startswith("claude"):
        llm = OpenAI(
            temperature=temp,
            model_name=model,
            max_tokens=5000,
            openai_api_key=api_key,

        )
    elif model.startswith("gpt") or model.startswith("deepseek"):
        llm = ChatOpenAI(model=model,
            temperature=0.1,

            timeout=1000,
            base_url=os.getenv("OPENAI_API_BASE"),
            callbacks=[StreamingStdOutCallbackHandler()],
            openai_api_key=api_key, max_tokens=5000,
        )
    elif model.startswith("llama"):
        llm = OllamaLLM(model=model,
            temperature=0.1,
        )
    else:
        raise ValueError(f"Invalid model name: {model}")
    return llm

# Improved human tool implementation
class HumanTool:
    def __init__(self):
        self._is_waiting = False
        self._response = None
        self.question_history = {}  # Store question history to avoid duplicates
        self.response_cache = {}    # Cache human responses
        self.interaction_count = 0  # Interaction counter
    
    def _generate_question_hash(self, question: str) -> str:
        """Generate a unique hash for the question to deduplicate"""
        return hashlib.md5(question.lower().strip().encode()).hexdigest()[:8]
    
    def ask_human_simple(self, question: str) -> str:
        """
        Simple synchronous human input – with deduplication check
        """
        # Check if this is a duplicate question
        question_hash = self._generate_question_hash(question)
        
        if question_hash in self.question_history:
            cached_response = self.response_cache.get(question_hash)
            if cached_response:
                print(f"\n🔄 DUPLICATE QUESTION DETECTED")
                print(f"Question: {question}")
                print(f"Previous response: {cached_response}")
                print(f"Returning cached response to avoid repetition.")
                return cached_response
        
        self._is_waiting = True
        self.interaction_count += 1
        
        print("\n" + "="*60)
        print(f"🤖 REQUESTING HUMAN ASSISTANCE #{self.interaction_count}")
        print("="*60)
        print(f"Question: {question}")
        print("Please provide your response:")
        print("-" * 60)
        
        try:
            response = input("> ").strip()
            self._is_waiting = False
            
            # Record question and response
            self.question_history[question_hash] = question
            self.response_cache[question_hash] = response
            
            print(f"✅ Received human response: {response}")
            self._response = response
            return response
            
        except KeyboardInterrupt:
            print("\n❌ Human input interrupted by user")
            self._is_waiting = False
            self._response = "HUMAN_INPUT_INTERRUPTED"
            return self._response
        except Exception as e:
            print(f"\n❌ Error getting human input: {str(e)}")
            self._is_waiting = False
            self._response = f"INPUT_ERROR: {str(e)}"
            return self._response
    
    def is_waiting_for_human(self) -> bool:
        """Check whether the system is waiting for human input"""
        return self._is_waiting
    
    def get_interaction_summary(self) -> Dict:
        """Get interaction summary"""
        return {
            "total_interactions": self.interaction_count,
            "unique_questions": len(self.question_history),
            "cached_responses": len(self.response_cache)
        }

# Global human tool instance
human_tool_instance = HumanTool()

@tool
def request_human_assistance(question: str, urgency: str = "normal", context: str = "", expected_format: str = "", follow_up_allowed: bool = True) -> str:
    """
    Request help from a human operator when the AI needs clarification, decision support, or expert knowledge.
    
    Usage scenarios:
    - Need clarification of ambiguous requirements
    - Need human judgment or preference-based decisions
    - Encounter complex problems requiring human expertise
    - Need subjective evaluation or creative input
    - Need confirmation before potentially impactful actions
    - Follow-up questions when human reply is incomplete
    
    Args:
        question: Clear, specific question for the human
        urgency: Priority ("low", "normal", "high")
        context: Additional context to help the human understand the background
        expected_format: Expected answer format (e.g., "Please provide 3 acceptor molecule names")
        follow_up_allowed: Whether follow-up questions are allowed (to avoid infinite loops)
    
    Returns:
        Human reply or timeout message
    """
    global human_tool_instance
    global global_tracker
    
    # Add urgency indicator
    urgency_indicator = {
        "low": "🟢",
        "normal": "🟡", 
        "high": "🔴"
    }.get(urgency, "🟡")
    
    # Construct full question
    formatted_question = f"{urgency_indicator} [{urgency.upper()}] {question}"
    if context:
        formatted_question += f"\n\nContext: {context}"
    if expected_format:
        formatted_question += f"\n\nExpected format: {expected_format}"
    
    # Obtain human reply
    human_response = human_tool_instance.ask_human_simple(formatted_question)
    
    # Log interaction in tracker
    global_tracker.add_human_interaction(question, human_response, urgency)
    
    # Return clear reply format
    if not any(keyword in human_response.lower() for keyword in ['timeout', 'error', 'interrupted']):
        result = f"Human Response: {human_response}\n\n"
        
        # Check whether the reply is complete (smarter logic can be added here)
        is_complete = _validate_human_response(human_response, question, expected_format)
        
        if is_complete:
            result += "IMPORTANT: Use this human feedback to continue and complete the task. The response appears complete."
            # Mark that human input has been received
            global_tracker.mark_human_input_received(question, human_response)
        else:
            result += "NOTE: This response may be incomplete or unclear. Consider asking follow-up questions for clarification if needed."
            if follow_up_allowed:
                result += " You can use request_human_assistance again with follow_up_allowed=True for clarification."
    else:
        result = f"Human Response Error: {human_response}"
    
    return result

def _validate_human_response(response: str, original_question: str, expected_format: str = "") -> bool:
    """
    Validate whether the human reply is complete
    Returns True if the reply appears complete, False if more information may be needed
    """
    response_lower = response.lower().strip()
    
    # Clearly incomplete replies
    incomplete_indicators = [
        "i don't know", "not sure", "unclear", "what do you mean",
        "can you clarify", "need more info", "more information",
        "don't know", "unclear", "what do you mean", "need more", "uncertain"
    ]
    
    if any(indicator in response_lower for indicator in incomplete_indicators):
        return False
    
    # Check for overly brief replies
    if len(response.strip()) < 10:
        return False
    
    # If the original question asks for multiple items, check that the reply has multiple items
    if any(keyword in original_question.lower() for keyword in ["three", "3", "multiple", "several", "list"]):
        # Simple check for multiple items (separated by comma, semicolon, newline, etc.)
        separators = [",", ";", "\n", "and", "or", "、"]
        has_multiple_items = any(sep in response for sep in separators) or len(response.split()) > 5
        if not has_multiple_items:
            return False
    
    # If an expected format is specified, more specific validation can be added here
    if expected_format:
        # More complex validation logic can be added based on expected_format
        pass
    
    return True

@tool  
def ask_follow_up_question(original_question: str, human_response: str, clarification_needed: str, urgency: str = "normal") -> str:
    """
    Ask a follow-up question when the human's initial reply is incomplete or unclear.
    
    Args:
        original_question: Original question
        human_response: Human's initial reply
        clarification_needed: Specific aspect that needs clarification
        urgency: Priority
    
    Returns:
        Human's clarification reply
    """
    global human_tool_instance
    global global_tracker
    
    # Construct follow-up question
    follow_up = f"Follow-up question regarding your previous response:\n\n"
    follow_up += f"Original question: {original_question}\n"
    follow_up += f"Your response: {human_response}\n\n"
    follow_up += f"Clarification needed: {clarification_needed}\n\n"
    follow_up += "Please provide the additional information or clarify your response."
    
    # Add urgency indicator
    urgency_indicator = {
        "low": "🟢",
        "normal": "🟡", 
        "high": "🔴"
    }.get(urgency, "🟡")
    
    formatted_question = f"{urgency_indicator} [{urgency.upper()}] {follow_up}"
    
    # Obtain human reply
    clarification_response = human_tool_instance.ask_human_simple(formatted_question)
    
    # Log follow-up interaction
    global_tracker.add_human_interaction(f"Follow-up: {clarification_needed}", clarification_response, urgency)
    
    # Return combined reply
    if not any(keyword in clarification_response.lower() for keyword in ['timeout', 'error', 'interrupted']):
        combined_response = f"Original Response: {human_response}\n"
        combined_response += f"Clarification: {clarification_response}\n\n"
        combined_response += "IMPORTANT: Use both the original response and clarification to complete the task."
        
        # Mark that complete human input has been received
        global_tracker.mark_human_input_received(original_question, f"{human_response} + {clarification_response}")
        
        return f"Human Response: {combined_response}"
    else:
        return f"Follow-up Response Error: {clarification_response}"

# Improved reasoning process tracker
class ReasoningTracker:
    def __init__(self):
        self.reasoning_steps = []
        self.tool_calls = []
        self.supervisor_decisions = []
        self.agent_interactions = []
        self.human_interactions = []
        self.max_iterations = 50  # Increased iteration limit
        self.current_iterations = 0
        self.task_completed = False
        self.human_inputs_received = {}  # Track received human inputs
        self.task_progress = {
            "started": False,
            "human_input_needed": False,
            "human_input_received": False,
            "final_processing": False,
            "completed": False
        }

    def add_reasoning_step(self, agent_name: str, step_type: str, content: str, timestamp=None):
        if timestamp is None:
            timestamp = datetime.datetime.now()

        step = {
            "agent": agent_name,
            "type": step_type,
            "content": content,
            "timestamp": timestamp
        }
        self.reasoning_steps.append(step)
        print(f"[{timestamp.strftime('%H:%M:%S')}] {agent_name} - {step_type}: {content}")

    def add_tool_call(self, tool_name: str, input_data: Any, output_data: Any):
        tool_call = {
            "tool": tool_name,
            "input": str(input_data),
            "output": str(output_data),
            "timestamp": datetime.datetime.now()
        }
        self.tool_calls.append(tool_call)
        print(f"[TOOL CALL] {tool_name}: {str(input_data)} -> {str(output_data)[:100]}...")

    def add_human_interaction(self, question: str, response: str, urgency: str = "normal"):
        """Track human interactions"""
        interaction = {
            "question": question,
            "response": response,
            "urgency": urgency,
            "timestamp": datetime.datetime.now()
        }
        self.human_interactions.append(interaction)
        print(f"[HUMAN INTERACTION] {urgency.upper()}: Q: {question[:50]}... A: {response[:50]}...")
        
        # Update task progress
        if not any(keyword in response.lower() for keyword in ['timeout', 'error', 'interrupted']):
            self.task_progress["human_input_received"] = True
            self.add_reasoning_step("human_tool", "successful_interaction", 
                                  f"Successfully obtained human input: {response[:100]}...")

    def mark_human_input_received(self, question: str, response: str):
        """Mark that human input has been received"""
        question_hash = hashlib.md5(question.lower().strip().encode()).hexdigest()[:8]
        self.human_inputs_received[question_hash] = {
            "question": question,
            "response": response,
            "timestamp": datetime.datetime.now()
        }
        self.task_progress["human_input_received"] = True
        
    def has_human_input_for_question(self, question: str) -> bool:
        """Check whether human input for this question already exists"""
        question_hash = hashlib.md5(question.lower().strip().encode()).hexdigest()[:8]
        return question_hash in self.human_inputs_received

    def add_supervisor_decision(self, decision: str, reasoning: str):
        decision_info = {
            "decision": decision,
            "reasoning": reasoning,
            "timestamp": datetime.datetime.now()
        }
        self.supervisor_decisions.append(decision_info)
        print(f"[SUPERVISOR DECISION] {decision}: {reasoning}")

    def add_agent_interaction(self, from_agent: str, to_agent: str, message: str):
        interaction = {
            "from": from_agent,
            "to": to_agent,
            "message": message,
            "timestamp": datetime.datetime.now()
        }
        self.agent_interactions.append(interaction)
        print(f"[INTERACTION] {from_agent} -> {to_agent}: {message[:100]}...")

    def increment_iteration(self):
        """Increment iteration counter"""
        self.current_iterations += 1
        print(f"[ITERATION] {self.current_iterations}/{self.max_iterations}")

    def should_force_finish(self):
        """Check whether completion should be forced"""
        return self.current_iterations >= self.max_iterations

    def mark_task_completed(self):
        """Mark task as completed"""
        self.task_completed = True
        self.task_progress["completed"] = True
        print("[TASK COMPLETED] Task marked as completed")
    
    def should_complete_task(self) -> tuple[bool, str]:
        """
        Improved task completion check logic
        Returns (should complete, completion reason)
        """
        reasons = []
        
        # Check whether enough tool calls have been made
        if len(self.tool_calls) > 0:
            reasons.append(f"{len(self.tool_calls)} tool calls completed")
        
        # Check human input status
        if len(self.human_interactions) > 0:
            successful_interactions = [h for h in self.human_interactions 
                                     if not any(keyword in h['response'].lower() 
                                              for keyword in ['timeout', 'error', 'interrupted'])]
            if successful_interactions:
                reasons.append(f"{len(successful_interactions)} human interactions completed")
        
        # Check task progress
        if self.task_progress["human_input_received"]:
            reasons.append("human input received and processed")
        
        # Check whether there is a final result
        recent_steps = self.reasoning_steps[-5:] if len(self.reasoning_steps) >= 5 else self.reasoning_steps
        has_final_result = any(
            any(indicator in step.get('content', '').lower() 
                for indicator in ['final', 'result', 'completed', 'generated', 'best match'])
            for step in recent_steps
        )
        
        if has_final_result:
            reasons.append("final result generated")
        
        # If there is enough activity and a result, the task should be completed
        should_complete = (
            len(reasons) >= 2 or  # At least two completion indicators
            (len(self.tool_calls) > 0 and has_final_result) or  # Tool calls plus final result
            (self.task_progress["human_input_received"] and len(self.tool_calls) > 0)  # Human input plus tool calls
        )
        
        completion_reason = "; ".join(reasons) if reasons else "no clear completion indicators"
        
        return should_complete, completion_reason

    def get_tool_call_count(self):
        """Return actual tool call count"""
        return len(self.tool_calls)

    def get_human_interaction_count(self):
        """Return human interaction count"""
        return len(self.human_interactions)

    def get_full_reasoning_trace(self):
        return {
            "reasoning_steps": self.reasoning_steps,
            "tool_calls": self.tool_calls,
            "supervisor_decisions": self.supervisor_decisions,
            "agent_interactions": self.agent_interactions,
            "human_interactions": self.human_interactions,
            "human_inputs_received": self.human_inputs_received,
            "task_progress": self.task_progress,
            "tool_call_count": self.get_tool_call_count(),
            "human_interaction_count": self.get_human_interaction_count(),
            "total_iterations": self.current_iterations,
            "task_completed": self.task_completed
        }

    def print_summary(self):
        print("\n" + "=" * 60)
        print("REASONING PROCESS SUMMARY")
        print("=" * 60)

        actual_tool_calls = self.get_tool_call_count()
        human_interactions = self.get_human_interaction_count()
        print(f"\nTotal reasoning steps: {len(self.reasoning_steps)}")
        print(f"Total tool calls: {actual_tool_calls}")
        print(f"Total human interactions: {human_interactions}")
        print(f"Total supervisor decisions: {len(self.supervisor_decisions)}")
        print(f"Total agent interactions: {len(self.agent_interactions)}")
        print(f"Total iterations: {self.current_iterations}")
        print(f"Task completed: {self.task_completed}")
        print(f"Task progress: {self.task_progress}")

        # Human tool summary
        human_summary = human_tool_instance.get_interaction_summary()
        print(f"Human tool summary: {human_summary}")

        if actual_tool_calls > 0:
            print("\n--- TOOL CALLS DETAIL ---")
            for i, call in enumerate(self.tool_calls):
                print(f"{i+1}. {call['tool']}: {call['input'][:50]}... -> {call['output'][:50]}...")

        if human_interactions > 0:
            print("\n--- HUMAN INTERACTIONS DETAIL ---")
            for i, interaction in enumerate(self.human_interactions):
                print(f"{i+1}. [{interaction['urgency'].upper()}] Q: {interaction['question']}")
                print(f"    A: {interaction['response']}")

# Global reasoning tracker
global_tracker = ReasoningTracker()

class State(MessagesState):
    next: str
    reasoning_tracker: Optional[ReasoningTracker] = None
    iteration_count: int = 0
    task_status: str = "in_progress"
    awaiting_human: bool = False
    human_responses: List[str] = []
    last_human_question: str = ""
    task_progress: Dict[str, bool] = {}

# Improved tool-agent wrapper
class ReasoningReactAgent:
    def __init__(self, base_agent, tracker: ReasoningTracker):
        self.base_agent = base_agent
        self.tracker = tracker

    async def ainvoke(self, state):
        """Async invoke method, enhanced tool detection and human-interaction handling"""
        # Check whether forced completion is needed
        if self.tracker.should_force_finish():
            self.tracker.add_reasoning_step("tool_agent", "forced_finish", "Reached maximum iterations, forcing completion")
            return {
                "messages": [HumanMessage(content="Task reached maximum iterations, forcing completion.")]
            }

        # Log tool agent start processing
        user_message = state["messages"][-1].content if state["messages"] else "No message"
        self.tracker.add_reasoning_step("tool_agent", "start_processing", f"Received task: {user_message[:100]}...")

        # Check whether the message contains a human reply
        if "Human Response:" in user_message:
            self.tracker.add_reasoning_step("tool_agent", "human_response_detected", 
                                          "Human response detected in message, processing...")
            # Update task progress
            self.tracker.task_progress["human_input_received"] = True

        try:
            # Invoke the original agent
            result = await self.base_agent.ainvoke(state)

            # Enhanced tool-call detection
            tool_calls_detected = 0
            human_calls_detected = 0
            
            if result and "messages" in result and result["messages"]:
                for msg in result["messages"]:
                    # Check the tool_calls attribute
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_calls_detected += 1
                            tool_name = tool_call.get('name', 'unknown_tool')
                            
                            if tool_name == 'request_human_assistance':
                                human_calls_detected += 1
                                args = tool_call.get('args', {})
                                question = args.get('question', 'No question provided')
                                
                                # Check whether this is a duplicate question
                                if not self.tracker.has_human_input_for_question(question):
                                    self.tracker.add_reasoning_step("tool_agent", "new_human_question", 
                                                                  f"New human question detected: {question}")
                                else:
                                    self.tracker.add_reasoning_step("tool_agent", "duplicate_human_question", 
                                                                  f"Duplicate question detected, should use cached response: {question}")
                            
                            self.tracker.add_tool_call(tool_name, tool_call.get('args', {}), "Tool call detected")

                    # Check tool calls in additional_kwargs
                    if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
                        if 'tool_calls' in msg.additional_kwargs:
                            tool_calls_in_kwargs = msg.additional_kwargs['tool_calls']
                            for tool_call in tool_calls_in_kwargs:
                                tool_calls_detected += 1
                                function_info = tool_call.get('function', {})
                                tool_name = function_info.get('name', 'unknown_tool')
                                
                                if tool_name == 'request_human_assistance':
                                    human_calls_detected += 1
                                    try:
                                        args = json.loads(function_info.get('arguments', '{}'))
                                        question = args.get('question', 'No question provided')
                                        # Same duplicate check
                                        if not self.tracker.has_human_input_for_question(question):
                                            self.tracker.add_reasoning_step("tool_agent", "new_human_question_kwargs", 
                                                                          f"New human question in kwargs: {question}")
                                    except:
                                        pass
                                
                                self.tracker.add_tool_call(tool_name, function_info.get('arguments', {}), "Tool call in kwargs")

                # Final result processing
                final_content = result["messages"][-1].content if result["messages"] else "No content"
                
                # Check whether human reply has been processed
                if "Human Response:" in final_content or "based on this human feedback" in final_content.lower():
                    self.tracker.add_reasoning_step("tool_agent", "human_response_processed",
                                                  "Human response content detected and processed")
                    self.tracker.task_progress["human_input_received"] = True
                
                # Check whether there is a final result
                if any(indicator in final_content.lower() for indicator in 
                       ['final', 'result', 'completed', 'best match', 'generated']):
                    self.tracker.add_reasoning_step("tool_agent", "final_result_detected",
                                                  "Final result indicators detected in response")
                    self.tracker.task_progress["final_processing"] = True

                summary = f"Analysis complete: {tool_calls_detected} total tool calls detected"
                if human_calls_detected > 0:
                    summary += f", {human_calls_detected} human tool calls"
                summary += f". Final result: {final_content[:150]}..."
                
                self.tracker.add_reasoning_step("tool_agent", "analysis_complete", summary)

            return result

        except Exception as e:
            self.tracker.add_reasoning_step("tool_agent", "error", f"Error occurred: {str(e)}")
            import traceback
            self.tracker.add_reasoning_step("tool_agent", "error_trace", f"Full traceback: {traceback.format_exc()}")
            return {
                "messages": [HumanMessage(content=f"Tool execution failed: {str(e)}")]
            }

# Fixed Router class for supervisor
class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: str
    reasoning: str
    confidence: Optional[float]
    completion_check: Optional[str]

# Improved supervisor node
def make_supervisor_node(llm: BaseChatModel, members: list[str], tracker: ReasoningTracker) -> str:
    options = ["FINISH"] + members
    valid_next_actions = members + ["__end__"]

    # Improved system prompt with clearer JSON format requirements
    system_prompt = prompt1 + """You are a task assignment supervisor. Your responsibilities are:
1. Analyze the user's query and current conversation history
2. Decide which agent should be invoked next or if the task should be completed
3. Provide clear reasoning and check if the task is completely finished

Available agents: {members}


""".format(members=members)

    def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
        """Improved LLM router with better completion detection and human awareness"""

        # Increment iteration counter
        tracker.increment_iteration()

        # Check whether the system is currently waiting for human input
        global human_tool_instance
        if human_tool_instance.is_waiting_for_human():
            tracker.add_reasoning_step("supervisor", "waiting_for_human", 
                                     "Currently waiting for human input, pausing supervisor decision")
            return Command(goto="tool_agent", update={"awaiting_human": True})

        # Check whether forced completion is needed
        if tracker.should_force_finish():
            tracker.add_reasoning_step("supervisor", "forced_completion",
                                       f"Reached maximum iterations ({tracker.max_iterations}), forcing task completion")
            tracker.mark_task_completed()
            return Command(goto="__end__", update={"task_status": "force_completed", "awaiting_human": False})

        # Use improved task-completion check
        should_complete, completion_reason = tracker.should_complete_task()
        
        # Log supervisor start analysis
        user_query = state["messages"][0].content if state["messages"] else "No query"
        tracker.add_reasoning_step("supervisor", "analyzing_query",
                                   f"Analyzing user query (iteration {tracker.current_iterations}): {user_query[:100]}...")

        # Analyze conversation history
        recent_messages = state["messages"][-5:] if len(state["messages"]) > 5 else state["messages"]
        conversation_analysis = []
        has_human_response = False
        has_final_result = False
        
        for msg in recent_messages:
            if hasattr(msg, 'content') and msg.content:
                content = str(msg.content)
                conversation_analysis.append(f"Message: {content[:100]}...")
                
                # Check for human replies
                if "Human Response:" in content:
                    has_human_response = True
                    tracker.add_reasoning_step("supervisor", "human_response_found", 
                                             "Human response detected in conversation history")
                
                # Check for final results
                if any(indicator in content.lower() for indicator in 
                       ['final answer', 'best match', 'result:', 'generated', 'completed']):
                    has_final_result = True
                    tracker.add_reasoning_step("supervisor", "final_result_found", 
                                             "Final result indicators found in conversation")

        # Build enhanced message
        status_summary = f"Status check: Should complete = {should_complete} ({completion_reason})"
        status_summary += f", Has human response = {has_human_response}, Has final result = {has_final_result}"
        status_summary += f", Tool calls = {len(tracker.tool_calls)}, Human interactions = {len(tracker.human_interactions)}"
        
        # Create messages for LLM
        messages = [
            HumanMessage(content=f"System context: {system_prompt}"),
            HumanMessage(content=f"Current status: {status_summary}"),
            HumanMessage(content=f"Conversation analysis: {'; '.join(conversation_analysis)}"),
            HumanMessage(content="Based on the above information, provide your routing decision in the required JSON format.")
        ]

        try:
            # Get response from LLM
            response = llm.invoke(messages)
            
            # Extract content from response
            if hasattr(response, 'content'):
                response_content = response.content
            else:
                response_content = str(response)
            
            tracker.add_reasoning_step("supervisor", "raw_llm_response", f"Raw LLM response: {response_content}")
            
            # Try to parse JSON from response
            try:
                # Clean the response content
                response_content = response_content.strip()
                
                # Try to find JSON in the response
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_response = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
                
                # Validate required fields
                if not isinstance(parsed_response, dict) or 'next' not in parsed_response:
                    raise ValueError("Invalid response structure - missing 'next' field")
                    
            except (json.JSONDecodeError, ValueError) as e:
                tracker.add_reasoning_step("supervisor", "json_parse_error", 
                                         f"JSON parsing failed: {str(e)}. Using intelligent fallback.")
                
                # Intelligent fallback logic
                response_lower = response_content.lower()
                
                # Decision logic based on content analysis
                if (should_complete or 
                    "finish" in response_lower or 
                    "complete" in response_lower or
                    (has_human_response and has_final_result)):
                    goto = "FINISH"
                    reasoning = f"Fallback completion decision: {completion_reason}"
                else:
                    goto = "tool_agent"
                    reasoning = "Fallback: continuing with tool_agent"
                
                parsed_response = {
                    "next": goto,
                    "reasoning": f"Fallback parsing due to JSON error: {str(e)}. Original: {response_content[:200]}... | Decision: {reasoning}",
                    "confidence": 0.6,
                    "completion_check": f"Fallback logic applied - should_complete: {should_complete}"
                }
            
            # Extract response fields with validation
            goto = parsed_response.get("next", "tool_agent")
            reasoning = parsed_response.get("reasoning", "No detailed reasoning provided")
            confidence = parsed_response.get("confidence", 0.5)
            completion_check = parsed_response.get("completion_check", "No completion check provided")

            # Validate the next action
            if goto not in valid_next_actions and goto not in options:
                tracker.add_reasoning_step("supervisor", "invalid_next_action", 
                                         f"Invalid next action received: {goto}. Defaulting to tool_agent.")
                goto = "tool_agent"
                reasoning += f" | Note: Invalid action '{goto}' corrected to 'tool_agent'"

            # Log supervisor decision process
            decision_detail = f"{reasoning} | Completion check: {completion_check} | Confidence: {confidence}"
            tracker.add_supervisor_decision(goto, decision_detail)
            
            # Apply decision logic
            if goto == "FINISH" or should_complete:
                tracker.add_reasoning_step("supervisor", "task_completion_decision",
                                           f"Task completion decided - Supervisor choice: {goto}, Auto-complete logic: {should_complete}, Reason: {completion_reason}")
                tracker.mark_task_completed()
                goto = "__end__"
            else:
                # Check whether there is a duplicate human question request
                if goto == "tool_agent" and has_human_response:
                    tracker.add_reasoning_step("supervisor", "processing_human_input",
                                             "Routing to tool_agent to process existing human response")
                elif goto == "tool_agent":
                    tracker.add_reasoning_step("supervisor", "continuing_task",
                                             "Routing to tool_agent to continue task execution")
                
                interaction_msg = f"Forwarding task (iteration {tracker.current_iterations}): {reasoning}"
                tracker.add_agent_interaction("supervisor", goto, interaction_msg)

            return Command(goto=goto, update={
                "next": goto,
                "iteration_count": tracker.current_iterations,
                "task_status": "completed" if goto == "__end__" else "in_progress",
                "awaiting_human": False,
                "task_progress": tracker.task_progress
            })

        except Exception as e:
            tracker.add_reasoning_step("supervisor", "critical_error", f"Critical error in supervisor: {str(e)}")
            # Default to task completion on critical error
            tracker.mark_task_completed()
            return Command(goto="__end__", update={"task_status": "error_completed", "awaiting_human": False})

    return supervisor_node

async def run(query: str, image_path: str = "", file_path: str = ""):
    query = query + " " + image_path + " " + file_path
    # Reset global tracker
    global global_tracker
    global_tracker = ReasoningTracker()
    global_tracker.add_reasoning_step("system", "initialization", f"Starting new query: {query}")

    # Reset human tool
    global human_tool_instance
    human_tool_instance = HumanTool()

    # Use passed system prompt or default prompt
  
    client = MultiServerMCPClient(
        {
            "property": {
                "command": "python",
                "args": [r"tool\property.py"],
                "transport": "stdio",
            },
            "search": {
                "command": "python",
                "args": [r"tool\search.py"],
                "transport": "stdio",
            },
            "rag": {
                "command": "python",
                "args": [r"tool\rag.py"],
                "transport": "stdio",
            },
            "imageanalysis": {
                "command": "python",
                "args": [r"tool\ImageAnalysis.py"],
                "transport": "stdio",
            },
            "converters": {
                "command": "python",
                "args": [r"tool\converters.py"],
                "transport": "stdio",
            },
            "coder": {
                "command": "python",
                "args": [r"tool\coder.py"],
                "transport": "stdio",
            },
            "orbital": {
                "command": "python",
                "args": [r"tool\orbital.py"],
                "transport": "stdio",
            },
            "pce": {
                "command": "python",
                "args": [r"tool\PCE.py"],
                "transport": "stdio",
            },
            "pdfreader": {
                "command": "python",
                "args": [r"tool\pdfreader.py"],
                "transport": "stdio",
            },
        }
    )

    global_tracker.add_reasoning_step("system", "mcp_connection", "Establishing MCP client connections")
    
    # Start MCP client and obtain tools
    try:
        global_tracker.add_reasoning_step("system", "mcp_started", "MCP client started successfully")
        
        # Obtain all MCP tools
        mcp_tools = await client.get_tools()
        global_tracker.add_reasoning_step("system", "mcp_tools_loaded", f"Loaded {len(mcp_tools)} MCP tools")
        
        # List all MCP tools
        for i, tool in enumerate(mcp_tools):
            tool_name = getattr(tool, 'name', str(tool))
            tool_description = getattr(tool, 'description', 'No description')
            global_tracker.add_reasoning_step("system", "mcp_tool_detail", 
                                             f"MCP Tool {i+1}: {tool_name} - {tool_description[:100]}...")
        
        # Merge MCP tools and human tools
        all_tools = list(mcp_tools)  # First copy MCP tool list
        
        # Add human tools to tool list
        human_assistance_tool = request_human_assistance
        follow_up_tool = ask_follow_up_question
        all_tools.extend([human_assistance_tool, follow_up_tool])
        
        global_tracker.add_reasoning_step("system", "tools_merged", 
                                         f"Merged tools: {len(mcp_tools)} MCP tools + 2 human tools = {len(all_tools)} total")
        
        # Verify all tools
        global_tracker.add_reasoning_step("system", "tools_verification", 
                                          f"Final tool list ({len(all_tools)} tools):")
        for i, tool in enumerate(all_tools):
            tool_name = getattr(tool, 'name', str(tool))
            tool_type = "MCP" if i < len(mcp_tools) else "Custom"
            global_tracker.add_reasoning_step("system", "tool_list", f"  {i+1}. [{tool_type}] {tool_name}")
            
            if 'human' in tool_name.lower() or 'follow_up' in tool_name.lower():
                global_tracker.add_reasoning_step("system", "human_tool_confirmed", 
                                                  f"✅ Human interaction tool confirmed: {tool_name}")
        
        # Set tools variable
        tools = all_tools
        
    except Exception as e:
        global_tracker.add_reasoning_step("system", "mcp_error", f"MCP client error: {str(e)}")
        print(f"❌ MCP client failed to start: {e}")
        
        # If MCP fails, at least use human tools
        tools = [request_human_assistance, ask_follow_up_question]
        global_tracker.add_reasoning_step("system", "fallback_tools", 
                                         "Using fallback: only human assistance tools available")
    
    global_tracker.add_reasoning_step("system", "tools_ready", 
                                      f"Tools ready for agent creation: {len(tools)} tools total")

    llm1 = _make_llm(model= "deepseek-v3.1-nothinking", temp=0.1, api_key=os.getenv("OPENAI_API_KEY"), streaming=True)

    # Create tool-agent with reasoning tracking – enhanced debugging
    enhanced_prompt = prompt2 
    
    base_tool_agent = create_react_agent(llm1, tools=tools, prompt=enhanced_prompt)
    reasoning_tool_agent = ReasoningReactAgent(base_tool_agent, global_tracker)

    async def tool_agent_node(state: State) -> Command[Literal["supervisor"]]:
        global_tracker.add_reasoning_step("tool_agent", "node_entry",
                                          f"Tool agent activated (iteration {global_tracker.current_iterations})")

        # Log incoming message
        input_messages = state.get("messages", [])
        if input_messages:
            latest_msg = input_messages[-1]
            content = getattr(latest_msg, 'content', str(latest_msg))
            global_tracker.add_reasoning_step("tool_agent", "input_received", f"Processing: {content[:100]}...")
            
            # Check whether the message contains a human reply that needs processing
            if "Human Response:" in content:
                global_tracker.add_reasoning_step("tool_agent", "human_response_ready", 
                                                 "Human response detected, preparing to process and complete task")

        # Invoke tool agent
        result = await reasoning_tool_agent.ainvoke(state)

        # Analyze result
        final_message = "No result"
        awaiting_human = False
        task_ready_for_completion = False
        
        if result and "messages" in result and result["messages"]:
            final_message = result["messages"][-1].content if result["messages"] else "No content"
            
            # Check whether the system is waiting for human input
            if human_tool_instance.is_waiting_for_human():
                awaiting_human = True
                global_tracker.add_reasoning_step("tool_agent", "human_input_pending", 
                                                "Waiting for human input before proceeding")
            else:
                # Check whether the task can be completed
                if ("Human Response:" in final_message or 
                    global_tracker.task_progress["human_input_received"] or
                    any(indicator in final_message.lower() for indicator in 
                        ['final', 'best match', 'completed', 'result', 'generated'])):
                    task_ready_for_completion = True
                    global_tracker.add_reasoning_step("tool_agent", "task_completion_ready", 
                                                    "Task appears ready for completion")
            
            global_tracker.add_reasoning_step("tool_agent", "final_result", f"Final result: {final_message[:200]}...")

        # Update status information
        next_action = "supervisor"
        status_update = {
            "messages": [HumanMessage(content=final_message, name="tool_agent")],
            "iteration_count": global_tracker.current_iterations,
            "awaiting_human": awaiting_human,
            "task_progress": global_tracker.task_progress
        }
        
        if task_ready_for_completion:
            status_update["task_status"] = "ready_for_completion"
            
        global_tracker.add_agent_interaction("tool_agent", "supervisor", 
                                           f"Returning result (ready for completion: {task_ready_for_completion}): {final_message[:100]}...")

        return Command(
            update=status_update,
            goto=next_action,
        )

    llm2 = _make_llm(model="deepseek-r1-250528", temp=0.1, api_key=os.getenv("OPENAI_API_KEY"), streaming=True)

    supervisor_node = make_supervisor_node(llm2, ['tool_agent'], global_tracker)

    # Build graph
    builder = StateGraph(State)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("tool_agent", tool_agent_node)
    builder.add_edge(START, "supervisor")
    graph = builder.compile()

    global_tracker.add_reasoning_step("system", "graph_execution", "Starting graph execution")

    # Execute graph
    final_result = None
    try:
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content=query)],
                "iteration_count": 0,
                "task_status": "in_progress",
                "awaiting_human": False,
                "human_responses": [],
                "last_human_question": "",
                "task_progress": {
                    "started": True,
                    "human_input_needed": False,
                    "human_input_received": False,
                    "final_processing": False,
                    "completed": False
                }
            },
            {"recursion_limit": global_tracker.max_iterations}
        )
        final_result = result

    except Exception as e:
        print(f"Graph execution failed: {e}")
        global_tracker.add_reasoning_step("system", "execution_error", f"Graph execution error: {str(e)}")
        final_result = {
            "messages": [HumanMessage(content=f"An error occurred during execution: {str(e)}")],
            "task_status": "error"
        }

    global_tracker.add_reasoning_step("system", "execution_complete", "Graph execution completed")

    # Output detailed reasoning process analysis
    global_tracker.print_summary()

    # Save reasoning trace to file
    reasoning_trace = global_tracker.get_full_reasoning_trace()

    return final_result, reasoning_trace

if __name__ == '__main__':
    try:
        print("\n🔄 Starting main execution...")
         
        final_result, trace = asyncio.run(run(''' The history of Y6
''', image_path=r"", file_path=r""))

        print("\n" + "=" * 60)
        print("FINAL EXECUTION RESULT:")
        print("=" * 60)

        # Extract and display the final result
        if final_result:
            if isinstance(final_result, dict) and 'messages' in final_result:
                messages = final_result['messages']
                if messages:
                    print("")
                    for msg in messages:
                        if hasattr(msg, 'content') and msg.content:
                            print(msg.content)

            # Display task status
            task_status = final_result.get('task_status', 'unknown')
            awaiting_human = final_result.get('awaiting_human', False)
            task_progress = final_result.get('task_progress', {})
            
            print(f"\nTask status: {task_status}")
            print(f"Task progress: {task_progress}")
            if awaiting_human:
                print("⚠️  Task was interrupted waiting for human input")
        else:
            print("No final result captured")
    
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        import traceback
        print("Full error traceback:")
        traceback.print_exc()
        if 'global_tracker' in globals():
            global_tracker.add_reasoning_step("system", "error", f"Execution failed: {str(e)}")
            global_tracker.print_summary()