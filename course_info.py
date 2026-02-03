from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import interrupt, Command, RetryPolicy
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Optional
from dotenv import load_dotenv
import json

load_dotenv()

llm = ChatOpenAI(model="gpt-5-nano")

class output_schema(TypedDict):
    topics: List[str]
    difficulty: int

class course_state(TypedDict, total=False):
    course_name: Optional[str]
    course_code: Optional[str]
    school: Optional[str]
    topics: Optional[List[str]]
    difficulty: Optional[int]


def get_course_info(state: course_state):

    course_info_llm = llm.with_structured_output(output_schema)

    course_info_prompt = f"""
        You are a professional tutor and your client is taking {state['course_code']}  –  {state['course_name']} at {state['school']}.
        Figure out the core topics of this course, and the expected difficulty. 
        Return only a list of the topics, and the difficulty on an integer scale of 1-10.
    """

    course_info: output_schema = course_info_llm.invoke(course_info_prompt)

    return Command(
        update=course_info
    )

def build_state(state: course_state):

    info = interrupt({
        "instruction": "Review and edit this content",
        "name": "",
        "code": "",
        "school": ""
    })

    return info




builder = StateGraph(course_state)

builder.add_node("build state", build_state)
builder.add_node("get course info", get_course_info)

builder.add_edge(START, "build state")
builder.add_edge("build state", "get course info")
builder.add_edge("get course info", END)

checkpoint = MemorySaver()
graph = builder.compile(checkpointer=checkpoint)

config = {"configurable": {"thread_id": "review-42"}}


empty: course_state = {}
first = graph.invoke(input=empty, config=config )
print(first["__interrupt__"])

course = input("course > ")
code = input("code   > ")
school = input("school > ")

res = {
    "course_name": course,
    "course_code": code,
    "school": school
}

next = graph.invoke(Command(resume = res), config=config)

print("=======================")
print(f"Course: {next["course_code"]} – {next["course_name"]}")
print(f"Taken at: {next["school"]}")
print("Topics: ")
for t in next["topics"]:
    print(t)
print(f"Difficulty (1-10): {next["difficulty"]}")