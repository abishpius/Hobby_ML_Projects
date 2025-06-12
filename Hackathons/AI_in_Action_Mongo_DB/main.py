## Imports
# Standard Library
import collections
import os
from typing import List, Tuple, Dict, Callable
import base64
import random
import time

# Third-Party Libraries
import pandas as pd

# MongoDB
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.database import Database

# Google Generative AI
from google.ai.generativelanguage_v1beta.types import Tool as GenAITool

# LangChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import ArxivLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

# LangChain MongoDB
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory

# LangChain Tools and Agents
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL

# Mesop
import mesop as me

## Set Keys
gemini_key = os.environ.get("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = gemini_key
mongo_user = os.environ.get("MONGO_USER")
mongo_pass = os.environ.get("MONGO_PASS")

## Initialize
# Upload to MongoDB
MONGO_URI = f"mongodb+srv://{mongo_user}:{mongo_pass}@cluster0.r3voc34.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Initialize MongoDB python client
client = MongoClient(MONGO_URI, server_api=ServerApi('1'))

# Handle Memory
def get_session_history(session_id: str) -> MongoDBChatMessageHistory:
  return MongoDBChatMessageHistory(MONGO_URI, session_id, database_name="agent_demo", collection_name="demo")

memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=get_session_history("my-session")
)

import random
import time
from typing import Callable

import mesop as me

_TEMPERATURE_MIN = 0.0
_TEMPERATURE_MAX = 2.0
_TOKEN_LIMIT_MIN = 1
_TOKEN_LIMIT_MAX = 8192

# Upload to MongoDB
MONGO_URI = f"mongodb+srv://{mongo_user}:{mongo_pass}@cluster0.r3voc34.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Initialize MongoDB python client
client = MongoClient(MONGO_URI, server_api=ServerApi('1'))

# Handle Memory
def get_session_history(session_id: str) -> MongoDBChatMessageHistory:
  return MongoDBChatMessageHistory(MONGO_URI, session_id, database_name="agent_demo", collection_name="demo")

memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=get_session_history("my-session")
)


@me.stateclass
class State:
  title: str = "Mongo DB Data Science Agent"
  # Prompt / Response
  input: str
  response: str
  # Tab open/close
  prompt_tab: bool = True
  response_tab: bool = True
  upload_file_box: bool = False
  # Model configs
  selected_model: str = "gemini-2.0-flash"
  selected_region: str = "us-east4"
  temperature: float = 1.0
  temperature_for_input: float = 1.0
  token_limit: int = _TOKEN_LIMIT_MAX
  token_limit_for_input: int = _TOKEN_LIMIT_MAX
  stop_sequence: str = ""
  stop_sequences: list[str]
  # Modal
  modal_open: bool = False
  # Workaround for clearing inputs
  clear_prompt_count: int = 0
  clear_sequence_count: int = 0
  upload_file: me.UploadedFile
  data_preprocessed: bool = False
  no_upload_file: bool = True
  agent_purpose: str = ""
  DB_NAME: str = "agent_demo"
  COLLECTION_NAME: str = "demo"



def load(e: me.LoadEvent):
  me.set_theme_mode("system")


@me.page(
  on_load=load,
  security_policy=me.SecurityPolicy(
    allowed_iframe_parents=["https://mesop-dev.github.io"]
  ),
 path="/",
  title="MongoDB Data Science Agent",
)
def app() -> None:
  state = me.state(State)

  # Modal
  with modal(modal_open=state.modal_open):
    me.text("Toogle Theme", type="headline-5")

  # Main content
  with me.box(style=_STYLE_CONTAINER):
    # Main Header
    with me.box(style=_STYLE_MAIN_HEADER):
      with me.box(style=_STYLE_TITLE_BOX):
        me.text(
          state.title,
          type="headline-6",
          style=me.Style(line_height="24px", margin=me.Margin(bottom=0)),
        )

    # Toolbar Header (Now Light/Dark Theme)
    with me.box(style=_STYLE_CONFIG_HEADER):
      icon_button(
        label="Toggle Dark Mode",
        icon="dark_mode" if me.theme_brightness() == "light" else "light_mode",
        tooltip="Dark mode"
        if me.theme_brightness() == "light"
        else "Light mode",
        on_click=on_click_theme_brightness,
      )

    # Main Content
    with me.box(style=_STYLE_MAIN_COLUMN):
      # Upload files Tab
      with tab_box(header="Upload Data File", key="upload_file_box"):
          me.markdown("Upload a **single** *.csv* file containing the data for analysis.")
          me.markdown("**Default**: Demo Data Set of Housing Prices.")

          if state.no_upload_file:
              with me.content_uploader(
                  accepted_file_types=[".csv"],
                  on_upload=handle_upload,
                  type="flat",
                  color="primary",
                  style=me.Style(font_weight="bold")
              ):
                  with me.box(style=me.Style(display="flex", gap=5)):
                      me.icon("upload")
                      me.text("Upload CSV", style=me.Style(line_height="25px"))
          else:
            if state.upload_file.size:
                with me.box(style=me.Style(margin=me.Margin.all(10))):
                    me.text(f"File name: {state.upload_file.name}")
                    me.text(f"File size: {state.upload_file.size} Bytes")
                    me.text(f"File type: {state.upload_file.mime_type}")
                    me.text(f"Please wait for {state.upload_file.name} preprocessing to finish.")

                    if not state.data_preprocessed:
                        preprocess_data_file(state.upload_file)
                        state.data_preprocessed = True  # Prevent reprocessing on reload

                    me.text(f"Data file {state.upload_file.name} ingestion into MongoDB completed.")

                    me.button(label="Change Data", on_click=on_click_change_data)

      # Prompt Tab
      with tab_box(header="Query", key="prompt_tab"):
        me.textarea(
          label="""Write your question here and then click Submit.

          (Upload a .csv above, else demo will use CA Housing Data.)""",
          # Workaround: update key to clear input.
          key=f"prompt-{state.clear_prompt_count}",
          on_input=on_prompt_input,
          style=_STYLE_INPUT_WIDTH,
        )
        me.button(label="Submit", type="flat", on_click=on_click_submit)
        me.button(label="Clear", on_click=on_click_clear)

      # Response Tab
      with tab_box(header="Response", key="response_tab"):
        if state.response:
          me.markdown(state.response)
          if os.path.exists("plot.png"):

            with open("plot.png", "rb") as image_file:
              encoded_image = base64.b64encode(image_file.read()).decode()

            # Construct a data URL
            data_url = f"data:image/png;base64,{encoded_image}"

            # Display the image
            me.image(src=data_url, style=me.Style(width="100%", max_width="600px"))
        else:
          me.markdown(
            "The model will generate a response after you click Submit."
          )

    # LLM Config Sidebar
    with me.box(style=_STYLE_CONFIG_COLUMN):
      me.select(
        options=[
          me.SelectOption(label="Gemini 2.0 Flash", value="gemini-2.0-flash"),
          me.SelectOption(label="Gemini 1.5 Flash", value="gemini-1.5-flash"),
          me.SelectOption(label="Gemini 2.5 Pro-Thinking", value="gemini-2.5-pro-exp-03-25"),
          me.SelectOption(label="Gemma 3 27B", value="gemma-3-27b-it")
        ],
        label="Model",
        style=_STYLE_INPUT_WIDTH,
        on_selection_change=on_model_select,
        value=state.selected_model,
      )

      me.text("Temperature", style=_STYLE_SLIDER_LABEL)
      with me.box(style=_STYLE_SLIDER_INPUT_BOX):
        with me.box(style=_STYLE_SLIDER_WRAP):
          me.slider(
            min=_TEMPERATURE_MIN,
            max=_TEMPERATURE_MAX,
            step=0.1,
            style=_STYLE_SLIDER,
            on_value_change=on_slider_temperature,
            value=state.temperature,
          )
        me.input(
          style=_STYLE_SLIDER_INPUT,
          value=str(state.temperature_for_input),
          on_input=on_input_temperature,
        )

      me.text("Output Token Limit", style=_STYLE_SLIDER_LABEL)
      with me.box(style=_STYLE_SLIDER_INPUT_BOX):
        with me.box(style=_STYLE_SLIDER_WRAP):
          me.slider(
            min=_TOKEN_LIMIT_MIN,
            max=_TOKEN_LIMIT_MAX,
            style=_STYLE_SLIDER,
            on_value_change=on_slider_token_limit,
            value=state.token_limit,
          )
        me.input(
          style=_STYLE_SLIDER_INPUT,
          value=str(state.token_limit_for_input),
          on_input=on_input_token_limit,
        )




# HELPER COMPONENTS
@me.component
def icon_button(*, icon: str, label: str, tooltip: str, on_click: Callable):
  """Icon button with text and tooltip."""
  with me.content_button(on_click=on_click):
    with me.tooltip(message=tooltip):
      with me.box(style=me.Style(display="flex")):
        me.icon(icon=icon)
        me.text(
          label, style=me.Style(line_height="24px", margin=me.Margin(left=5))
        )


@me.content_component
def tab_box(*, header: str, key: str):
  """Collapsible tab box"""
  state = me.state(State)
  tab_open = getattr(state, key)
  with me.box(style=me.Style(width="100%", margin=me.Margin(bottom=20))):
    # Tab Header
    with me.box(
      key=key,
      on_click=on_click_tab_header,
      style=me.Style(padding=_DEFAULT_PADDING, border=_DEFAULT_BORDER),
    ):
      with me.box(style=me.Style(display="flex")):
        me.icon(
          icon="keyboard_arrow_down" if tab_open else "keyboard_arrow_right"
        )
        me.text(
          header,
          style=me.Style(
            line_height="24px", margin=me.Margin(left=5), font_weight="bold"
          ),
        )
    # Tab Content
    with me.box(
      style=me.Style(
        padding=_DEFAULT_PADDING,
        border=_DEFAULT_BORDER,
        display="block" if tab_open else "none",
      )
    ):
      me.slot()


@me.content_component
def modal(modal_open: bool):
  """Basic modal box."""
  with me.box(style=_make_modal_background_style(modal_open)):
    with me.box(style=_STYLE_MODAL_CONTAINER):
      with me.box(style=_STYLE_MODAL_CONTENT):
        me.slot()


# EVENT HANDLERS
def on_click_clear(e: me.ClickEvent):
  """Click event for clearing prompt text."""
  state = me.state(State)
  state.clear_prompt_count += 1
  state.input = ""
  state.response = ""

def on_click_change_data(e: me.ClickEvent):
  """Click event for clearing prompt text."""
  state = me.state(State)
  state.no_upload_file = True
 


def on_prompt_input(e: me.InputEvent):
  """Capture prompt input."""
  state = me.state(State)
  state.input = e.value


def on_model_select(e: me.SelectSelectionChangeEvent):
  """Event to select model."""
  state = me.state(State)
  state.selected_model = e.value


def on_slider_temperature(e: me.SliderValueChangeEvent):
  """Event to adjust temperature slider value."""
  state = me.state(State)
  state.temperature = float(e.value)
  state.temperature_for_input = state.temperature


def on_input_temperature(e: me.InputEvent):
  """Event to adjust temperature slider value by input."""
  state = me.state(State)
  try:
    temperature = float(e.value)
    if _TEMPERATURE_MIN <= temperature <= _TEMPERATURE_MAX:
      state.temperature = temperature
  except ValueError:
    pass


def on_slider_token_limit(e: me.SliderValueChangeEvent):
  """Event to adjust token limit slider value."""
  state = me.state(State)
  state.token_limit = int(e.value)
  state.token_limit_for_input = state.token_limit


def on_input_token_limit(e: me.InputEvent):
  """Event to adjust token limit slider value by input."""
  state = me.state(State)
  try:
    token_limit = int(e.value)
    if _TOKEN_LIMIT_MIN <= token_limit <= _TOKEN_LIMIT_MAX:
      state.token_limit = token_limit
  except ValueError:
    pass

def on_click_tab_header(e: me.ClickEvent):
  """Open and closes tab content."""
  state = me.state(State)
  setattr(state, e.key, not getattr(state, e.key))

def on_click_theme_brightness(e: me.ClickEvent):
  """Toggles dark mode."""
  if me.theme_brightness() == "light":
    me.set_theme_mode("dark")
  else:
    me.set_theme_mode("light")

def on_click_modal(e: me.ClickEvent):
  """Allows modal to be closed."""
  state = me.state(State)
  if state.modal_open:
    state.modal_open = False

def on_click_submit(e: me.ClickEvent):
  """Submits prompt to test model configuration.

  This example returns canned text. A real implementation
  would call APIs against the given configuration.
  """
  if os.path.exists("data.csv"):
    os.remove("data.csv")
  if os.path.exists("plot.png"):
    os.remove("plot.png")
  state = me.state(State)
  res = transform(state.input)
  state.response = res['output']
  # for line in transform(state.input):
  #   state.response += line
  #   yield

def handle_upload(event: me.UploadEvent):
  state = me.state(State)
  state.upload_file = event.file
  state.data_preprocessed = False
  if state.no_upload_file:
    state.no_upload_file = False
  else:
    state.no_upload_file = True


## FUNCTIONALITY

def preprocess_data_file(file_path):
  """Preprocess data file."""
  state = me.state(State)

  data = pd.read_csv(file_path)

  MONGO_URI = f"mongodb+srv://{mongo_user}:{mongo_pass}@cluster0.r3voc34.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

  # Initialize MongoDB python client
  client = MongoClient(MONGO_URI, server_api=ServerApi('1'))

  DB_NAME = "agent_demo"
  COLLECTION_NAME = "demo"
  ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
  collection = client.get_database(DB_NAME).get_collection(COLLECTION_NAME)
  # Delete any existing records in the collection
  collection.delete_many({})

  # Data Ingestion
  records = data.to_dict('records')
  collection.insert_many(records)

  state.data_preprocessed = True
  state.no_upload_file = False
  state.DB_NAME = DB_NAME
  state.COLLECTION_NAME = COLLECTION_NAME

  # Create Prompt
  schema = describe_table_initialize(db_name = state.DB_NAME, collection_name = state.COLLECTION_NAME)
  schema_description = "\n".join(f"{col:<20} {dtype}" for col, dtype in schema)

  state.agent_purpose = f"""
  You are a data analyst working off a MongoDB database with collection_name = {COLLECTION_NAME},
  you will answer user queries and provide graphs and figures as needed.

  The collection schema is as follows, infer relevant column names as close as possible to input user queries:
  {schema_description}

  Use the code execution tool as necessary to answer complex queries and/or generate graphs and figures.
  When asked to generate a graph or figure use matplotlib and save the figure to a file `plot.png`.

  General best practice tips to follow:
  - Keep units where relevant
  - Round to 3 digits
  """



def transform(input: str):
  """Transform function that returns agent responses."""
  state = me.state(State)

  # Initialize LLM
  llm = ChatGoogleGenerativeAI(model=state.selected_model)

  # Mongo DB relevant custom tools

  @tool
  def list_tables() -> List[str]:
      """Retrieve the names of all collections in the MongoDB database."""
      print(' - DB CALL: list_tables()')

      tables = client.get_database(state.DB_NAME).list_collection_names()
      return tables

  @tool
  def describe_table(collection_name: str = state.COLLECTION_NAME) -> List[Tuple[str, str]]:
      """Infer the schema of a MongoDB collection by sampling documents.

      Args:
          collection_name: Name of the collection to inspect.

      Returns:
          A list of (field, inferred_type) tuples.
      """
      print(f' - DB CALL: describe_table({collection_name})')

      collection = client.get_database(state.DB_NAME)[collection_name]
      sample_docs = collection.find().limit(10)

      field_types = collections.defaultdict(set)

      for doc in sample_docs:
          for key, value in doc.items():
              field_types[key].add(type(value).__name__)

      # Convert sets to comma-separated strings if multiple types are found
      inferred_schema = [(field, ", ".join(sorted(types))) for field, types in field_types.items()]
      return inferred_schema

  @tool
  def execute_query(collection_name: str, query: Dict = {}, projection: Dict = None) -> List[Dict]:
      """Execute a MongoDB query, save results to CSV that is accessible in the environment called data.csv. Returns the first 5 rows of the result as a DataFrame.

      Args:
          collection_name: Name of the collection to query.
          query: The MongoDB query filter (like SQL WHERE clause).
          projection: Fields to include or exclude (like SELECT columns).

      Returns:
          A list of result documents (as dictionaries).

      Example Use:
      # Retrieves documents from the 'knowledge' collection where 'housing_median_age' > 30, returning only 'housing_median_age' and 'median_income' fields.
      execute_query(
      collection_name="knowledge",
      query={"housing_median_age": {"$gt": 30}},
      projection={"housing_median_age": 1, "median_income": 1}
      )

      """
      print(f' - DB CALL: execute_query(query={query})')

      collection = client.get_database(state.DB_NAME)[collection_name]
      cursor = collection.find(query, projection)

      results = list(cursor)
      df = pd.DataFrame(results)
      df.to_csv("data.csv", index=False)

      return df.head()

  # Coding Agent Toolkit
  python_repl = PythonREPL()
  repl_tool = Tool(
      name="python_repl",
      description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
      func=python_repl.run,
  )

  # Create the Toolkit
  db_tools = [list_tables, describe_table, execute_query, repl_tool]


  if state.no_upload_file:
    schema = describe_table.invoke({'collection_name' :'demo'})
    schema_description = "\n".join(f"{col:<20} {dtype}" for col, dtype in schema)
    state.agent_purpose = f"""
  You are a data analyst working off a MongoDB database with collection_name = demo,
  you will answer user queries and provide graphs and figures as needed.

  The collection schema is as follows, infer relevant column names as close as possible to input user queries:
  {schema_description}

  Use the code execution tool as necessary to answer complex queries and/or generate graphs and figures.
  When asked to generate a graph or figure use matplotlib and save the figure to a file `plot.png`.

  General best practice tips to follow:
  - Keep units where relevant
  - Round to 3 digits
  """

  agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", state.agent_purpose),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ]
)
  agent = create_tool_calling_agent(llm, db_tools, agent_prompt)
  agent_executor = AgentExecutor(
    agent=agent,
    tools=db_tools,
    verbose=True, # True for now to demo/debug thinking steps
    handle_parsing_errors=True,
    memory=memory,
)
  res = agent_executor.invoke({"input": input})

  return res

# HELPERS
def describe_table_initialize(db_name: str, collection_name: str) -> List[Tuple[str, str]]:
    """Infer the schema of a MongoDB collection by sampling documents.

    Args:
        collection_name: Name of the collection to inspect.

    Returns:
        A list of (field, inferred_type) tuples.
    """
    print(f' - DB CALL: describe_table({collection_name})')

    collection = client.get_database(db_name)[collection_name]
    sample_docs = collection.find().limit(10)

    field_types = collections.defaultdict(set)

    for doc in sample_docs:
        for key, value in doc.items():
            field_types[key].add(type(value).__name__)

    # Convert sets to comma-separated strings if multiple types are found
    inferred_schema = [(field, ", ".join(sorted(types))) for field, types in field_types.items()]
    return inferred_schema

# STYLES

def _make_modal_background_style(modal_open: bool) -> me.Style:
  """Makes style for modal background.

  Args:
    modal_open: Whether the modal is open.
  """
  return me.Style(
    display="block" if modal_open else "none",
    position="fixed",
    z_index=1000,
    width="100%",
    height="100%",
    overflow_x="auto",
    overflow_y="auto",
    background="rgba(0,0,0,0.4)",
  )


_DEFAULT_PADDING = me.Padding.all(15)
_DEFAULT_BORDER = me.Border.all(
  me.BorderSide(color=me.theme_var("outline-variant"), width=1, style="solid")
)

_STYLE_INPUT_WIDTH = me.Style(width="100%")
_STYLE_SLIDER_INPUT_BOX = me.Style(display="flex", flex_wrap="wrap")
_STYLE_SLIDER_WRAP = me.Style(flex_grow=1)
_STYLE_SLIDER_LABEL = me.Style(padding=me.Padding(bottom=10))
_STYLE_SLIDER = me.Style(width="90%")
_STYLE_SLIDER_INPUT = me.Style(width=75)

_STYLE_STOP_SEQUENCE_BOX = me.Style(display="flex")
_STYLE_STOP_SEQUENCE_WRAP = me.Style(flex_grow=1)

_STYLE_CONTAINER = me.Style(
  display="grid",
  grid_template_columns="5fr 2fr",
  grid_template_rows="auto 5fr",
  height="100vh",
)

_STYLE_MAIN_HEADER = me.Style(
  border=_DEFAULT_BORDER, padding=me.Padding.all(15)
)

_STYLE_MAIN_COLUMN = me.Style(
  border=_DEFAULT_BORDER,
  padding=me.Padding.all(15),
  overflow_y="scroll",
)

_STYLE_CONFIG_COLUMN = me.Style(
  border=_DEFAULT_BORDER,
  padding=me.Padding.all(15),
  overflow_y="scroll",
)

_STYLE_TITLE_BOX = me.Style(display="inline-block")

_STYLE_CONFIG_HEADER = me.Style(
  border=_DEFAULT_BORDER, padding=me.Padding.all(10)
)

_STYLE_STOP_SEQUENCE_CHIP = me.Style(margin=me.Margin.all(3))

_STYLE_MODAL_CONTAINER = me.Style(
  background=me.theme_var("surface-container-high"),
  margin=me.Margin.symmetric(vertical="0", horizontal="auto"),
  width="min(1024px, 100%)",
  box_sizing="content-box",
  height="100vh",
  overflow_y="scroll",
  box_shadow=("0 3px 1px -2px #0003, 0 2px 2px #00000024, 0 1px 5px #0000001f"),
)

_STYLE_MODAL_CONTENT = me.Style(margin=me.Margin.all(30))

_STYLE_CODE_BOX = me.Style(
  font_size=13,
  margin=me.Margin.symmetric(vertical=10, horizontal=0),
)
