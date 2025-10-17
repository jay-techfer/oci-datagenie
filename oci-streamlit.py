# # coding: utf-8
# # Copyright (c) 2023, Oracle and/or its affiliates.  All rights reserved.
# # This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.

# ##########################################################################
# # chat_demo.py
# # Supports Python 3
# ##########################################################################
# # Info:
# # Get texts from LLM model for given prompts using OCI Generative AI Service.
# ##########################################################################
# # Application Command line(no parameter needed)
# # python chat_demo.py
# ##########################################################################
# import oci

# # Setup basic variables
# # Auth Config
# # TODO: Please update config profile name and use the compartmentId that has policies grant permissions for using Generative AI Service
# compartment_id = "ocid1.tenancy.oc1..aaaaaaaa7egpyu6f5rtjhu3cyj4dgduzv4cqr634ph2oftrsd4co3gqp4sjq"
# CONFIG_PROFILE = "NUVAMA"
# config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)

# # Service endpoint
# endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

# generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(config=config, service_endpoint=endpoint, retry_strategy=oci.retry.NoneRetryStrategy(), timeout=(10,240))
# chat_detail = oci.generative_ai_inference.models.ChatDetails()

# chat_request = oci.generative_ai_inference.models.CohereChatRequest()
# chat_request.message = "Describe OCI Generative AI Service"
# chat_request.max_tokens = 600
# chat_request.temperature = 1

# chat_request.frequency_penalty = 0

# chat_request.top_p = 0.75

# chat_request.top_k = 0


# chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
#     model_id="cohere.command-r-plus-08-2024"
#     )
# chat_detail.chat_request = chat_request
# chat_detail.compartment_id = compartment_id

# chat_response = generative_ai_inference_client.chat(chat_detail)

# # Print result
# print("**************************Chat Result**************************")
# #print(vars(chat_response))

# # Extract the question and response
# question = chat_response.data.chat_response.chat_history[0].message
# response = chat_response.data.chat_response.text
# print("Question:", question)
# print("\nResponse:", response)


# Follow driver installation and setup instructions here:
# https://www.oracle.com/database/technologies/appdev/python/quickstartpython.html

import oracledb
import pandas as pd
import oci
import streamlit as st
import time
import json
import os
import time
from datetime import datetime, date
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import ChatDetails, OnDemandServingMode, CohereChatRequest

DB_USER = "DEMOUSER"
DB_PASSWORD = "Nuvm@db2025!"

CONNECT_STRING = """(description= (retry_count=20)(retry_delay=3)(address=(protocol=tcps)(port=1522)(host=adb.ap-mumbai-1.oraclecloud.com))(connect_data=(service_name=gffbf4347d32fde_nuvmadb23ai_high.adb.oraclecloud.com))(security=(ssl_server_dn_match=yes)))"""
# CONFIG_PROFILE = "NUVAMA"  # OCI config profile from ~/.oci/config
COMPARTMENT_ID = "ocid1.compartment.oc1..aaaaaaaalsc7muakqzjbzflbgrywn2s62nwmhayworeml36iujkcim3jitca"
ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
MODEL_ID = "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyapnibwg42qjhwaxrlqfpreueirtwghiwvv2whsnwmnlva"

CONFIG_PROFILE = "DEFAULT"
config_path = "/home/opc/Oracle_Nuvama_UseCase/.oci.config"
config = oci.config.from_file(
    file_location=config_path, profile_name=CONFIG_PROFILE)

st.set_page_config("DataGenie", layout="centered",
                   initial_sidebar_state="expanded")

st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        max-width: 1000px;
        min-width: 500px;
        overflow-x: auto;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-right: 1rem;
    }

    .canvas-box {
        background-color: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 0 12px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
        min-height: 400px;
    }
    </style>
""", unsafe_allow_html=True)

client = GenerativeAiInferenceClient(config=config, service_endpoint=ENDPOINT)

chat_file = f"chat_history_{date.today()}.json"


def load_chat_history():
    if os.path.exists(chat_file):
        with open(chat_file, "r") as f:
            return json.load(f)
        # return {}


def save_chat_history(history):
    with open(chat_file, "w") as f:
        json.dump(history, f, indent=4)


# Load existing history
history = load_chat_history()


def fetch_schema():
    try:
        # Create connection pool
        pool = oracledb.create_pool(
            user=DB_USER,
            password=DB_PASSWORD,
            dsn=CONNECT_STRING,
            # min=1,
            # max=5,
            # increment=1
        )

        # Acquire connection from pool
        with pool.acquire() as connection:
            with connection.cursor() as cursor:
                # Fetch all tables and their columns
                query = """
                SELECT table_name, column_name, data_type
                FROM user_tab_columns
                WHERE table_name IN ('HOTEL_MAPPING','SCRAPED_DATA')
                ORDER BY table_name, column_name
                """
                cursor.execute(query)
                rows = cursor.fetchall()
                connection.commit()
                print("rows", rows)
                # Convert to DataFrame
                df = pd.DataFrame(
                    rows, columns=["TABLE_NAME", "COLUMN_NAME", "DATA_TYPE"])
                print("df.........", df)

                return df

    except oracledb.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()


def execute_generated_sql(generated_sql: str):
    try:
        pool = oracledb.create_pool(
            user=DB_USER,
            password=DB_PASSWORD,
            dsn=CONNECT_STRING,
            min=1,
            max=5,
            increment=1
        )

        with pool.acquire() as connection:
            with connection.cursor() as cursor:
                cursor.execute(generated_sql)
                columns = [col[0] for col in cursor.description]
                rows = cursor.fetchall()
                df = pd.DataFrame(rows, columns=columns)
                return df

    except oracledb.Error as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()


def detect_mode(user_text: str) -> str:
    global COMPARTMENT_ID
    classification_prompt = f"""
You are an intent classifier.
The user said: "{user_text}"
Detects the intent of the user input and classifies it into one of the following:
Classify the intent into one of the following categories:

- "Query": When the user asks for raw data, SQL, data fetching, tables, aggregations, or database queries.
- "Descriptive": When the user wants summaries, facts, trends, or general descriptions about what the data shows.
- "Diagnostic": When the user wants explanations or reasons behind data trends or results.
- "Predictive": When the user asks for forecasts or predictions based on data.
- "Prescriptive": When the user wants actionable suggestions, decisions, or recommendations based on the data
DO NOT GIVE PYTHON CODE.
Output ONLY one of the following words: Query, Descriptive, Diagnostic, Predictive, Prescriptive.
    """

    # Prepare GenAI request
    chat_request = CohereChatRequest(
        message=classification_prompt,
        max_tokens=50,
        temperature=0.0,  # deterministic output
    )

    chat_detail = ChatDetails(
        compartment_id=COMPARTMENT_ID,
        serving_mode=OnDemandServingMode(model_id=MODEL_ID),
        chat_request=chat_request
    )

    # Call OCI GenAI
    response = client.chat(chat_detail)

    # Extract text from response
    if hasattr(response.data, "chat_response") and hasattr(response.data.chat_response, "text"):
        return response.data.chat_response.text.strip()

    # fallback: inspect chat_history
    if hasattr(response.data, "chat_response") and hasattr(response.data.chat_response, "chat_history"):
        for item in response.data.chat_response.chat_history:
            if item.get("role") == "CHATBOT" and "message" in item:
                return item["message"].replace("```", "").strip()

    raise ValueError("Could not extract intent from OCI GenAI response")


def build_chat_context():
    conversation = []
    for msg in st.session_state.chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation.append(f"{role}: {msg['message']}")
    return "\n".join(conversation)


def ask_oci_genai_for_sql(user_question: str, schema_text: str) -> str:
    global COMPARTMENT_ID

    # ✅ Include chat history context
    history_text = build_chat_context()

    system_prompt = (
        "You are an expert Oracle SQL developer. "
        "Use the previous conversation context to maintain continuity. "
        "Given a database schema, history chat and a user's question, write the best possible SQL query. "
        "Output only the SQL query — do not include explanations or text."

    )

    full_prompt = f"""
{system_prompt}

CONVERSATION HISTORY:
{history_text}

SCHEMA:
{schema_text}

USER QUESTION:
{user_question}
"""

    chat_request = CohereChatRequest(
        message=full_prompt,
        max_tokens=500,
        temperature=0.3,
    )

    chat_detail = ChatDetails(
        compartment_id=COMPARTMENT_ID,
        serving_mode=OnDemandServingMode(model_id=MODEL_ID),
        chat_request=chat_request,
    )

    response = client.chat(chat_detail)

    # ✅ Store the new interaction into history
    st.session_state.chat_history.append(
        {"role": "user", "message": user_question})
    if hasattr(response.data, "chat_response") and hasattr(response.data.chat_response, "text"):
        sql = response.data.chat_response.text
        sql = sql.replace("```sql", "").replace("```", "").strip()
        st.session_state.chat_history.append(
            {"role": "assistant", "message": sql})
        return sql

    raise ValueError("Could not extract SQL from OCI GenAI response")


def analyze_data_with_genai(df: pd.DataFrame, user_question: str, mode: str) -> str:
    """
    Uses OCI GenAI to analyze existing DataFrame based on user question and mode.
    Mode: Descriptive, Diagnostic, Predictive, Prescriptive
    """
    global COMPARTMENT_ID

    # Convert DataFrame to readable format
    data_sample = df.to_markdown(index=False)

    system_prompt = f"""
You are a highly skilled Data Analyst.
You are provided with a dataset and a user's question.

Mode: {mode}

Guidelines:
- If mode is 'Descriptive': summarize what the data shows (e.g., key trends, averages, patterns).
- If mode is 'Diagnostic': explain why certain trends or anomalies exist.
- If mode is 'Predictive': make logical predictions based on the data.
- If mode is 'Prescriptive': suggest data-driven recommendations or next actions.
- Respond in a concise, human-readable format (no code unless asked).
"""

    full_prompt = f"""
{system_prompt}

DATA
{data_sample}

USER QUESTION:
{user_question}
"""

    chat_request = CohereChatRequest(
        message=full_prompt,
        max_tokens=600,
        temperature=0.5,
    )

    chat_detail = ChatDetails(
        compartment_id=COMPARTMENT_ID,
        serving_mode=OnDemandServingMode(model_id=MODEL_ID),
        chat_request=chat_request,
    )

    response = client.chat(chat_detail)

    # Extract text
    if hasattr(response.data, "chat_response") and hasattr(response.data.chat_response, "text"):
        return response.data.chat_response.text.strip()

    raise ValueError("Could not extract response from OCI GenAI")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_result_df" not in st.session_state:
    st.session_state.query_result_df = pd.DataFrame()
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_query_columns" not in st.session_state:
    st.session_state.last_query_columns = []

with st.sidebar:
    st.markdown("""
    <h1 style='font-size: 40px; color: #2C3E50; margin-bottom: 10px;'>DataGenie</h1>
    """, unsafe_allow_html=True)

    if not st.session_state.query_result_df.empty:
        new_df1 = st.session_state.query_result_df
        new_df1.reset_index(drop=True, inplace=True)
        new_df1.index = new_df1.index + 1

        # === Initialize States ===
        if "active_tab" not in st.session_state:
            st.session_state.active_tab = "data"
        if "chart_metadata" not in st.session_state:
            st.session_state["chart_metadata"] = []

        # === Tab Buttons ===
        colA, colB = st.columns([1, 4])
        with colA:
            if st.button("📊 Data"):
                st.session_state.active_tab = "data"
        with colB:
            if st.button("📈 Visualize"):
                st.session_state.active_tab = "viz"

        # === Data Tab ===
        if st.session_state.active_tab == "data":
            st.dataframe(new_df1)

        # === Visualization Tab ===
        if st.session_state.active_tab == "viz":

            if "last_df_shape" not in st.session_state or st.session_state["last_df_shape"] != new_df1.shape:
                st.session_state.pop("generated_chart_code", None)
                st.session_state["last_df_shape"] = new_df1.shape

            x_axis_cols = st.multiselect(
                "📌 Select X-axis columns",
                new_df1.columns.tolist(),
                default=st.session_state.get("x_axis_cols", [])
            )
            y_axis_cols = st.multiselect(
                "📌 Select Y-axis columns",
                new_df1.columns.tolist(),
                default=st.session_state.get("y_axis_cols", [])
            )

            chart_prompt = st.text_area(
                "📝 Describe the chart you want to generate",
                value=st.session_state.get("chart_prompt", "")
            )

            st.subheader("📈 Gemini Chart Canvas")

            if st.button("🎨 Create Chart"):
                if not x_axis_cols or not y_axis_cols or not chart_prompt:
                    st.warning(
                        "Select X & Y columns and enter chart description.")
                else:
                    st.session_state["x_axis_cols"] = x_axis_cols
                    st.session_state["y_axis_cols"] = y_axis_cols
                    st.session_state["chart_prompt"] = chart_prompt

                    x_list = ", ".join(x_axis_cols)
                    y_list = ", ".join(y_axis_cols)

                    chart_gen_prompt = f"""
                        You are a Python data visualization assistant.

                        The user wants a chart based on this request: {chart_prompt}

                        Selected columns from the DataFrame named `df`:
                        - X-axis: {x_list}
                        - Y-axis: {y_list}

                        Instructions:
                        - Use the existing DataFrame `df` as-is. Do not create or redefine `df` or generate any mock/sample data.
                        - Use Plotly Express or Plotly Graph Objects.
                        - If widgets are selected, integrate them.
                        - Before plotting, drop any rows where required columns (like X, Y, hierarchy path, or value columns) are null, NaN, or blank strings ('').
                        - Output only the Python code inside a markdown code block.
                        """

                    response = model.generate_content(chart_gen_prompt).text
                    print("chart code : ", response)
                    chart_code = re.search(
                        r"```python(.*?)```", response, re.DOTALL)

                    if chart_code:
                        st.session_state["generated_chart_code"] = chart_code.group(
                            1).strip()
                    else:
                        st.error("⚠️ Couldn't parse chart code.")

            # === Create & Store Charts ===
            if "generated_chart_code" in st.session_state:
                try:
                    exec_globals = {"pd": pd, "df": new_df1,
                                    "px": px, "go": go, "np": np}
                    exec(
                        st.session_state["generated_chart_code"], exec_globals)

                    new_figs = [
                        exec_globals[name]
                        for name in exec_globals
                        if re.match(r"fig\d*$", name) and isinstance(exec_globals[name], go.Figure)
                    ]

                    if new_figs:
                        for fig in new_figs:
                            # ✅ Add this to render new chart
                            st.plotly_chart(fig, use_container_width=True)

                        st.session_state["chart_metadata"].append({
                            "code": st.session_state["generated_chart_code"],
                            "x_cols": x_axis_cols,
                            "y_cols": y_axis_cols
                        })

                    st.session_state.pop("generated_chart_code", None)

                except Exception as e:
                    st.error("❌ Chart rendering failed.")
                    st.exception(e)

            if st.session_state["chart_metadata"]:
                st.subheader("📊 Created Charts")

                # Column Filters
                df = st.session_state.query_result_df
                filter_cols = st.multiselect(
                    "Select columns to filter", df.columns.tolist()
                )

                filters = {}
                for col in filter_cols:
                    unique_vals = sorted(df[col].dropna().unique())
                    selected_vals = st.multiselect(
                        f"Filter {col}", unique_vals, default=unique_vals
                    )
                    filters[col] = selected_vals

                # Apply filters
                filtered_df = df.copy()
                for col, vals in filters.items():
                    filtered_df = filtered_df[filtered_df[col].isin(vals)]

                # Display last 6 charts using filtered_df as df
                grid_cols = st.columns(3)
                # Directly loop over the last 6 chart entries with their true indices
                for display_i, chart_index in enumerate(range(max(0, len(st.session_state["chart_metadata"]) - 6), len(st.session_state["chart_metadata"]))):
                    meta = st.session_state["chart_metadata"][chart_index]
                    exec_globals = {"pd": pd, "df": filtered_df,
                                    "px": px, "go": go, "np": np}
                    exec(meta["code"], exec_globals)
                    fig = next(v for v in exec_globals.values()
                               if isinstance(v, go.Figure))

                    with grid_cols[display_i % 3]:
                        delete_key = f"delete_chart_{chart_index}"
                        if st.button("❌", key=delete_key):
                            st.session_state["chart_metadata"].pop(chart_index)
                            st.rerun()

                        st.plotly_chart(fig, use_container_width=True,
                                        key=f"chart_{display_i}")
            else:
                st.info(
                    "No chart generated yet. Use the controls above to create one.")

        else:
            st.info("Please request data to generate chart!!")


# Fetch schema once
if "schema_text" not in st.session_state:
    schema_df = fetch_schema()
    print(schema_df)
    st.session_state.schema_text = schema_df.to_string(index=False)
    msg = st.success("✅ Schema fetched from ADB.")
    time.sleep(2)
    msg.empty()

st.title("NUVAMA chat ✅")

# Chat input
user_input = st.chat_input("Ask your question here:")

if user_input:
    mode = detect_mode(user_input)

    if mode == "Query":
        sql_query = ask_oci_genai_for_sql(
            user_input, st.session_state.schema_text)
        result_df = execute_generated_sql(sql_query)
        st.session_state.query_result_df = result_df

        # ✅ Append chat messages using 'message'
        st.session_state.chat_history.append(
            {"role": "user", "message": user_input})

        if result_df.empty:
            st.session_state.chat_history.append(
                {"role": "assistant", "message": "No data found as requested."})
        else:
            st.session_state.chat_history.append(
                {"role": "assistant", "message": "✅ Data extracted successfully."})
            # Optionally show small preview of results
            st.session_state.chat_history.append(
                {"role": "assistant", "message": result_df.head(
                    5).to_markdown(index=False)}
            )

    else:
        if st.session_state.query_result_df.empty:
            st.session_state.chat_history.append(
                {"role": "user", "message": user_input})
            st.session_state.chat_history.append(
                {"role": "assistant", "message": "Please ask for data first!"})
        else:
            ai_analysis = analyze_data_with_genai(
                st.session_state.query_result_df, user_input, mode)
            st.session_state.chat_history.append(
                {"role": "user", "message": user_input})
            st.session_state.chat_history.append(
                {"role": "assistant", "message": ai_analysis})

    st.rerun()


# ✅ Render chat history
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.chat_message("user").write(chat["message"])
    elif chat["role"] == "assistant":
        st.chat_message("assistant").write(chat["message"])
