
import oracledb
import pandas as pd
import oci
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
import time
import json
import os
import time
from datetime import datetime, date
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import ChatDetails, OnDemandServingMode, CohereChatRequest
import secrets
from cryptography.fernet import Fernet

DB_USER = "DEMOUSER"
DB_PASSWORD = "Nuvm@db2025!"

CONNECT_STRING = """(description= (retry_count=20)(retry_delay=3)(address=(protocol=tcps)(port=1522)(host=adb.ap-mumbai-1.oraclecloud.com))(connect_data=(service_name=gffbf4347d32fde_nuvmadb23ai_high.adb.oraclecloud.com))(security=(ssl_server_dn_match=yes)))"""
# CONFIG_PROFILE = "NUVAMA"  # OCI config profile from ~/.oci/config
COMPARTMENT_ID = "ocid1.compartment.oc1..aaaaaaaalsc7muakqzjbzflbgrywn2s62nwmhayworeml36iujkcim3jitca"
ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
MODEL_ID = "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyapnibwg42qjhwaxrlqfpreueirtwghiwvv2whsnwmnlva"

CONFIG_PROFILE = "DEFAULT"
config_path = "/home/opc/.oci/config"
config = oci.config.from_file(
    file_location=config_path, profile_name=CONFIG_PROFILE)

st.set_page_config("23AI chat", layout="centered",
                   initial_sidebar_state="expanded")


# Initialize encryption (use your actual key)
cipher_suite = Fernet(Fernet.generate_key())

# Hardcoded credentials
USER_CREDENTIALS = {
    "admin": "admin123",
    "jay": "jay2025",
    "DEMOUSER": "Nuvm@db2025!"
}

def login_page():
    col1, col2, col3 = st.columns([3, 3, 3])
    with col1:
        st.image("techfer_logo_new.png", width=150)
    with col2:
        st.header("👨‍💻 Login ")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("🔑 Login"):
            if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
                st.success(f"✅ Welcome, {username}!")

                # Generate session token
                token = secrets.token_hex(16)
                data = json.dumps({"username": username, "token": token}).encode()
                encrypted_data = cipher_suite.encrypt(data).decode()

                # Update session state
                st.session_state.update({
                    "username": username,
                    "authenticated": True,
                    "last_activity": time.time(),
                    "encrypted_token": encrypted_data,
                    "page": "landing"
                })

                st.rerun()
            else:
                st.error("❌ Invalid username or password")

def landing_page():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "query_result_df" not in st.session_state:
        st.session_state.query_result_df = pd.DataFrame()
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "last_query_columns" not in st.session_state:
        st.session_state.last_query_columns = []
    if "username" not in st.session_state:
        st.session_state.username = "guest"
    if "history" not in st.session_state:
        st.session_state.history = {}
        
    m_p = st.empty()
    timeout_seconds = 1200 

    last_activity = st.session_state.get("last_activity", time.time())

    # Check for inactivity
    if time.time() - last_activity > timeout_seconds:
        placeholder = st.empty()
        placeholder.warning("⚠️ Session expired due to inactivity.")
        time.sleep(3)
        placeholder.empty()

        # Clear session and redirect to login
        st.session_state.clear()
        st.session_state["page"] = "login"
        st.rerun()

    # Update last_activity on every rerun
    st.session_state["last_activity"] = time.time()

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
                try:
                    data = json.load(f)
                    if data is None:
                        return {}
                    return data
                except json.JSONDecodeError:
                    # File is empty or corrupted
                    return {}
        return {}  # file doesn't exist, return empty dict


    def save_chat_history(history):
        with open(chat_file, "w") as f:
            json.dump(history, f, indent=4)


    # Load existing history safely
    history = load_chat_history()

    # Make sure it's a dict
    if history is None:
        history = {}


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
                    WHERE table_name IN ('SCRAPED_DATA')
                    ORDER BY table_name, column_name
                    """
                    cursor.execute(query)
                    rows = cursor.fetchall()
                    connection.commit()
                    # print("rows", rows)
                    # Convert to DataFrame
                    df = pd.DataFrame(
                        rows, columns=["TABLE_NAME", "COLUMN_NAME", "DATA_TYPE"])
                    # print("df.........", df)

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
            return pd.DataFrame()   # ✅ Always return empty DataFrame
        except Exception as e:
            import traceback
            traceback.print_exc()
            return pd.DataFrame()   # ✅ Always return empty DataFrame

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

    def ask_oci_genai_for_chart(chart_prompt: str, x_list: list, y_list: list, df_preview: str) -> str:
        global COMPARTMENT_ID

        system_prompt = (
            "You are a Python data visualization assistant. "
            "Generate Python code to create a chart using Plotly Express or Plotly Graph Objects "
            "based on the user's request and given DataFrame structure. "
            "Do not redefine or create the DataFrame; assume it already exists as `df`. "
            "Do NOT call fig.show(). "
            "Output only Python code in a markdown code block."
        )

        full_prompt = f"""
        {system_prompt}

        USER REQUEST:
        {chart_prompt}

        DATAFRAME INFO:
        {df_preview}

        SELECTED COLUMNS:
        - X-axis: {x_list}
        - Y-axis: {y_list}

        Additional Rules:
        - Drop any rows where X or Y columns are null, NaN, or empty ('').
        - Integrate widgets if referenced.
        - Do not include explanations or text outside the code block.
        """

        chat_request = CohereChatRequest(
            message=full_prompt,
            max_tokens=500,
            temperature=0.4
        )

        chat_detail = ChatDetails(
            compartment_id=COMPARTMENT_ID,
            serving_mode=OnDemandServingMode(model_id=MODEL_ID),
            chat_request=chat_request
        )

        response = client.chat(chat_detail)
        print(response)

        if hasattr(response.data, "chat_response") and hasattr(response.data.chat_response, "text"):
            chart_code = response.data.chat_response.text
            # Remove any fig.show() if present
            chart_code = re.sub(r'fig\.show\(\)', '', chart_code)
            chart_code = chart_code.replace("```python", "").replace("```", "").strip()
            print("Generated chart code:\n", chart_code)
            return chart_code

        raise ValueError("Could not extract chart code from OCI GenAI response")


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
            "When converting dates, always use 24-hour format (HH24) in TO_DATE or TO_TIMESTAMP functions."
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
        # st.session_state.chat_history.append(
        #     {"role": "user", "message": user_question})
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

    m_p = st.empty()

    with st.sidebar:
        if st.session_state.get("username"):
            if st.button("🚪 Logout"):
                # ✅ Optional: 3-2-1 countdown
                countdown_placeholder = st.empty()
                for i in range(3, 0, -1):
                    countdown_placeholder.warning(f"⚡ Logging out in {i}...")
                    time.sleep(1)
                countdown_placeholder.empty()

                # ✅ Clear session completely
                st.session_state.clear()

                # ✅ Set page to login before rerun
                st.session_state["page"] = "login"
                st.rerun()

        # -------------------- App UI --------------------
        st.markdown("""
            <h1 style="font-size: 35px; color: #2C3E50; margin-top: -40px; text-align: center;">
                NUVAMA chat
            </h1>
        """, unsafe_allow_html=True)

        # ✅ Initialize session variables safely
        for key, default in {
            "query_result_df": pd.DataFrame(),
            "chat_history": [],
            "last_query": "",
            "last_query_columns": []
        }.items():
            if key not in st.session_state:
                st.session_state[key] = default
        try:
            if not st.session_state.query_result_df.empty:
                new_df1 = st.session_state.query_result_df.copy()
                new_df1.reset_index(drop=True, inplace=True)
                new_df1.index = new_df1.index + 1

                # === Initialize States ===
                st.session_state.setdefault("active_tab", "viz")
                st.session_state.setdefault("chart_metadata", [])

                # === Tab Buttons ===
                # colB, colA = st.columns([1, 1])
                # with colB:
                #     if st.button("📈 Visualize"):
                #         st.session_state.active_tab = "viz"
                # === Visualization Tab ===
                if st.session_state.active_tab == "viz":
                # Reset chart code if DataFrame shape changed
                    if "last_df_shape" not in st.session_state or st.session_state["last_df_shape"] != new_df1.shape:
                        st.session_state.pop("generated_chart_code", None)
                        st.session_state["last_df_shape"] = new_df1.shape
                        # st.warning("⚠️ Parent dataset changed — please reselect chart options if needed.")

                    # Column selectors
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

                    # === Create Chart ===
                    if st.button("🎨 Create Chart"):
                        if not x_axis_cols or not y_axis_cols or not chart_prompt:
                            st.warning("Select X & Y columns and enter chart description.")
                        else:
                            st.session_state["x_axis_cols"] = x_axis_cols
                            st.session_state["y_axis_cols"] = y_axis_cols
                            st.session_state["chart_prompt"] = chart_prompt

                            x_list = ", ".join(x_axis_cols)
                            y_list = ", ".join(y_axis_cols)

                            try:
                                # Use your function to generate chart code
                                chart_code = ask_oci_genai_for_chart(
                                    chart_prompt=chart_prompt,
                                    x_list=x_list,
                                    y_list=y_list,
                                    df_preview=str(new_df1.head(3))
                                )

                                # Extract Python code
                                chart_code_match = re.search(r"```python(.*?)```", chart_code, re.DOTALL)
                                if chart_code_match:
                                    st.session_state["generated_chart_code"] = chart_code_match.group(1).strip()
                                else:
                                    st.session_state["generated_chart_code"] = chart_code.strip()

                                st.success("✅ Chart code generated successfully!")

                            except Exception as e:
                                st.error(f"⚠️ Error generating chart code: {e}")
                                m_p.empty()

                    # === Execute & Store Charts ===
                    if "generated_chart_code" in st.session_state:
                        try:
                            exec_globals = {
                                "pd": pd,
                                "df": new_df1,
                                "px": px,
                                "go": go,
                                "np": np,
                                "st": st
                            }

                            # Remove any fig.show() if present in generated code
                            safe_code = re.sub(r'fig\.show\(\)', '', st.session_state["generated_chart_code"])
                            exec(safe_code, exec_globals)

                            # Collect all Plotly figures
                            new_figs = [obj for obj in exec_globals.values() if isinstance(obj, go.Figure)]

                            # Render figures with unique keys
                            if new_figs:
                                for i, fig in enumerate(new_figs):
                                    st.plotly_chart(fig, use_container_width=True, key=f"fig_{i}_{time.time()}")

                                # Save chart metadata
                                st.session_state["chart_metadata"].append({
                                    "code": st.session_state["generated_chart_code"],
                                    "x_cols": x_axis_cols,
                                    "y_cols": y_axis_cols
                                })

                            # Remove temporary generated code
                            st.session_state.pop("generated_chart_code", None)

                        except st.errors.StreamlitAPIException:
                            # st.warning("⚠️ Parent data changed — please reselect columns or regenerate the chart.")
                            st.session_state.pop("generated_chart_code", None)

                        except Exception as e:
                            st.error("❌ Chart rendering failed due to an unexpected error.")
                            st.exception(e)

                    # === Display previously created charts ===
                    if st.session_state.get("chart_metadata"):
                        st.subheader("📊 Created Charts")

                        df = st.session_state.query_result_df

                        # Column filters
                        try:
                            filter_cols = st.multiselect("Select columns to filter", df.columns.tolist())
                            filters = {}
                            for col in filter_cols:
                                unique_vals = sorted(df[col].dropna().unique())
                                selected_vals = st.multiselect(f"Filter {col}", unique_vals, default=unique_vals)
                                filters[col] = selected_vals

                            # Apply filters safely
                            filtered_df = df.copy()
                            for col, vals in filters.items():
                                if col in filtered_df.columns:
                                    filtered_df = filtered_df[filtered_df[col].isin(vals)]

                        except st.errors.StreamlitAPIException:
                            st.warning("⚠️ Filter settings reset — parent data structure changed.")
                            filtered_df = new_df1.copy()

                        # Display last 6 charts
                        grid_cols = st.columns(3)
                        for display_i, chart_index in enumerate(
                            range(max(0, len(st.session_state["chart_metadata"]) - 6),
                                len(st.session_state["chart_metadata"]))
                        ):
                            meta = st.session_state["chart_metadata"][chart_index]

                            try:
                                exec_globals = {"pd": pd, "df": filtered_df, "px": px, "go": go, "np": np}
                                exec(meta["code"], exec_globals)
                                fig = next((obj for obj in exec_globals.values() if isinstance(obj, go.Figure)), None)

                                if fig:
                                    with grid_cols[display_i % 3]:
                                        delete_key = f"delete_chart_{chart_index}"
                                        if st.button("❌", key=delete_key):
                                            st.session_state["chart_metadata"].pop(chart_index)
                                            st.rerun()
                                        st.plotly_chart(fig, use_container_width=True)

                            except st.errors.StreamlitAPIException:
                                st.warning("⚠️ Some charts couldn't render — parent data changed.")
                                continue

                            except Exception as e:
                                st.warning(f"⚠️ Chart {display_i + 1} could not be displayed: {e}")
                                continue
                    else:
                        st.info("No chart generated yet. Use the controls above to create one.")
            else:
                st.info("Please request data to generate chart first!")
        except st.errors.StreamlitAPIException:
            st.warning("⚠️ Parent dataset has changed — refreshing components...")
            time.sleep(2)
            for key in ["generated_chart_code", "x_axis_cols", "y_axis_cols", "chart_prompt", "chart_metadata"]:
                if key in st.session_state:
                    st.session_state.pop(key, None)

            st.rerun()
        

            

    # Fetch schema once
    if "schema_text" not in st.session_state:
        schema_df = fetch_schema()
        # print(schema_df)
        st.session_state.schema_text = schema_df.to_string(index=False)
        msg = st.success("✅ Schema fetched from ADB.")
        time.sleep(2)
        msg.empty()

    # st.title("NUVAMA chat ✅")

    current_user = st.session_state.get("username")
    if current_user is None:
        st.warning("⚠️ You are not logged in. Please login first.")
        st.stop() 

    if current_user not in history:
        history[current_user] = []

    user_input = st.chat_input("Ask your question here:")

    if user_input:
        mode = detect_mode(user_input)

        # Always append the question first
        st.session_state.chat_history.append({
            "role": "user",
            "message": user_input
        })
        history[current_user].append({
            "role": "user",
            "message": user_input
        })

        try:
            if mode == "Query":
                sql_query = ask_oci_genai_for_sql(user_input, st.session_state.schema_text)
                result_df = execute_generated_sql(sql_query)
                st.session_state.query_result_df = result_df

                # Append the generated query (as string)
                history[current_user].append({
                    "role": "assistant",
                    "message": sql_query
                })
                # st.session_state.chat_history.append({
                #     "role": "assistant",
                #     "message": sql_query
                # })
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "message": result_df
                })

            else:
                if st.session_state.query_result_df.empty:
                    response_msg = "Please ask for data first!"
                else:
                    ai_analysis = analyze_data_with_genai(
                        st.session_state.query_result_df, user_input, mode
                    )
                    response_msg = str(ai_analysis)

                history[current_user].append({
                    "role": "assistant",
                    "message": response_msg
                })
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "message": response_msg
                })

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            history[current_user].append({
                "role": "assistant",
                "message": error_msg
            })
            st.session_state.chat_history.append({
                "role": "assistant",
                "message": error_msg
            })

        # Save after each interaction
        save_chat_history(history)
        st.rerun()


    # Render chat history
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.chat_message("user").write(chat["message"])
        else:
            # If it's a DataFrame, render nicely
            if isinstance(chat["message"], pd.DataFrame):
                st.chat_message("assistant").write(chat["message"])
            else:
                st.chat_message("assistant").write(str(chat["message"]))


if "page" not in st.session_state:
    st.session_state["page"] = "login"

if st.session_state["page"] == "login":
    login_page()
    st.stop()  # Don't run landing page until login
elif st.session_state["page"] == "landing":
    if not st.session_state.get("authenticated", False):
        st.session_state["page"] = "login"
        st.rerun()
    else:
        landing_page()  # ✅ call landing page here