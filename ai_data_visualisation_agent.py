import os
import json
import re
import sys
import io
import contextlib
import warnings
from typing import Optional, List, Any, Tuple
from PIL import Image
import streamlit as st
import pandas as pd
import base64
from io import BytesIO
from together import Together
from e2b_code_interpreter import Sandbox

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="AI Data Visualization Agent",
    page_icon="📊",
    layout="wide",
)

# ---------------- CODE EXECUTION ----------------
def code_interpret(e2b_code_interpreter: Sandbox, code: str):
    with st.spinner('⚡ Running code in E2B sandbox...'):
        exec = e2b_code_interpreter.run_code(code)

        if exec.error:
            st.error(f"❌ Slow Internet Connection : ") 
            return None

        results = exec.results or []

        # ✅ Always check for active matplotlib figures
        figs = [plt.figure(i) for i in plt.get_fignums()]
        if figs:
            results.extend(figs)
            plt.close('all')  # close after capturing

        return results

def match_code_blocks(llm_response: str) -> str:
    match = pattern.search(llm_response)
    return match.group(1) if match else ""

def chat_with_llm(e2b_code_interpreter: Sandbox, user_message: str, dataset_path: str) -> Tuple[Optional[List[Any]], str]:
    system_prompt = f"""You're a Python data scientist and data visualization expert. 
You are given a dataset at path '{dataset_path}' and also the user's query.
Always analyze and provide clear answers with charts or tables.
Use the dataset path '{dataset_path}' when reading the CSV file."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    with st.spinner('🤖 Talking to AI model...'):
        client = Together(api_key=st.session_state.together_api_key)
        response = client.chat.completions.create(
            model=st.session_state.model_name,
            messages=messages,
        )

        response_message = response.choices[0].message
        python_code = match_code_blocks(response_message.content)
        
        if python_code:
            code_interpreter_results = code_interpret(e2b_code_interpreter, python_code)
            return code_interpreter_results, response_message.content
        else:
            st.warning("⚠️ No Python code detected in AI response")
            return None, response_message.content

def upload_dataset(code_interpreter: Sandbox, uploaded_file) -> str:
    dataset_path = f"./{uploaded_file.name}"
    code_interpreter.files.write(dataset_path, uploaded_file)
    return dataset_path


# ---------------- MAIN APP ----------------
def main():
    st.title("📊 AI Data Visualization Agent")
    st.caption("Ask questions about your dataset and get instant AI-powered insights with charts & tables.")

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.header("🔑 API Keys")
        st.session_state.together_api_key = st.text_input("Together AI API Key", type="password")
        st.info("💡 Free $1 credit on Together AI [Get Key](https://api.together.ai/signin)")

        st.session_state.e2b_api_key = st.text_input("E2B API Key", type="password")
        st.markdown("[Get E2B Key](https://e2b.dev/docs/legacy/getting-started/api-key)")

        st.header("🤖 Model Selection")
        model_options = {
            "Meta-Llama 3.1 405B": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "DeepSeek V3": "deepseek-ai/DeepSeek-V3",
            "Qwen 2.5 7B": "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "Meta-Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        }
        model_choice = st.selectbox("Choose an AI Model", list(model_options.keys()))
        st.session_state.model_name = model_options[model_choice]

    # ---------- MAIN CONTENT ----------
    uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        with st.expander("📑 Dataset Preview", expanded=True):
            # ✅ CHANGE 1: Slider for preview instead of checkbox
            num_rows = st.slider("Select number of rows to preview:", 5, 50, 10, step=5)
            st.dataframe(df.head(num_rows))

        # ✅ CHANGE 2: Removed metric cards (rows/columns)
        st.success("✅ Dataset loaded successfully")

        query = st.text_area("💬 Ask a question about your data",
                             "Can you compare the average cost for two people between different categories?",
                             height=100)

        if st.button("🚀 Analyze", use_container_width=True):
            if not st.session_state.together_api_key or not st.session_state.e2b_api_key:
                st.error("⚠️ Please enter both API keys in the sidebar.")
            else:
                with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter:
                    dataset_path = upload_dataset(code_interpreter, uploaded_file)
                    code_results, llm_response = chat_with_llm(code_interpreter, query, dataset_path)

                    with st.expander("🤖 AI Response", expanded=True):
                        st.write(llm_response)

                    if code_results:
                        st.subheader("📊 Visualization / Results")
                        for result in code_results:
                            if hasattr(result, 'png') and result.png:
                                png_data = base64.b64decode(result.png)
                                image = Image.open(BytesIO(png_data))
                                st.image(image, caption="Generated Chart", use_container_width=True)
                            elif hasattr(result, 'figure'):
                                st.pyplot(result.figure)
                            elif hasattr(result, 'show'):
                                st.plotly_chart(result)
                            elif isinstance(result, (pd.DataFrame, pd.Series)):
                                st.dataframe(result)
                            else:
                                st.write(result)  

if __name__ == "__main__":
    main()

