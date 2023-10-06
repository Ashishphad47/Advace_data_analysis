import streamlit as st
import pandas as pd
import json
#import Openai
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from apikey import apikey
import numpy as np
import warnings

# Suppress the UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

openai.api_key = apikey

# ... [Your existing functions here] ...

def csv_agent_func(file_path, user_message):
    """Run the CSV agent with the given file path and user message."""
    agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=apikey),
        #ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=apikey),
        file_path, 
       #verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )

    try:
        # Properly format the user's input and wrap it with the required "input" key
        tool_input = {
            "input": {
                "name": "python",
                #"arguments": user_message
                "arguments": json.dumps(user_message)
            }
        }
        
        response = agent.run(tool_input)
        print("Agent Response:", response)  # Debugging statement
        return response
    except Exception as e:
        st.write(f"Error: {e}")
        return None





def display_content_from_json(json_response):
    """
    Display content to Streamlit based on the structure of the provided JSON.
    """
    
    # Check if the response has plain text.
    if "answer" in json_response:
        st.write(json_response["answer"])

    # Check if the response has a bar chart.
    if "bar" in json_response:
        data = json_response["bar"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.bar_chart(df)

    # Check if the response has a table.
    if "table" in json_response:
        data = json_response["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)




def extract_code_from_response(response):
    """Extracts Python code from a string response."""
    # Use a regex pattern to match content between triple backticks
    code_pattern = r"```python(.*?)```"
    match = re.search(code_pattern, response, re.DOTALL)
    
    if match:
        # Extract the matched code and strip any leading/trailing whitespaces
        code= match.group(1).strip()
        code = code.replace("plt.show()", "# plt.show()")
        #return code
    else:
        code = ""
    parts = response.split('---INSIGHTS---')
    if len(parts) > 1:
        insights = parts[1].strip()
    else:
        insights = ""

    return code, insights
            
    #return None
    #return ""


def do_nothing():
    pass




def advanced_analysis_query(file_path, user_query):
    """Constructs an advanced query for GPT-4 based on the user input."""
   # base_query = f"Perform an advanced data analysis on the data in the CSV file at '{file_path}' considering the following user request: {user_query}. Provide Python code for the analysis, visualizations, and insights."
    base_query =(f"Perform an advanced data analysis on the data in the CSV file at '{file_path}' "
                  f"considering the following user request: {user_query}. "
                  f"Provide Python code for the analysis, visualizations, and also provide textual insights."
                  f"Generate a detailed textual analysis and visualizations explain about plots,."#please provide me python code provide textual insights of outpts ."
                  f"The response should be structured with insights first, followed by any code. "
                  f"Ensure the insights are comprehensive and the code is efficient and tailored to the dataset."
                  f"give responses like gpt-4 advance data analysis")
    return base_query

def csv_analyzer_app():
    """Main Streamlit application for CSV analysis."""

    st.title('CSV Assistant')
    st.write('Please upload your CSV file and enter your detailed analysis query below:')
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(file_details)
        
        # Save the uploaded file to disk
        file_path = os.path.join(uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        df = pd.read_csv(file_path)
        st.dataframe(df)
        
        user_input = st.text_area("Enter your analysis query here:")
        
        # Construct the advanced query
        advanced_query = advanced_analysis_query(file_path, user_input)
        
        if st.button('Run Analysis'):
            response = csv_agent_func(file_path, advanced_query)
            if response:  # This checks if the response is not None
                code_to_execute = extract_code_from_response(response)
                textual_analysis= extract_code_from_response(response)
                actual_code_to_execute = code_to_execute[0]
                if actual_code_to_execute:
                    plots = actual_code_to_execute.split('plt.show()')
#                '''if code_to_execute:
 #                   st.write("Executing Code:", code_to_execute)'''
  #              '''output = exec(code_to_execute, globals(), {"df": df, "plt": plt, 'sns': sns})
   #                 st.write("Output:", output)'''
                try:
                    sns.set(rc={'figure.figsize': (11.7, 8.27)})
                    plots = actual_code_to_execute.split('plt.show()')
                        #output = exec(code_to_execute, globals(), {"df": df, "plt": plt, 'sns': sns}) 
                    for plot_code in plots:
                        if plot_code.strip():  # Check if there's actual code to execute
                            exec(plot_code, globals(), {"df": df, "plt": plt, 'sns': sns})
                            fig = plt.gcf()

                                # Only display the figure if it has content
                            if fig.axes:
                                st.pyplot(fig)  
                            plt.close(fig)
                        #exec(code_to_execute, globals(), {"df": df, "plt": plt, 'sns': sns, 'show': do_nothing}) 
                        #exec(code_to_execute, globals(), {"df": df, "plt": plt, 'sns': sns})
                        #'''fig = plt.gcf()
                        #if fig.axes:
                         #   st.pyplot(fig)
                        #st.pyplot(fig)  # Only display if the figure has axes (i.e., content)
                        #if output:
                        #   st.write(output)  
                        #st.pyplot(fig)  '''
                except Exception as e:
                    st.write(f"Error executing code: {e}")
                    import traceback
                    st.write(traceback.format_exc())
                else:
                    if response and isinstance(response, str):
                        st.write(response)

                    else:
                        st.warning("The response from the agent did not contain executable code or a valid message.")  
                #st.write("Analysis Insights:")          
                st.write(textual_analysis)        
            else:
                st.write("No valid response received.")
    else:
        st.warning("Please upload a CSV file to proceed.")

csv_analyzer_app()


