import streamlit as st
import pandas as pd
import requests
import json


def parse_response(query_response):
    response = json.loads(query_response)
    predicted_label, score = (
        response[0]['label'],
        response[0]['score']
    )
    return predicted_label, score

def convert_to_alert_severity(predicted_label):
    label_to_severity_map = {
        "LABEL_0": "P0",
        "LABEL_1": "P1",
        "LABEL_2": "P2",
        "LABEL_3": "P3",
        "LABEL_4": "P4"
    }
    return label_to_severity_map.get(predicted_label, 'Unknown')


def segmented_progress_bar(current_value):
    percentage = current_value

    html_code = f"""
    <div class="gauge">
    </div>
    
    <style>
        .gauge {{
          width: 200px; 
          height: 10px;
          background:
            linear-gradient(to right, #FF3E3E 0% 75%, #FFD700 75% 90%, #4C87FF 90% 97%, #43D315 97% 100%);
          position: relative;
          border: 1px solid #666;
        }}
        
        .gauge::before {{
          content: "";
          width: 0;
          height: 0;
          border-left: 4px solid transparent;
          border-right: 4px solid transparent;
          border-top: 8px solid black;
          position: absolute;
          top: -8px;
          left: {percentage}%;
          transform: translateX(-50%);
        }}
    </style>
    """

    return html_code


def main():
    st.set_page_config(page_title="AI Alert Classifier - Model Inference",
                       page_icon="https://budibase-bucket-3.s3.eu-west-1.amazonaws.com/logos/ai-alert-violet.png")
    st.title(":violet[AI Alert Classifier] :lightgrey[-] Model Inference")

    st.subheader("Input")
    # Input text area
    input_data = st.text_area("Enter input data", value="[pr-cp-reg-12345 - kube-system] - CPUThrottlingHigh -  throttling of CPU in namespace kube-system for container aws-vpc-cni-init in pod aris-kube-prometheus-stack-kube-state-metrics-785d575975-s2j2k.")

    # Submit button
    if st.button("Predict"):
        try:
            #local_endpoint_url = 'http://localhost:8080/invocations'
            local_endpoint_url = 'http://192.168.49.4:8080/invocations'

            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }

            input_json = {"inputs": input_data}
            input_json_str = json.dumps(input_json)

            response = requests.post(local_endpoint_url, data=input_json_str, headers=headers)

            if response.status_code == 200:
                predicted_label, score = parse_response(response.text)
                alert_severity = convert_to_alert_severity(predicted_label)
                rounded_score = round(score, 5)
                confidence_percentage = round(score * 100, 2)

                st.subheader("Inference")
                col1, col2, col3 = st.columns(3)
                col1.metric("Alert Severity", alert_severity)
                col2.metric("Score", rounded_score)
                confidence_help_md = '''0 - 75% = Bad\n
75 - 90% = Average\n
90 - 97% = Good\n
97 - 100% = Best'''
                col3.metric("Confidence", f"{confidence_percentage}%", help=confidence_help_md)
                col3.markdown(segmented_progress_bar(confidence_percentage), unsafe_allow_html=True)
            else:
                st.error(f"Request failed with status code {response.status_code}")

            status_code = response.status_code
            headers = response.headers
            content_type = headers.get('Content-Type', 'N/A')
            content_length = headers.get('Content-Length', 'N/A')
            response_text = response.text
            response_time = response.elapsed.total_seconds()

            response_stats = {
                "Field": ["Status code", "Content type", "Content length", "Response time (s)"],
                "Value": [str(status_code), str(content_type), str(content_length), str(response_time)]
            }

            st.subheader("Response statistics")
            df = pd.DataFrame.from_dict(response_stats)
            st.dataframe(df, hide_index=True)
            st.caption("Response data")
            st.json(response_text, expanded=False)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()