import streamlit as st
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

# Streamlit app
def main():
    st.title("AI Alert Classifier - Model Inference")

    st.subheader("Input")
    # Input text area
    input_data = st.text_area("Enter Input Data", value="[pr-cp-reg-12345 - kube-system] - CPUThrottlingHigh -  throttling of CPU in namespace kube-system for container aws-vpc-cni-init in pod aris-kube-prometheus-stack-kube-state-metrics-785d575975-s2j2k.")

    # Submit button
    if st.button("Predict"):
        try:
            # Define the endpoint URL
            local_endpoint_url = 'http://localhost:8080/invocations'
            # local_endpoint_url = 'http://ai-alert-classifier-inference-service.polyaxon:8080/invocations'

            # Define headers
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }

            input_json = {"inputs": input_data}
            input_json_str = json.dumps(input_json)

            # Make a POST request to the local endpoint
            response = requests.post(local_endpoint_url, data=input_json_str, headers=headers)

            if response.status_code == 200:
                predicted_label, score = parse_response(response.text)
                alert_severity = convert_to_alert_severity(predicted_label)
                st.subheader("Inference")
                col1, col2, col3 = st.columns(3)
                col1.metric("Alert Severity", alert_severity)
                col2.metric("Score", round(score, 5))
                col3.metric("Accuracy", f"{round(score * 100, 2)}%")
            else:
                st.error(f"Request failed with status code {response.status_code}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()
