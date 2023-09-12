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

def segmented_progress_bar(current_value):
    percentage = current_value  # Use accuracy percentage as current value

    # Create HTML and CSS for the segmented progress bar
    html_code = f"""
    <div class="gauge">
      <div class="needle"></div>
    </div>
    
    <style>
        .gauge {{
            width: 300px; /* Adjust the width of the gauge */
          height: 20px; /* Adjust the height of the gauge */
          background: 
            linear-gradient(to right, #FF6B6B 0% 75%, #FFD700 75% 90%, #87CEEB 90% 97%, #98FB98 97% 100%);
          position: relative;
        }}
        
        .needle {{
          width: 0;
          height: 0;
          border-left: 10px solid transparent;
          border-right: 10px solid transparent;
          border-top: 20px solid black; /* Color of the arrow needle is black */
          position: absolute;
          top: -20px; /* Adjust the top position to position it at the top */
          left: {percentage}%;
          transform: translateX(-50%);
        }}
    </style>

    """

    return html_code

# Streamlit app
def main():
    st.title("AI Alert Classifier - Model Inference")

    st.subheader("Input")
    # Input text area
    input_data = st.text_area("Enter input data", value="[pr-cp-reg-12345 - kube-system] - CPUThrottlingHigh -  throttling of CPU in namespace kube-system for container aws-vpc-cni-init in pod aris-kube-prometheus-stack-kube-state-metrics-785d575975-s2j2k.")

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
                rounded_score = round(score, 5)
                accuracy_percentage = round(score * 100, 2)

                st.subheader("Inference")
                col1, col2, col3 = st.columns(3)
                col1.metric("Alert Severity", alert_severity)
                col2.metric("Score", rounded_score)
                col3.metric("Accuracy", f"{accuracy_percentage}%")
                col3.markdown(segmented_progress_bar(accuracy_percentage), unsafe_allow_html=True)
            else:
                st.error(f"Request failed with status code {response.status_code}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()
