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

# Streamlit app
def main():
    st.title("Model Inference App")

    # Input text area
    input_data = st.text_area("Enter Input Data", value="\"inputs\": \"[pr-cp-reg-12345 - kube-system] - CPUThrottlingHigh -  throttling of CPU in namespace kube-system for container aws-vpc-cni-init in pod aris-kube-prometheus-stack-kube-state-metrics-785d575975-s2j2k.\"")

    # Submit button
    if st.button("Predict"):
        try:
            # Define the endpoint URL
            local_endpoint_url = 'http://ai-alert-classifier-inference-service.polyaxon:8080/invocations'

            # Define headers
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }

            # Make a POST request to the local endpoint
            response = requests.post(local_endpoint_url, data=input_data, headers=headers)

            if response.status_code == 200:
                predicted_label, score = parse_response(response.json())
                st.subheader("Inference")
                st.write(f"Input text: {input_data}")
                st.write("Model prediction")
                st.write(f"Label: {predicted_label}")
                st.write(f"Score: {score}")
            else:
                st.error(f"Request failed with status code {response.status_code}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()