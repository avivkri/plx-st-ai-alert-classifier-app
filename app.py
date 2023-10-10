import streamlit as st
import pandas as pd
import requests
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from sklearn.metrics import confusion_matrix
import ast


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


def plot_metrics(true_labels, predicted_labels, predicted_scores):
    # Calculate metrics
    # Assuming accuracy and other metrics are computed similarly as previously shown
    data_size = len(true_labels)
    accuracy = np.sum(true_labels == predicted_labels) / data_size
    prec_weighted = metrics.precision_score(true_labels, predicted_labels, average='weighted')
    recall_weighted = metrics.recall_score(true_labels, predicted_labels, average='weighted')
    f1_weighted = metrics.f1_score(true_labels, predicted_labels, average='weighted')
    roc_auc_weighted = metrics.roc_auc_score(true_labels, predicted_scores, multi_class='ovr', average='weighted')

    prec_micro = metrics.precision_score(true_labels, predicted_labels, average='micro')
    recall_micro = metrics.recall_score(true_labels, predicted_labels, average='micro')
    f1_micro = metrics.f1_score(true_labels, predicted_labels, average='micro')
    roc_auc_micro = metrics.roc_auc_score(true_labels, predicted_scores, multi_class='ovr', average='micro')

    prec_macro = metrics.precision_score(true_labels, predicted_labels, average='macro')
    recall_macro = metrics.recall_score(true_labels, predicted_labels, average='macro')
    f1_macro = metrics.f1_score(true_labels, predicted_labels, average='macro')
    roc_auc_macro = metrics.roc_auc_score(true_labels, predicted_scores, multi_class='ovr', average='macro')

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=['Accuracy'], y=[accuracy], width=0.1, palette='pastel')
    plt.text(0, accuracy - 0.05, f"{accuracy:.2f}", ha='center')
    st.pyplot(plt.gcf())

    # Grouped Bar Plot for other metrics
    df = pd.DataFrame({
        'Metric': ['precision', 'recall', 'f1', 'roc_auc'],
        'Weighted': [prec_weighted, recall_weighted, f1_weighted, roc_auc_weighted],
        'Micro': [prec_micro, recall_micro, f1_micro, roc_auc_micro],
        'Macro': [prec_macro, recall_macro, f1_macro, roc_auc_macro]
    })

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.25
    r1 = np.arange(len(df))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    bars1 = ax.bar(r1, df['Weighted'], width=bar_width, label='Weighted', color='#698396')
    bars2 = ax.bar(r2, df['Micro'], width=bar_width, label='Micro', color='#7c98ab')
    bars3 = ax.bar(r3, df['Macro'], width=bar_width, label='Macro', color='#a9c8c0')

    ax.set_xlabel('Metric', fontweight='bold')
    ax.set_xticks([r + bar_width for r in range(len(df))])
    ax.set_xticklabels(df['Metric'])
    ax.legend(loc="lower right")

    # Add values on top of bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')
    st.pyplot(fig)

    # ROC curve for multiclass
    n_classes = len(np.unique(true_labels))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(np.array(pd.get_dummies(true_labels))[:, i], predicted_scores[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i, color in zip(range(n_classes), ['blue', 'red', 'green', 'yellow', 'purple']):
        plt.plot(fpr[i], tpr[i], color=color,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for each class')
    plt.legend(loc="lower right")
    st.pyplot(plt.gcf())

    cm = confusion_matrix(true_labels, predicted_labels)
    print(cm)
    type(cm)
    # Labels for the classes
    labels = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]

    # Define bins for discrete colormap
    bins = [0, 1, 10, 50, 100, 200, 500, 600]
    colors = sns.color_palette("Blues", len(bins) - 1)
    colormap = plt.cm.colors.ListedColormap(colors)
    norm = plt.cm.colors.BoundaryNorm(bins, colormap.N)

    plt.figure(figsize=(5, 4))
    ax = sns.heatmap(cm, annot=False, fmt="d", cmap=colormap, norm=norm, linewidths=0.5)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    st.pyplot(plt.gcf())

    # Precision-Recall curve for multiclass
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = metrics.precision_recall_curve(np.array(pd.get_dummies(true_labels))[:, i], predicted_scores[:, i])
        average_precision[i] = metrics.average_precision_score(np.array(pd.get_dummies(true_labels))[:, i], predicted_scores[:, i])

    plt.figure(figsize=(8, 6))
    for i, color in zip(range(n_classes), ['blue', 'red', 'green', 'yellow', 'purple']):
        plt.plot(recall[i], precision[i], color=color,
                 label='Precision-Recall curve of class {0} (area = {1:0.2f})'.format(i, average_precision[i]))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve for each class')
    plt.legend(loc="upper right")
    st.pyplot(plt.gcf())


def main():
    st.set_page_config(page_title="AI Alert Classifier - Model Inference",
                       page_icon="https://budibase-bucket-3.s3.eu-west-1.amazonaws.com/logos/ai-alert-violet.png")
    st.title(":violet[AI Alert Classifier] :grey[-] Model Inference")

    st.subheader("Input")
    # Input text area
    input_data = st.text_area("Enter input data",
                              value="[pr-cp-reg-12345 - kube-system] - CPUThrottlingHigh -  throttling of CPU in namespace kube-system for container aws-vpc-cni-init in pod aris-kube-prometheus-stack-kube-state-metrics-785d575975-s2j2k.",
                              height=150)

    # Submit button
    if st.button("Predict"):
        try:
            # local_endpoint_url = 'http://localhost:8080/invocations'
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
                col1.metric("Alert Severity", alert_severity,
                            help="P0 - None, P1 - Critical, P2 - Error, P3 - Warning, P4 - Info")
                col2.metric("Score", rounded_score, help="0 - 1")
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

            col4, col5 = st.columns(2)
            col4.subheader("Response data")
            col4.json(response_text)
            col5.subheader("Response statistics")
            df = pd.DataFrame.from_dict(response_stats)
            col5.dataframe(df, hide_index=True)

            train_data = pd.read_csv("./data.csv", header=None, names=["label", "inputs"])

            test_data = train_data[["inputs"]]

            with open("predict_cleaned.csv.out", "r") as predict_file:
                predict_all = [ast.literal_eval(line.rstrip()) for line in predict_file]
                predict_all = [{"predicted_label": int(predict_line['predicted_label'][-1]),
                                "score": predict_line['probabilities'][
                                    predict_line['labels'].index(predict_line['predicted_label'])]} for predict_line in
                               predict_all]

            data_size = len(test_data)
            df_predict = pd.DataFrame(predict_all)

            true_labels = train_data.loc[: data_size - 1, "label"]
            predicted_labels = df_predict.loc[: data_size - 1, "predicted_label"]
            predicted_scores = pd.get_dummies(df_predict.loc[: data_size - 1, "predicted_label"]).values

            # After the predictions, plot the metrics
            st.subheader("Model metrics")
            plot_metrics(true_labels, predicted_labels, predicted_scores)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)


if __name__ == "__main__":
    main()