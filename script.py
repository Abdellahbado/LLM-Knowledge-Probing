import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering # Keep import for consistency, though not explicitly used after linkage
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import os

hf_token = os.environ.get("HF_TOKEN") 
if not hf_token:
     print("Warning: HF_TOKEN not found in environment variables. Model loading might fail if it requires authentication.")
     

model_name = "google/gemma-3-4b-it" 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token 
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token) 
    tokenizer.pad_token = tokenizer.eos_token 
except Exception as e:
    print(f"Error loading model {model_name}. Make sure HF_TOKEN is set correctly if needed, or try an open model.")
    print(e)
    exit() 

prompt_template = "Subject: {country_name}. This country is located in the continent of:"
layers_to_probe = [0, 4, 8, 12, 16, 20, 24, 28, 31]
print(f"Probing layers: {layers_to_probe}")

os.makedirs("plots", exist_ok=True) 

train_pairs = [
    # Europe (10)
    ("France", "Europe"), ("Germany", "Europe"), ("Spain", "Europe"), ("Poland", "Europe"),
    ("Sweden", "Europe"), ("Greece", "Europe"), ("Portugal", "Europe"), ("Belgium", "Europe"),
    ("Italy", "Europe"), ("Netherlands", "Europe"),
    # Asia (10)
    ("Japan", "Asia"), ("China", "Asia"), ("Thailand", "Asia"), ("Vietnam", "Asia"),
    ("South Korea", "Asia"), ("Malaysia", "Asia"), ("Indonesia", "Asia"), ("Philippines", "Asia"),
    ("India", "Asia"), ("Pakistan", "Asia"),
    # Africa (10)
    ("Nigeria", "Africa"), ("Ghana", "Africa"), ("Algeria", "Africa"), ("Morocco", "Africa"),
    ("Egypt", "Africa"), ("Tunisia", "Africa"), ("Senegal", "Africa"), ("Cameroon", "Africa"),
    ("Kenya", "Africa"), ("South Africa", "Africa"),
    # North America (10)
    ("United States", "North America"), ("Mexico", "North America"), ("Cuba", "North America"),
    ("Panama", "North America"), ("Costa Rica", "North America"), ("Jamaica", "North America"),
    ("Honduras", "North America"), ("Guatemala", "North America"), ("Canada", "North America"),
    ("Dominican Republic", "North America"),
    # South America (10)
    ("Brazil", "South America"), ("Argentina", "South America"), ("Peru", "South America"),
    ("Colombia", "South America"), ("Chile", "South America"), ("Bolivia", "South America"),
    ("Uruguay", "South America"), ("Paraguay", "South America"), ("Venezuela", "South America"),
    ("Ecuador", "South America")
]
test_pairs = [
    ("Switzerland", "Europe"), ("Norway", "Europe"),
    ("Bangladesh", "Asia"), ("Nepal", "Asia"),
    ("Uganda", "Africa"), ("Zambia", "Africa"),
    ("Haiti", "North America"), ("Nicaragua", "North America"),
    ("Guyana", "South America"), ("Suriname", "South America")
]
print(f"Train examples: {len(train_pairs)}, Test examples: {len(test_pairs)}")

def get_subject_token_index(prompt, subject_name, tokenizer):
    try:
        start_char = prompt.index(subject_name)
        end_char = start_char + len(subject_name)

        inputs = tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=True)
        offset_mapping = inputs["offset_mapping"][0].tolist() # Convert to list for easier iteration

        token_indices = [
            i for i, (token_start, token_end) in enumerate(offset_mapping)
            if max(start_char, token_start) < min(end_char, token_end) # Check for overlap
        ]

        if not token_indices:
             for i, (token_start, token_end) in enumerate(offset_mapping):
                  if token_start <= start_char < token_end:
                       return i
             return inputs.input_ids.shape[1] - (1 if inputs.input_ids[0,-1].item() == tokenizer.eos_token_id else 0) - 1


        return token_indices[-1]

    except ValueError:
        print(f"Warning: Subject '{subject_name}' not found in prompt string '{prompt}'. Using fallback position.")
        return len(tokenizer.encode(prompt, return_tensors="pt")[0]) - (1 if tokenizer.encode(prompt, return_tensors="pt")[0][0,-1].item() == tokenizer.eos_token_id else 0) - 1
    except Exception as e:
         print(f"An error occurred in get_subject_token_index for subject '{subject_name}': {e}")
         return len(tokenizer.encode(prompt, return_tensors="pt")[0]) - (1 if tokenizer.encode(prompt, return_tensors="pt")[0][0,-1].item() == tokenizer.eos_token_id else 0) - 1


def get_representations(subjects, model, tokenizer, layers, prompt_template):
    representations = {layer: [] for layer in layers}

    model.eval() 
    with torch.no_grad(): 
        for subject in subjects:
            prompt = prompt_template.format(country_name=subject)
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
            subject_token_index = get_subject_token_index(prompt, subject, tokenizer)

            if subject_token_index >= inputs.input_ids.shape[1]:
                 print(f"Warning: Subject token index out of bounds for '{subject}'. Index: {subject_token_index}, Input length: {inputs.input_ids.shape[1]}. Using last token.")
                 subject_token_index = inputs.input_ids.shape[1] - 1 # Fallback to last token

            outputs = model(**inputs, output_hidden_states=True)

            for layer in layers:
                if layer < len(outputs.hidden_states):
                    hidden_state = outputs.hidden_states[layer][0, subject_token_index, :].clone().detach()

                    rep = hidden_state.float().cpu().numpy()
                    representations[layer].append(rep)
                else:
                    print(f"Warning: Layer {layer} is out of bounds for model with {len(outputs.hidden_states)} layers. Skipping.")
                    pass


    for layer in layers:
         if representations[layer]: 
            representations[layer] = np.array(representations[layer])
         else:
            representations[layer] = np.array([]) 

    return representations

train_subjects = [pair[0] for pair in train_pairs]
test_subjects = [pair[0] for pair in test_pairs]
print("Extracting train representations...")
train_reps = get_representations(train_subjects, model, tokenizer, layers_to_probe, prompt_template)
print("Extracting test representations...")
test_reps = get_representations(test_subjects, model, tokenizer, layers_to_probe, prompt_template)
print("Representation extraction complete.")

label_encoder = LabelEncoder()
all_objects = [obj for _, obj in train_pairs + test_pairs]
label_encoder.fit(all_objects)
y_train_labels = label_encoder.transform([obj for _, obj in train_pairs])
y_test_labels = label_encoder.transform([obj for _, obj in test_pairs])

accuracies = {}
cv_scores = {}
predictions = {}
probes = {} 
print("\nTraining and evaluating probes across layers...")
for layer in layers_to_probe:
    X_train = train_reps[layer]
    X_test = test_reps[layer]

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print(f"Skipping layer {layer} due to missing representations.")
        accuracies[layer] = 0.0
        cv_scores[layer] = 0.0
        predictions[layer] = np.array([])
        continue 

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=1.0, max_iter=1000, multi_class="auto", solver="liblinear"))
    ])
    pipeline.fit(X_train, y_train_labels)
    y_pred_labels = pipeline.predict(X_test)
    accuracies[layer] = accuracy_score(y_test_labels, y_pred_labels)
    if X_train.shape[0] >= 5:
        cv_scores[layer] = cross_val_score(pipeline, X_train, y_train_labels, cv=min(5, X_train.shape[0])).mean() # Use min(5, n_samples)
    else:
        cv_scores[layer] = np.nan # Not enough samples for CV
        print(f"Warning: Not enough training samples ({X_train.shape[0]}) for 5-fold CV at layer {layer}.")

    predictions[layer] = y_pred_labels
    probes[layer] = pipeline # Store the trained pipeline
    print(f"Layer {layer} - Test Accuracy: {accuracies[layer]:.4f}, CV Score: {cv_scores[layer]:.4f}")

print("\nGenerating layer-wise accuracy plots...")
plot_layers = [layer for layer in layers_to_probe if layer in accuracies and (accuracies[layer] > 0 or cv_scores[layer] > 0 or not np.isnan(cv_scores[layer]))]
plot_accuracies = [accuracies[layer] for layer in plot_layers]
plot_cv_scores = [cv_scores[layer] if not np.isnan(cv_scores[layer]) else 0 for layer in plot_layers] # Handle NaN for plotting

fig = go.Figure()
fig.add_trace(go.Scatter(x=plot_layers, y=plot_accuracies, mode='lines+markers', name='Test Accuracy'))
fig.add_trace(go.Scatter(x=plot_layers, y=plot_cv_scores, mode='lines+markers', name='CV Score'))
fig.update_layout(title='Probe Accuracy and CV Score Across Layers', xaxis_title='Layer', yaxis_title='Score', hovermode='x')
fig.write_html("plots/layer_wise_accuracy.html")
print("Saved plots/layer_wise_accuracy.html")

plt.figure(figsize=(10, 6))
plt.plot(plot_layers, plot_accuracies, 'b-o', label='Test Accuracy')
plt.plot(plot_layers, plot_cv_scores, 'r-o', label='CV Score (Train)')
plt.title('Probe Accuracy and CV Score Across Layers')
plt.xlabel('Layer')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.savefig("plots/layer_wise_accuracy.png")
print("Saved plots/layer_wise_accuracy.png")
plt.close() # Close the plot to free memory

if not accuracies or max(accuracies.values()) == 0:
    best_layer = layers_to_probe[-1] # Default to last probed layer or handle error
    print("\nCould not determine best layer based on accuracy (all zero or no data). Defaulting to last probed layer.")
else:
    best_layer = max(accuracies, key=accuracies.get)
    print(f"\nBest Layer: {best_layer} with Test Accuracy: {accuracies[best_layer]:.4f}")

print(f"\nAnalyzing representations at the best layer ({best_layer})...")
all_subjects = train_subjects + test_subjects
if best_layer not in train_reps or best_layer not in test_reps or train_reps[best_layer].shape[0] == 0 or test_reps[best_layer].shape[0] == 0:
    print(f"Error: Representations for best layer {best_layer} not available. Skipping similarity and PCA analysis.")
else:
    all_reps = np.vstack([train_reps[best_layer], test_reps[best_layer]])
    all_labels = [obj for _, obj in train_pairs] + [obj for _, obj in test_pairs]

    sorted_indices = np.argsort(all_labels)
    sorted_subjects = [all_subjects[i] for i in sorted_indices]
    sorted_reps = all_reps[sorted_indices]
    sorted_labels = [all_labels[i] for i in sorted_indices]


    if sorted_reps.shape[0] > 1:
        distance_matrix = pairwise_distances(sorted_reps, metric='cosine')
        similarity_matrix = 1 - distance_matrix

        plt.figure(figsize=(12, 10))
        sns.heatmap(similarity_matrix, xticklabels=sorted_subjects, yticklabels=sorted_subjects, cmap='viridis')
        plt.title(f"Cosine Similarity Heatmap (Layer {best_layer})")
        plt.tight_layout()
        plt.savefig("plots/similarity_heatmap.png")
        print("Saved plots/similarity_heatmap.png")
        plt.close()

        if sorted_reps.shape[0] > 1:
            plt.figure(figsize=(12, 6))
            linkage_matrix = linkage(distance_matrix, method='average')
            dendrogram(linkage_matrix, labels=sorted_subjects, leaf_rotation=90)
            plt.title(f"Dendrogram of Representations (Layer {best_layer})")
            plt.tight_layout()
            plt.savefig("plots/dendrogram.png")
            print("Saved plots/dendrogram.png")
            plt.close()
        else:
            print("Skipping dendrogram: Not enough samples.")

    else:
        print("Skipping similarity heatmap and dendrogram: Not enough samples.")


    if all_reps.shape[0] > 1 and all_reps.shape[1] >= 3:
        pca = PCA(n_components=3)
        reduced_reps = pca.fit_transform(all_reps)

        import pandas as pd
        df = pd.DataFrame({
            'PC1': reduced_reps[:, 0],
            'PC2': reduced_reps[:, 1],
            'PC3': reduced_reps[:, 2],
            'Continent': all_labels,
            'Country': all_subjects,
            'Set': ['Train'] * len(train_subjects) + ['Test'] * len(test_subjects)
        })

        fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='Continent',
                            symbol='Set', hover_name='Country',
                            title=f'3D PCA of Representations (Layer {best_layer})',
                            labels={'PC1': 'Principal Component 1',
                                    'PC2': 'Principal Component 2',
                                    'PC3': 'Principal Component 3'})
        fig.write_html("plots/pca_3d.html")
        print("Saved plots/pca_3d.html")


        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        continents = df['Continent'].unique()
        colors = plt.cm.get_cmap('tab10', len(continents)) # Using a colormap
        markers = {'Train': 'o', 'Test': '^'}

        for i, continent in enumerate(continents):
            for dataset in ['Train', 'Test']:
                subset = df[(df['Continent'] == continent) & (df['Set'] == dataset)]
                if not subset.empty:
                    ax.scatter(subset['PC1'], subset['PC2'], subset['PC3'],
                            color=colors(i), marker=markers[dataset],
                            label=f"{continent} ({dataset})")

        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.set_title(f'3D PCA of Representations (Layer {best_layer})')
        plt.legend()
        plt.savefig("plots/pca_3d_static.png")
        print("Saved plots/pca_3d_static.png")
        plt.close()

    else:
         print("Skipping 3D PCA: Not enough samples or dimensions.")


print("\nPerforming Error Analysis (excluding gradient feature importance)...")
misclassified = []
continent_misclassifications = defaultdict(lambda: defaultdict(int))

if best_layer in predictions and len(predictions[best_layer]) == len(y_test_labels):
    for i, (true_label, pred_label) in enumerate(zip(y_test_labels, predictions[best_layer])):
        true_continent = label_encoder.inverse_transform([true_label])[0]
        pred_continent = label_encoder.inverse_transform([pred_label])[0]

        if true_label != pred_label:
            subject = test_subjects[i]
            misclassified.append((subject, true_continent, pred_continent))
            continent_misclassifications[true_continent][pred_continent] += 1

    print("\nMisclassified Test Subjects:")
    if misclassified:
        for subject, true, pred in misclassified:
            print(f"Subject: {subject}, True: {true}, Predicted: {pred}")
    else:
        print("No misclassified test subjects found at the best layer.")


    print("\nMisclassification Patterns:")
    if continent_misclassifications:
        for true_continent, misclassified_to in continent_misclassifications.items():
            print(f"{true_continent} was misclassified as:")
            for pred_continent, count in misclassified_to.items():
                print(f"  - {pred_continent}: {count} times")
    else:
        print("No misclassification patterns found.")



else:
    print("Skipping detailed error analysis: Predictions for the best layer are not available or do not match test labels.")



print("\nCalculating Accuracy per Continent...")
continent_accuracies = defaultdict(lambda: {'correct': 0, 'total': 0})

if best_layer in predictions and len(predictions[best_layer]) == len(y_test_labels):
    for i, (true_label, pred_label) in enumerate(zip(y_test_labels, predictions[best_layer])):
        true_continent = label_encoder.inverse_transform([true_label])[0]
        continent_accuracies[true_continent]['total'] += 1
        if true_label == pred_label:
            continent_accuracies[true_continent]['correct'] += 1

    print("\nAccuracy per Continent:")
    if continent_accuracies:
        for continent, stats in continent_accuracies.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"{continent}: {accuracy:.2f} ({stats['correct']}/{stats['total']})")

        plt.figure(figsize=(10, 6))
        continents = list(continent_accuracies.keys())
        accs = [stats['correct'] / stats['total'] if stats['total'] > 0 else 0 for stats in continent_accuracies.values()]
        plt.bar(continents, accs)
        plt.title(f"Accuracy per Continent (Layer {best_layer})")
        plt.xlabel("Continent")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.1) # Extend y-limit slightly for text labels
        plt.xticks(rotation=45, ha='right') # Rotate labels if they overlap
        for i, v in enumerate(accs):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
        plt.tight_layout() # Adjust layout to prevent labels overlapping
        plt.savefig("plots/continent_accuracy.png")
        print("Saved plots/continent_accuracy.png")
        plt.close()

    else:
        print("No continent accuracy data available.")

else:
    print("Skipping accuracy per continent analysis: Predictions for the best layer are not available or do not match test labels.")


print("\nAnalysis Complete. Results saved to 'plots' directory.")