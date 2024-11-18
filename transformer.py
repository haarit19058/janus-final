from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd



data = pd.read_csv("data.csv")  


s1t1_threshold_lower = 0.31
s1t1_threshold_higher = 0.8 
oscillator_strength_threshold_lower = 0.2 
oscillator_strength_threshold_upper = 0.6

mr_tadf_emitters = data[
    (data['S1-T1'] >= s1t1_threshold_lower) & 
    (data['S1-T1'] <= s1t1_threshold_higher) &  # Filtering based on a range of S1-T1
    (data['f'] >= oscillator_strength_threshold_lower) &  # Filtering based on f value
    (data['f']<=oscillator_strength_threshold_upper)&
    (data['SMILES'].str.contains('[Bb]'))  # Check if 'B' or 'b' is in SMILES
]


data = mr_tadf_emitters

print(data)




class SMILESDataset(Dataset):
    def __init__(self, smiles, labels, tokenizer):
        self.smiles = smiles
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        # Access the SMILES string using .iloc to handle custom indices:
        encoding = self.tokenizer(self.smiles.iloc[idx], return_tensors="pt", padding='max_length', truncation=True, max_length=512) 
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return {**encoding, 'labels': label}

# Load your data
smiles_list = data['SMILES']

s1_t1 = data['S1-T1'].tolist()


# Split the dataset (80% train, 20% validation)
train_size = int(0.8 * len(smiles_list))
train_smiles = smiles_list[:train_size]
val_smiles = smiles_list[train_size:]
train_labels = s1_t1[:train_size]
val_labels = s1_t1[train_size:]

# Load the tokenizer and model
model_name = "seyonec/ChemBERTa-zinc-base-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

# Prepare datasets and dataloaders
train_dataset = SMILESDataset(train_smiles, train_labels, tokenizer)
val_dataset = SMILESDataset(val_smiles, val_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
for epoch in range(3):  # Adjust epochs as needed
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].squeeze(1).to(device)
        attention_mask = batch['attention_mask'].squeeze(1).to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Validation (optional)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

    print(f"Epoch {epoch + 1}: Validation Loss: {total_loss / len(val_loader)}")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_model_s1t1")
tokenizer.save_pretrained("fine_tuned_model_s1t1")



from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd

class SMILESDataset(Dataset):
    def __init__(self, smiles, labels, tokenizer):
        self.smiles = smiles
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        # Access the SMILES string using .iloc to handle custom indices:
        encoding = self.tokenizer(self.smiles.iloc[idx], return_tensors="pt", padding='max_length', truncation=True, max_length=512) 
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return {**encoding, 'labels': label}

# Load your data
smiles_list = data['SMILES']

s1_t1 = data['f'].tolist()


# Split the dataset (80% train, 20% validation)
train_size = int(0.8 * len(smiles_list))
train_smiles = smiles_list[:train_size]
val_smiles = smiles_list[train_size:]
train_labels = s1_t1[:train_size]
val_labels = s1_t1[train_size:]

# Load the tokenizer and model
model_name = "seyonec/ChemBERTa-zinc-base-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

# Prepare datasets and dataloaders
train_dataset = SMILESDataset(train_smiles, train_labels, tokenizer)
val_dataset = SMILESDataset(val_smiles, val_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
for epoch in range(3):  # Adjust epochs as needed
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].squeeze(1).to(device)
        attention_mask = batch['attention_mask'].squeeze(1).to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Validation (optional)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

    print(f"Epoch {epoch + 1}: Validation Loss: {total_loss / len(val_loader)}")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_model_f")
tokenizer.save_pretrained("fine_tuned_model_f")



print(data.shape)




import pandas as pd
import torch
from janus import JANUS, utils
from rdkit import Chem, RDLogger
from transformers import AutoTokenizer, AutoModelForSequenceClassification

RDLogger.DisableLog("rdApp.*")

# Load data
# data = pd.read_csv('data.csv')

initSmiles = data['SMILES']
# Save SMILES to a text file for initial population in JANUS
with open('smiles.txt', 'w') as f:
    for smi in initSmiles:
        f.write(f"{smi.strip()}\n")

# Load the fine-tuned models and tokenizers
model_s1t1 = AutoModelForSequenceClassification.from_pretrained("fine_tuned_model_s1t1")
tokenizer_s1t1 = AutoTokenizer.from_pretrained("fine_tuned_model_s1t1")
model_f = AutoModelForSequenceClassification.from_pretrained("fine_tuned_model_f")
tokenizer_f = AutoTokenizer.from_pretrained("fine_tuned_model_f")

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_s1t1.to(device)
model_s1t1.eval()
model_f.to(device)
model_f.eval()



def largest_ring_size(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES string")

    ring_info = molecule.GetRingInfo()
    ring_sizes = [len(ring) for ring in ring_info.BondRings()]
    return max(ring_sizes) if ring_sizes else 0


# Define the fitness function
def fitness_function(smi: str) -> float:
    inputs_s1t1 = tokenizer_s1t1(smi, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    inputs_f = tokenizer_f(smi, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    if largest_ring_size(smi) > 7:
        return -10

    with torch.no_grad():
        prediction_s1t1 = model_s1t1(**inputs_s1t1).logits.squeeze().cpu().numpy()
        prediction_f = model_f(**inputs_f).logits.squeeze().cpu().numpy()

    #return prediction_f * 10

    if prediction_s1t1<s1t1_threshold_higher and prediction_s1t1>s1t1_threshold_lower and prediction_f>oscillator_strength_threshold_lower:
        return  (10*prediction_f + 10/(prediction_s1t1))

    else:
        return -1




# Define a custom filter function
def custom_filter(smi: str) -> bool:
    return 60 < len(smi) <= 100

# Configure JANUS parameters
params_dict = {
    "generations": 100,
    "generation_size": 300,
    "num_exchanges": 2,
    "custom_filter": custom_filter,
    "use_fragments": True,
    "use_classifier": True,
}

# Create JANUS agent and start the optimization
agent = JANUS(
    work_dir='RESULTS',
    fitness_function=fitness_function,
    start_population='./smiles.txt',
    **params_dict
)

# Alternatively, load parameters from a YAML file and start the agent
params_dict = utils.from_yaml(
    work_dir='RESULTS',
    fitness_function=fitness_function,
    start_population='./smiles.txt',
    yaml_file='./default_params.yml',
    **params_dict
)

print("Starting JANUS...")
agent = JANUS(**params_dict)
agent.run()
print("JANUS run complete.")

