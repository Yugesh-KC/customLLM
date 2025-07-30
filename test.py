from llm import GPTConfig  # needed only to load original checkpoint
import torch

# Load full checkpoint with class (this part needs GPTConfig)
checkpoint = torch.load('model.pt', map_location='cpu', weights_only=False)

# Extract raw model weights tensor dict
model_weights = checkpoint['model']

# Extract config as a plain dict (not a class instance)
if isinstance(checkpoint['config'], dict):
    config_dict = checkpoint['config']
else:
    # If config is a dataclass object (GPTConfig), convert to dict
    # You can do this by using vars() or dataclasses.asdict if dataclasses
    try:
        from dataclasses import asdict
        config_dict = asdict(checkpoint['config'])
    except ImportError:
        config_dict = vars(checkpoint['config'])

# Save weights separately (torch.save raw dict)
torch.save(model_weights, 'model_weights.pt')

# Save config separately as JSON for human-readable and portable format
import json
with open('config.json', 'w') as f:
    json.dump(config_dict, f, indent=4)

print("Saved model weights and config separately.")
