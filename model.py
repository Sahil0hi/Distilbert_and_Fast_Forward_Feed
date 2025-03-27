import torch
import torch.nn as nn
import torch.optim as optim

# Hugging Face transformer
from transformers import DistilBertModel

class MyModel(nn.Module):
    """
    Example: DistilBERT for the 'Overview' text + feedforward for tabular features.
    Combines the two to predict 'Gross'.
    """
    def __init__(self, tabular_input_dim, text_embedding_dim=768):
        super(MyModel, self).__init__()

        # DistilBERT
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # Small feed-forward for tabular data
        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Combine text embedding (768) + tabular embedding (32) => 800
        combined_dim = text_embedding_dim + 32
        self.final_regressor = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # final output: box-office gross
        )

    def forward(self, inputs):
        """
        'inputs' should be a dict:
          {
            "text_input": {
                "input_ids": ...,
                "attention_mask": ...
            },
            "tabular_input": ...
          }
        """
        text_input = inputs["text_input"]        # { "input_ids": ..., "attention_mask": ... }
        tab_input  = inputs["tabular_input"]     # numeric features [batch_size, n_tab_features]

        # 1) DistilBERT forward
        bert_out = self.bert(**text_input)       # returns BaseModelOutput
        # Last hidden state shape: [batch_size, seq_len, hidden_size]
        # We'll take the [CLS] token (index=0) from each sequence
        cls_embedding = bert_out.last_hidden_state[:, 0, :]  # [batch_size, 768]

        # 2) Tabular feed-forward
        tab_out = self.tabular_net(tab_input)    # [batch_size, 32]

        # 3) Concatenate
        combined = torch.cat([cls_embedding, tab_out], dim=1)  # [batch_size, 800]

        # 4) Final regression
        output = self.final_regressor(combined)  # [batch_size, 1]
        return output


def create_model(features):
    tabular_input_dim = features["tabular_input"].shape[1]
    model = MyModel(tabular_input_dim)
    
    # Reduce learning rate
    optimizer = optim.Adam(model.parameters(), lr=1e-5)  # Changed from 1e-4
    
    return model, optimizer


if __name__ == '__main__':
    # Simple test with dummy data
    dummy_tab = torch.randn(5, 10)
    dummy_text_ids = torch.randint(0, 30522, (5, 12))
    dummy_mask = torch.ones(5, 12)

    dummy_features = {
        "text_input": {"input_ids": dummy_text_ids, "attention_mask": dummy_mask},
        "tabular_input": dummy_tab
    }

    model, optimizer = create_model(dummy_features)
    out = model(dummy_features)
    print("Output shape:", out.shape)  # [5, 1]
    print(model)
