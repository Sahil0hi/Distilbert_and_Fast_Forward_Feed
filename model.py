import torch
import torch.nn as nn
import torch.optim as optim
from transformers import DistilBertModel

class MyModel(nn.Module):
    """
    Multimodal model:
    - Text input: Movie 'Overview' processed with DistilBERT
    - Tabular input: Numerical features (like rating, votes, etc.)
    Predicts: Movie 'Gross' revenue (regression)
    """
    def __init__(self, tabular_input_dim, text_embedding_dim=768):
        super(MyModel, self).__init__()

        # Text encoder: DistilBERT
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # Tabular feed-forward network
        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # ✅ Added dropout for regularization
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Combined head: Text (768) + Tabular (32)
        combined_dim = text_embedding_dim + 32
        self.final_regressor = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # ✅ Added dropout
            nn.Linear(64, 1)  # Final output: predicted gross revenue
        )

    def forward(self, inputs):
        """
        Inputs:
            inputs (dict):
                - text_input: {"input_ids": ..., "attention_mask": ...}
                - tabular_input: [batch_size, n_tabular_features]
        """
        text_input = inputs["text_input"]
        tab_input = inputs["tabular_input"]

        # Handle any NaN values in tabular input
        if torch.isnan(tab_input).any():
            tab_input = torch.nan_to_num(tab_input, nan=0.0)

        # 1) Text: DistilBERT
        bert_output = self.bert(**text_input)
        # Take [CLS] token embedding (index 0)
        cls_embedding = bert_output.last_hidden_state[:, 0, :]  # shape: [batch_size, 768]
        
        # Check for NaNs in text embeddings
        if torch.isnan(cls_embedding).any():
            cls_embedding = torch.nan_to_num(cls_embedding, nan=0.0)

        # 2) Tabular
        tab_out = self.tabular_net(tab_input)  # shape: [batch_size, 32]
        
        # Check for NaNs in tabular output
        if torch.isnan(tab_out).any():
            tab_out = torch.nan_to_num(tab_out, nan=0.0)

        # 3) Combine
        combined = torch.cat([cls_embedding, tab_out], dim=1)  # shape: [batch_size, 800]

        # 4) Regression head
        output = self.final_regressor(combined)  # shape: [batch_size, 1]
        
        # Final NaN check
        if torch.isnan(output).any():
            output = torch.nan_to_num(output, nan=0.0)
            
        return output


def create_model(features):
    """
    Create model and optimizer.
    Args:
        features (dict): Preprocessed input dictionary from get_prepared_data()

    Returns:
        model (MyModel): initialized model
        optimizer (torch.optim.Optimizer): optimizer for training
    """
    tabular_input_dim = features["tabular_input"].shape[1]
    model = MyModel(tabular_input_dim)

    # Use Adam optimizer with small learning rate (fine-tuning BERT)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    return model, optimizer


if __name__ == '__main__':
    # Dummy test to make sure the model runs

    from transformers import DistilBertTokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Dummy inputs
    dummy_tab = torch.randn(5, 10)  # batch size 5, tabular features 10
    dummy_texts = ["This is a dummy movie overview."] * 5
    dummy_tokens = tokenizer(dummy_texts, padding=True, truncation=True, return_tensors="pt")

    dummy_features = {
        "text_input": dummy_tokens,
        "tabular_input": dummy_tab
    }

    model, optimizer = create_model(dummy_features)
    out = model(dummy_features)

    print("Output shape:", out.shape)  # Should be [5, 1]
    print(model)
