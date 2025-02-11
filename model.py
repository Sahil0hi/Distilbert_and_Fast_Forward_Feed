import torch
import torch.nn as nn
import torch.optim as optim

# Hugging Face transformer
from transformers import DistilBertModel

# Define model (feed-forward, two hidden layers)
# TODO: This is where most of the work will be done. You can change the model architecture,
#       add more layers, change activation functions, etc.

class MyModel(nn.Module):
    """using DistilBERT for text embeddings plus
    a feedforward network for tabular features
    then combines
    """
    def __init__(self, tabular_input_dim, text_embedding_dim=768):
        super(MyModel, self).__init__()

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # feed-forward for the tabular portion
        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # combines [BERT-embedding + tabular features]
        #    DistilBERT’s pooled output is typically 768-dim. We pass that plus
        #    the 32-dim output from tabular_net = 800 dims total
        combined_dim = text_embedding_dim + 32
        self.final_regressor = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # single value for box-office gross
        )

    def forward(self, inputs):
        """
        Expects 'inputs' to be a dict or tuple containing: text_input: a dict of { "input_ids", "attention_mask" } for DistilBERT

          tabular_input: a float tensor of shape [batch_size, tabular_input_dim]
        """
        text_input = inputs["text_input"]
        tab_input  = inputs["tabular_input"]

        #  DistilBert forward
        #    Output is a tuple; the last_hidden_state is first element
        bert_out = self.bert(**text_input)
        cls_embedding = bert_out.last_hidden_state[:, 0, :]  # shape [batch_size, 768]

        # 2) Tabular forward
        tab_out = self.tabular_net(tab_input)  # shape [batch_size, 32]

       #combine
        combined = torch.cat([cls_embedding, tab_out], dim=1)  # [batch_size, 768 + 32] = 800
        output = self.final_regressor(combined)  # [batch_size, 1]
        return output


def create_model(features):

    tabular_input_dim = features["tabular_input"].shape[1]  # e.g. 10 numeric features
    model = MyModel(tabular_input_dim)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    return model, optimizer


if __name__ == '__main__':
    
    dummy_tab = torch.randn(5, 10)
    dummy_text_ids = torch.randint(0, 30522, (5, 12))  # random tokens
    dummy_mask = torch.ones(5, 12)

   
    dummy_features = {
        "text_input": {"input_ids": dummy_text_ids, "attention_mask": dummy_mask},
        "tabular_input": dummy_tab
    }

    model, optimizer = create_model(dummy_features)
    out = model(dummy_features)  # forward pass
    print("Output shape:", out.shape)  # should be [5, 1]
    print(model)
