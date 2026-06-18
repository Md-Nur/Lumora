from torchvision import models
from torch import nn
from transformers import GPT2LMHeadModel
import torch

class MedicalReportGenerator(nn.Module):
    def __init__(self, vocab_size=50257, embed_dim=768):
        super().__init__()
        self.encoder = models.densenet121(weights=None)
        num_ftrs = self.encoder.classifier.in_features
        self.encoder.classifier = nn.Identity()
        self.projector = nn.Linear(num_ftrs, embed_dim)
        self.decoder = GPT2LMHeadModel.from_pretrained("gpt2")
        self.decoder.resize_token_embeddings(vocab_size)

    def forward(self, images, input_ids, attention_mask):
        visual_features = self.encoder(images)
        visual_embeddings = self.projector(visual_features).unsqueeze(1)
        text_embeddings = self.decoder.transformer.wte(input_ids)
        inputs_embeds = torch.cat((visual_embeddings, text_embeddings), dim=1)
        visual_mask = torch.ones((images.size(0), 1), device=images.device)
        extended_mask = torch.cat((visual_mask, attention_mask), dim=1)
        return self.decoder(inputs_embeds=inputs_embeds, attention_mask=extended_mask)
