

def load_hybrid_model(model_dir):
    """Load the trained hybrid LSTM classifier"""
    config_path = Path(model_dir) / "model_config.json"
    model_path = Path(model_dir) / "hybrid_lstm_classifier.pt"

    with open(config_path, "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridLSTMClassifier(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        dense_size=config["dense_size"],
        num_classes=config["num_classes"],
        dropout=config["dropout"]
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, config, device
