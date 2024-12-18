import torch 

def generate_pred_for_kaggle(model, loader):
    model.eval()
    preds = []
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        preds.append(pred)
    return torch.cat(preds)

def generate_kaggle_file(preds, output_file="kaggle.csv"):
    with open(output_file, "w") as f:
        f.write("ID, TARGET\n")
        for i, pred in enumerate(preds):
            f.write(f"{i},{pred.item()}\n")
    