import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score

def train_model(model, train_loader, val_loader, epochs=15, lr=1e-3,
                device='cuda', save_path='checkpoint.pt'):
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()  # espera logits [B, C] y target long [B]

    best_val_f1 = 0.0

    print(f"Iniciando entrenamiento por {epochs} epochs en {str(device).upper()}...")
    print("=" * 85)
    print(f"{'Epoch':<10} | {'Train Loss':<15} | {'Val Loss':<15} | {'Val Acc':<12} | {'Val F1 (Macro)'}")
    print("-" * 85)

    for epoch in range(1, epochs + 1):
        # TRAIN
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).long()  # CrossEntropy requiere long

            optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Val
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).long()

                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(all_targets, all_preds, average='macro')
        val_acc = accuracy_score(all_targets, all_preds)

        resumen = (f"{epoch:02d}/{epochs:<7} | {avg_train_loss:<15.4f} | "
                   f"{avg_val_loss:<15.4f} | {val_acc:<12.4f} | {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save(save_path)
            print(resumen + " [NUEVO MEJOR CHECKPOINT]")
        else:
            print(resumen)

    print("-" * 85)
    print(f"Entrenamiento completado. Mejor F1 (macro) en validación: {best_val_f1:.4f}")
    print(f"Modelo guardado en: {save_path}")

    return best_val_f1