import wandb
import pandas as pd

def compare_metrics(new_run):
    # Initialize WandB API
    wandb.login()

    # Specify the project and tag for the baseline model
    WANDB_PROJECT = 'titanic_survived' 

    # Fetch the runs for the baseline model
    runs = wandb.Api().runs(WANDB_PROJECT, {"tags": 'baseline'})
    assert len(runs) == 1, 'There must be exactly one run with the tag "baseline"'
    baseline_run = runs[0]
    print(baseline_run)

    # Create an empty DataFrame
    df_compare = pd.DataFrame(columns=["Tag", "Run ID", "train_acc", "val_acc"])

    # Add the baseline model to the DataFrame
    baseline_metrics = baseline_run.summary
    df_compare = df_compare._append({
        "Tag": "Baseline",
        "Run ID": f'<a href="{baseline_run.url}" target="_blank">{baseline_run.id}</a>',
        "train_acc": baseline_metrics["train_acc"],
        "val_acc": baseline_metrics[f"val_acc"]
    }, ignore_index=True)

    # Add newly trained model to the DataFrame
    new_metrics = new_run.summary
    df_compare = df_compare._append({
        "Tag": "Candidate",
        "Run ID": f'<a href="{new_run.url}" target="_blank">{new_run.id}</a>',
        "train_acc": new_metrics["train_acc"],
        "val_acc": new_metrics[f"val_acc"]
    }, ignore_index=True)

    return df_compare #HTML(df_compare.to_html(escape=False))

if __name__ == "__main__":
    # If you want to compare metrics, call the compare_metrics function.
    # Otherwise, train your model using the train function.
    compare_metrics()
