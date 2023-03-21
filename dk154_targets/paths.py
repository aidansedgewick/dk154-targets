from pathlib import Path

base_path = Path(__file__).absolute().parent.parent


config_path = base_path / "config"


data_path = base_path / "data"
archive_path = data_path / "archive"
alertDB_path = data_path / "alertDB"

outputs_path = base_path / "outputs"
plot_path = outputs_path / "plots"
ranked_target_lists_path = outputs_path / "ranked_target_lists"

targets_of_opportunity_path = base_path / "targets_of_opportunity"
scratch_path = outputs_path / "scratch"
test_path = base_path / "test_dk154_targets"

### data_paths 
alerce_data_path = data_path / "alerce"
atlas_data_path = data_path / "atlas"
fink_data_path = data_path / "fink"
tns_data_path = data_path / "tns"


def create_all_paths():
    data_path.mkdir(exist_ok=True, parents=True)
    alertDB_path.mkdir(exist_ok=True, parents=True)
    archive_path.mkdir(exist_ok=True, parents=True)

    outputs_path.mkdir(exist_ok=True, parents=True)
    plot_path.mkdir(exist_ok=True, parents=True)
    targets_of_opportunity_path.mkdir(exist_ok=True, parents=True)
    scratch_path.mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    print("creating paths...")
    