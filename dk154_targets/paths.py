from pathlib import Path

base_path = Path(__file__).absolute().parent.parent

config_path = base_path / "config"

alertDB_path = base_path / "alertDB"
data_path = base_path / "data"
plot_path = base_path / "plots"
targets_of_opportunity_path = base_path / "targets_of_opportunity"

### data_paths 
atlas_data_path = data_path / "atlas"


def create_all_paths():
    alertDB_path.mkdir(exist_ok=True, parents=True)
    data_path.mkdir(exist_ok=True, parents=True)
    plot_path.mkdir(exist_ok=True, parents=True)
    targets_of_opportunity_path.mkdir(exist_ok=True, parents=True)

    ### data paths
    atlas_data_path.mkdir(exist_ok=True, parents=True)