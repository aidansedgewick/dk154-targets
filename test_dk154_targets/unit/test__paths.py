from dk154_targets import paths

def test__paths_exist():
    assert paths.data_path.exists()
    assert paths.plot_path.exists()
    assert paths.alertDB_path.exists()