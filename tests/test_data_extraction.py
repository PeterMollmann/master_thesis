from GetDataFromScratch.GetTopographyData import GetTopographyData, LoadScratchTestData


def test_data_extraction():
    dataPath = "src/GetDataFromScratch/ScratchData/MaterialSweep/"
    coords = LoadScratchTestData(
        fileNameID="SY600_n01_d0050_E200000_mu00_rho78_Poisson03",
        path=dataPath,
        toLoad="coords",
    )
    h_r, w, h_p = GetTopographyData(coords, lowerBound=2.00, upperBound=2.44)
    print(h_r, w, h_p)
    assert round(h_r, 3) == 0.042
    assert round(w, 3) == 0.374
    assert round(h_p, 3) == 0.025


if __name__ == "__main__":
    test_data_extraction()
