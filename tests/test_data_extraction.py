from GetDataFromScratch.GetTopographyData import GetTopographyData, LoadScratchTestData


def test_data_extraction():
    dataPath = "src/GetDataFromScratch/ScratchData/TestData/"
    coords = LoadScratchTestData(
        fileNameID="SY600_n01", path=dataPath, toLoad="coords")
    h_r, w, h_p = GetTopographyData(coords, lowerBound=1.5, upperBound=2.0)
    print(h_r, w, h_p)
    assert round(h_r, 3) == 0.054
    assert round(w, 3) == 0.376
    assert round(h_p, 3) == 0.025


if __name__ == "__main__":
    test_data_extraction()
