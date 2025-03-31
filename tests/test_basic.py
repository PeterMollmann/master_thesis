from GetDataFromScratch.GetTopographyData import GetTopographyData
from GetDataFromScratch.DataLoader import DataLoader


def test_example():
    # assert 1 + 1 == 2
    coords = DataLoader(fileNameID="SY600_n01", toLoad="coords")
    x, y, z = coords
    h_r, w, h_p = GetTopographyData(x, y, z, lowerBound=1.5, upperBound=2.0)
    print(h_r, w, h_p)
