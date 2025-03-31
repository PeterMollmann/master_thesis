from GetDataFromScratch.GetTopographyData import GetTopographyData
from GetDataFromScratch.DataLoader import DataLoader


def test_example():
    coords = DataLoader(fileNameID="SY600_n01", toLoad="coords")
    # print(coords)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    h_r, w, h_p = GetTopographyData(x, y, z, lowerBound=1.5, upperBound=2.0)
    # print(h_r, w, h_p)
    assert h_r < 0.5
    assert w < 0.5
    assert h_p < 0.5


if __name__ == "__main__":
    test_example()
