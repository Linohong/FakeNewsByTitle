from torch.utils.data import Dataset, DataLoader

class TestDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = self.form_data()

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        sample = self.landmarks_frame[idx]
        return sample

    def form_data(self) :
        data = []
        for i in range(3) :
            data.append({'key1':i, 'key2':i*2})

        return data



dataset = TestDataset()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=32)

for i in dataloader :
    print(i)