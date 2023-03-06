import glob


class DataLoader:
    def __init__(self, dir, extension='jpg') -> None:
        assert extension in ['jpg', 'JPG'], 'File extension must be "jpg"!'
        self.data = self._get_img_files(dir, extension)

    def _get_img_files(self, dir, extension):
        fpath = f'{dir}/*.{extension}'
        img_ls = glob.glob(fpath)
        if len(img_ls) == 0:
            fpath = f'{dir}/*.{extension}'
            img_ls = glob.glob(fpath)
        img_ls = list(map(lambda x: x.split('/')[-1], img_ls))
        return img_ls

    '''
    tf.data.Dataset.skip() & .take()로 처리 다 된 상태에서 train과 val로 나누는 것이 더 효율적

    def train_test_split(self, test_size=0.2, random_state=706):
        self.data = model_selection.train_test_split(self.data, test_size=test_size, random_state=random_state)
    '''
    def get_data(self):
        return self.data