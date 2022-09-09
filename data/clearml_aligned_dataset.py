from data.base_dataset import BaseDataset, get_params, get_transform
from allegroai import DataView
from PIL import Image


class ClearmlAlignedDataset(BaseDataset):
    def initialize(self, opt):
        assert opt.label_nc == 0
        assert opt.no_instance
        assert not opt.load_features

        self.opt = opt
        self.root = opt.dataroot

        self.input_frame = opt.input_frame
        self.output_frame = opt.output_frame

        dataview = DataView()
        dataview.add_query(dataset_name=opt.clearml_name, version_name=opt.clearml_version)
        dataview.prefetch_files()
        self.data = dataview.to_list()

    def __getitem__(self, index):
        A_path = self.data[index][self.input_frame].get_local_source(raise_on_error=True)
        B_path = self.data[index][self.output_frame].get_local_source(raise_on_error=True)
        A = Image.open(A_path)
        B = Image.open(B_path)
        params = get_params(self.opt, A.size)
        transform_A = get_transform(self.opt, params)
        A_tensor = transform_A(A.convert('RGB'))
        transform_B = get_transform(self.opt, params)
        B_tensor = transform_B(B.convert('RGB'))
        return {'label': A_tensor, 'inst': 0, 'image': B_tensor,  'feat': 0, 'path': A_path}

    def __len__(self):
        return len(self.data)

    def name(self):
        return 'ClearmlAlignedDataset'
