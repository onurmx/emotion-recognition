import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest


class PytorchDatasetTest(unittest.TestCase):
    workdir = "D:/emo/appfiles"

    def test_dataset_ckplus_p1(self):
        from datasets.pytorch import ckplus as ckplus_pytorch
        from utils.pytorch import device_management as dm
        train_dl, _2, _3 = ckplus_pytorch.load_ckplus(
            self.workdir + "/training_datas/ckplus",
            dm.get_default_device(),
            48,
            batch_size=8,
            cfg_OnsuNet=True
        )

        self.assertEqual(type(train_dl).__name__, "DeviceDataLoader")

    def test_dataset_ckplus_p2(self):
        from datasets.pytorch import ckplus as ckplus_pytorch
        from utils.pytorch import device_management as dm
        _1, valid_dl, _3 = ckplus_pytorch.load_ckplus(
            self.workdir + "/training_datas/ckplus",
            dm.get_default_device(),
            48,
            batch_size=8,
            cfg_OnsuNet=True
        )

        self.assertEqual(type(valid_dl).__name__, "DeviceDataLoader")

    def test_dataset_ckplus_p3(self):
        from datasets.pytorch import ckplus as ckplus_pytorch
        from utils.pytorch import device_management as dm
        _1, _2, test_dl = ckplus_pytorch.load_ckplus(
            self.workdir + "/training_datas/ckplus",
            dm.get_default_device(),
            48,
            batch_size=8,
            cfg_OnsuNet=True
        )

        self.assertEqual(type(test_dl).__name__, "DeviceDataLoader")


if __name__ == '__main__':
    unittest.main()
