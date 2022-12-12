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

    def test_dataset_fer2013_p1(self):
        from datasets.pytorch import fer2013 as fer2013_pytorch
        from utils.pytorch import device_management as dm
        train_dl, _2, _3 = fer2013_pytorch.load_fer2013(
            self.workdir + "/training_datas/fer2013/fer2013.csv",
            dm.get_default_device(),
            48,
            batch_size=8,
            cfg_OnsuNet=True
        )

        self.assertEqual(type(train_dl).__name__, "DeviceDataLoader")

    def test_dataset_fer2013_p2(self):
        from datasets.pytorch import fer2013 as fer2013_pytorch
        from utils.pytorch import device_management as dm
        _1, valid_dl, _3 = fer2013_pytorch.load_fer2013(
            self.workdir + "/training_datas/fer2013/fer2013.csv",
            dm.get_default_device(),
            48,
            batch_size=8,
            cfg_OnsuNet=True
        )

        self.assertEqual(type(valid_dl).__name__, "DeviceDataLoader")

    def test_dataset_fer2013_p3(self):
        from datasets.pytorch import fer2013 as fer2013_pytorch
        from utils.pytorch import device_management as dm
        _1, _2, test_dl = fer2013_pytorch.load_fer2013(
            self.workdir + "/training_datas/fer2013/fer2013.csv",
            dm.get_default_device(),
            48,
            batch_size=8,
            cfg_OnsuNet=True
        )

        self.assertEqual(type(test_dl).__name__, "DeviceDataLoader")

    def test_dataset_kdef_p1(self):
        from datasets.pytorch import kdef as kdef_pytorch
        from utils.pytorch import device_management as dm
        train_dl, _2, _3 = kdef_pytorch.load_kdef(
            self.workdir + "/training_datas/kdef",
            dm.get_default_device(),
            48,
            batch_size=8,
            cfg_OnsuNet=True
        )

        self.assertEqual(type(train_dl).__name__, "DeviceDataLoader")

    def test_dataset_kdef_p2(self):
        from datasets.pytorch import kdef as kdef_pytorch
        from utils.pytorch import device_management as dm
        _1, valid_dl, _3 = kdef_pytorch.load_kdef(
            self.workdir + "/training_datas/kdef",
            dm.get_default_device(),
            48,
            batch_size=8,
            cfg_OnsuNet=True
        )

        self.assertEqual(type(valid_dl).__name__, "DeviceDataLoader")

    def test_dataset_kdef_p3(self):
        from datasets.pytorch import kdef as kdef_pytorch
        from utils.pytorch import device_management as dm
        _1, _2, test_dl = kdef_pytorch.load_kdef(
            self.workdir + "/training_datas/kdef",
            dm.get_default_device(),
            48,
            batch_size=8,
            cfg_OnsuNet=True
        )

        self.assertEqual(type(test_dl).__name__, "DeviceDataLoader")

class TensorflowDatasetTest(unittest.TestCase):
    workdir = "D:/emo/appfiles"

    def test_dataset_ckplus_p1(self):
        from datasets.tensorflow import ckplus as ckplus_tensorflow
        training_data, _2, _3, _4, _5, _6 = ckplus_tensorflow.load_ckplus(
            self.workdir + "/training_datas/ckplus",
            48,
            batch_size=8,
            cfg_OnsuNet = True
        )

        self.assertEqual(type(training_data).__name__, "NumpyArrayIterator")

    def test_dataset_ckplus_p2(self):
        from datasets.tensorflow import ckplus as ckplus_tensorflow
        _1, validation_data, _3, _4, _5, _6 = ckplus_tensorflow.load_ckplus(
            self.workdir + "/training_datas/ckplus",
            48,
            batch_size=8,
            cfg_OnsuNet = True
        )

        self.assertEqual(type(validation_data).__name__, "NumpyArrayIterator")

    def test_dataset_ckplus_p3(self):
        from datasets.tensorflow import ckplus as ckplus_tensorflow
        _1, _2, testing_data, _4, _5, _6 = ckplus_tensorflow.load_ckplus(
            self.workdir + "/training_datas/ckplus",
            48,
            batch_size=8,
            cfg_OnsuNet = True
        )

        self.assertEqual(type(testing_data).__name__, "NumpyArrayIterator")

    def test_dataset_fer2013_p1(self):
        from datasets.tensorflow import fer2013 as fer2013_tensorflow
        training_data, _2, _3, _4, _5, _6 = fer2013_tensorflow.load_fer2013(
            self.workdir + "/training_datas/fer2013/fer2013.csv",
            48,
            batch_size=8,
            cfg_OnsuNet = True
        )

        self.assertEqual(type(training_data).__name__, "NumpyArrayIterator")

    def test_dataset_fer2013_p2(self):
        from datasets.tensorflow import fer2013 as fer2013_tensorflow
        _1, validation_data, _3, _4, _5, _6 = fer2013_tensorflow.load_fer2013(
            self.workdir + "/training_datas/fer2013/fer2013.csv",
            48,
            batch_size=8,
            cfg_OnsuNet = True
        )

        self.assertEqual(type(validation_data).__name__, "NumpyArrayIterator")

    def test_dataset_fer2013_p3(self):
        from datasets.tensorflow import fer2013 as fer2013_tensorflow
        _1, _2, testing_data, _4, _5, _6 = fer2013_tensorflow.load_fer2013(
            self.workdir + "/training_datas/fer2013/fer2013.csv",
            48,
            batch_size=8,
            cfg_OnsuNet = True
        )

        self.assertEqual(type(testing_data).__name__, "NumpyArrayIterator")

    def test_dataset_kdef_p1(self):
        from datasets.tensorflow import kdef as kdef_tensorflow
        training_data, _2, _3, _4, _5, _6 = kdef_tensorflow.load_kdef(
            self.workdir + "/training_datas/kdef",
            48,
            batch_size=8,
            cfg_OnsuNet = True
        )

        self.assertEqual(type(training_data).__name__, "NumpyArrayIterator")

    def test_dataset_kdef_p2(self):
        from datasets.tensorflow import kdef as kdef_tensorflow
        _1, validation_data, _3, _4, _5, _6 = kdef_tensorflow.load_kdef(
            self.workdir + "/training_datas/kdef",
            48,
            batch_size=8,
            cfg_OnsuNet = True
        )

        self.assertEqual(type(validation_data).__name__, "NumpyArrayIterator")

    def test_dataset_kdef_p3(self):
        from datasets.tensorflow import kdef as kdef_tensorflow
        _1, _2, testing_data, _4, _5, _6 = kdef_tensorflow.load_kdef(
            self.workdir + "/training_datas/kdef",
            48,
            batch_size=8,
            cfg_OnsuNet = True
        )

        self.assertEqual(type(testing_data).__name__, "NumpyArrayIterator")


if __name__ == '__main__':
    unittest.main()
