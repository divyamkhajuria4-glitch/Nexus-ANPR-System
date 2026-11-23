import unittest

from utils.device import DeviceContext, select_device


class _NS:
    def __init__(self, **kw): self.__dict__.update(kw)


def fake_torch(cuda=False, mps=False):
    return _NS(cuda=_NS(is_available=lambda: cuda),
               backends=_NS(mps=_NS(is_available=lambda: mps, is_built=lambda: mps)))


class DeviceSelectionTests(unittest.TestCase):
    def test_cpu_always_cpu(self):
        self.assertEqual(select_device("cpu", fake_torch(cuda=True, mps=True)), "cpu")

    def test_auto_prefers_cuda_then_mps_then_cpu(self):
        self.assertEqual(select_device("auto", fake_torch(cuda=True, mps=True)), "cuda")
        self.assertEqual(select_device("auto", fake_torch(mps=True)), "mps")
        self.assertEqual(select_device("auto", fake_torch()), "cpu")

    def test_requested_but_unavailable_falls_back_to_cpu(self):
        self.assertEqual(select_device("mps", fake_torch()), "cpu")
        self.assertEqual(select_device("cuda", fake_torch(mps=True)), "cpu")

    def test_requested_and_available(self):
        self.assertEqual(select_device("mps", fake_torch(mps=True)), "mps")

    def test_no_torch_falls_back_to_cpu(self):
        self.assertEqual(select_device("auto", None), "cpu")
        self.assertEqual(select_device("mps", None), "cpu")

    def test_invalid_name_rejected(self):
        with self.assertRaises(ValueError):
            select_device("tpu", fake_torch())

    def test_context_retries_once_on_cpu(self):
        ctx, seen = DeviceContext("mps"), []

        def fn(device):
            seen.append(device)
            if device == "mps":
                raise RuntimeError("unsupported op")
            return "ok"

        self.assertEqual(ctx.run(fn), "ok")
        self.assertEqual(seen, ["mps", "cpu"])
        self.assertTrue(ctx.fell_back)
        self.assertEqual(ctx.device, "cpu")

    def test_context_does_not_swallow_cpu_errors(self):
        def boom(device): raise RuntimeError("real bug")
        with self.assertRaises(RuntimeError):
            DeviceContext("cpu").run(boom)


if __name__ == "__main__":
    unittest.main()
