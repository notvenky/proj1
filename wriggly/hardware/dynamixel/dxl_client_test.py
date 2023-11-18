from absl.testing import absltest

from dxl_client import DynamixelClient
from mock_dynamixel_sdk import patch_dynamixel


class DynamixelClientTest(absltest.TestCase):
    """Unit test class for DynamixelClient."""

    @patch_dynamixel(test=[11, 12, 20, 21, 22])
    def test_connect(self, sdk):
        client = DynamixelClient([11, 12, 20, 21, 22], port='test')
        self.assertFalse(client.is_connected)

        client.connect()
        self.assertIn('test', sdk.used_ports)
        self.assertListEqual(sdk.get_enabled_motors('test'), [11, 12, 20, 21, 22])
        client.disconnect()

    @patch_dynamixel(test=[11, 12, 20, 21, 22])
    def test_torque_enabled(self, sdk):
        client = DynamixelClient([11, 12, 20, 21, 22], port='test')
        client.connect()
        self.assertListEqual(sdk.get_enabled_motors('test'), [11, 12, 20, 21, 22])

        client.set_torque_enabled([11, 20], False)
        self.assertListEqual(sdk.get_enabled_motors('test'), [12, 21, 22])

        client.set_torque_enabled([11, 20], True)
        self.assertListEqual(sdk.get_enabled_motors('test'), [11, 12, 20, 21, 22])

        client.disconnect()
        self.assertListEqual(sdk.get_enabled_motors('test'), [])


if __name__ == '__main__':
    absltest.main()