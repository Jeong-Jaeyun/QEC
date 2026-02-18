import unittest
import importlib.util

_QISKIT_AVAILABLE = importlib.util.find_spec("qiskit") is not None

if _QISKIT_AVAILABLE:
    from qiskit import QuantumCircuit
    from scenarios.nonlocal_buffer import NonLocalBufferParams, make_nonlocal_buffer_hook


def _qubits_as_indices(qc, qargs) -> list:
    return [qc.find_bit(q).index for q in qargs]


@unittest.skipUnless(_QISKIT_AVAILABLE, "qiskit is not installed")
class TestNonLocalBuffer(unittest.TestCase):
    def test_zx_mode_uses_configured_endpoints(self):
        p = NonLocalBufferParams(
            enabled=True,
            bridge_mode="zx",
            bridge_strength=1,
            extra_anc_ticks=0,
            d_src=2,
            d_dst=0,
        )
        qc = QuantumCircuit(4)
        hook = make_nonlocal_buffer_hook(p)
        hook(qc, 0)

        self.assertEqual(len(qc.data), 2)
        op0 = qc.data[0]
        op1 = qc.data[1]
        self.assertEqual(op0.operation.name, "cx")
        self.assertEqual(_qubits_as_indices(qc, op0.qubits), [2, 3])
        self.assertEqual(op1.operation.name, "cz")
        self.assertEqual(_qubits_as_indices(qc, op1.qubits), [3, 0])

    def test_xx_mode_uses_configured_endpoints(self):
        p = NonLocalBufferParams(
            enabled=True,
            bridge_mode="xx",
            bridge_strength=1,
            extra_anc_ticks=0,
            d_src=1,
            d_dst=2,
        )
        qc = QuantumCircuit(4)
        hook = make_nonlocal_buffer_hook(p)
        hook(qc, 0)

        self.assertEqual(len(qc.data), 2)
        op0 = qc.data[0]
        op1 = qc.data[1]
        self.assertEqual(op0.operation.name, "cx")
        self.assertEqual(_qubits_as_indices(qc, op0.qubits), [1, 3])
        self.assertEqual(op1.operation.name, "cx")
        self.assertEqual(_qubits_as_indices(qc, op1.qubits), [3, 2])

    def test_ham_bus_mode_uses_configured_endpoints(self):
        p = NonLocalBufferParams(
            enabled=True,
            bridge_mode="ham_bus",
            bridge_strength=1,
            extra_anc_ticks=0,
            d_src=0,
            d_dst=2,
        )
        qc = QuantumCircuit(4)
        hook = make_nonlocal_buffer_hook(p)
        hook(qc, 0)

        self.assertEqual(len(qc.data), 1)
        op = qc.data[0]
        self.assertEqual(_qubits_as_indices(qc, op.qubits), [0, 3, 2])

    def test_invalid_same_endpoint_raises(self):
        with self.assertRaises(ValueError):
            make_nonlocal_buffer_hook(NonLocalBufferParams(d_src=1, d_dst=1))


if __name__ == "__main__":
    unittest.main()
