import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
def build_unitary_quantum_hash_circuit(x_reg, h_reg):
    """
    Xây dựng mạch lượng tử đơn nhất U_H cho hàm băm dựa trên image_96d577.png.
    U_H: |x_reg>|h_reg (khởi tạo 0)> -> |x_reg>|H(x)>.
    x_reg: 8 qubit đầu vào (qubit điều khiển)
    h_reg: 4 qubit đầu ra băm (qubit đích)
    Đầu ra: QuantumCircuit (subroutine)
    """
    qc = QuantumCircuit(x_reg, h_reg, name='U_H')
    # --- Layer 1 ---
    
    qc.crx(np.pi,x_reg[0], h_reg[0])
    qc.crx(np.pi,x_reg[1], h_reg[1])
    qc.crx(np.pi,x_reg[2], h_reg[2])
    qc.crx(np.pi,x_reg[3], h_reg[3])
    qc.barrier()  # Thêm barrier để phân tách các lớp
    # --- CNOTs sau Layer 1 ---
    qc.cx(h_reg[3], h_reg[0])
    qc.cx(h_reg[2], h_reg[3])
    qc.cx(h_reg[1], h_reg[2])
    qc.cx(h_reg[0], h_reg[1])
    qc.barrier()  # Thêm barrier để phân tách các lớp
        # --- Layer 2 ---
    qc.crx(np.pi,x_reg[4], h_reg[0])
    qc.crx(np.pi,x_reg[5], h_reg[1])
    qc.crx(np.pi,x_reg[6], h_reg[2])
    qc.crx(np.pi,x_reg[7], h_reg[3])

    qc.barrier()  # Thêm barrier để phân tách các lớp
    # --- CNOTs sau Layer 2 ---
    qc.cx(h_reg[3], h_reg[2])
    qc.cx(h_reg[0], h_reg[3])
    qc.cx(h_reg[1], h_reg[0])
    qc.cx(h_reg[2], h_reg[1])
    qc.barrier()  # Thêm barrier để phân tách các lớp
    return qc

# --- MAIN EXECUTION ---
def main():
    print("\n=== Quantum Hash Circuit Generator ===")
    user_input = input("Nhập chuỗi 8 bit (ví dụ: 01101001): ").strip()
    if len(user_input) != 8 or any(c not in '01' for c in user_input):
        print("Lỗi: Vui lòng nhập đúng 8 ký tự '0' hoặc '1'.")
        return
    x_reg = QuantumRegister(8, 'x_input')
    h_reg = QuantumRegister(4, 'h_output')
    c_reg = ClassicalRegister(4, 'c_measure')
    qc = QuantumCircuit(x_reg, h_reg, c_reg)
    # Khởi tạo các qubit x_input theo input
    for i, bit in reversed(list(enumerate(user_input))):
        if bit == '1':
            qc.x(x_reg[i])
    # Thêm mạch hash
    hash_circuit = build_unitary_quantum_hash_circuit(x_reg, h_reg)
    # Kết hợp mạch hash vào mạch chính
    qc.compose(hash_circuit, inplace=True)
    # Đo các qubit h_output
    qc.measure(h_reg, c_reg)
    qc.draw('mpl', reverse_bits=False)
    plt.show()
    # Mô phỏng kết quả
    simulator = AerSimulator()
    compiled = transpile(qc, simulator)
    job = simulator.run(compiled, shots=1024)
    result = job.result()
    counts = result.get_counts(compiled)
    print("\nKết quả đo hash lượng tử:")
    for k, v in counts.items():
        print(f"Hash output: {k[::-1]} (số lần: {v})")

if __name__ == "__main__":
    main()
