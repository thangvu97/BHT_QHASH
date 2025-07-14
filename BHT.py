from matplotlib import pyplot as plt
import numpy as np
import random
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCMT, MCXGate # For multi-controlled Toffoli and MCX gates
from qiskit.visualization import plot_histogram
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler

# This function is called by build_unitary_pqc_hash_function_circuit
def quantum_hash_circuit_logic(qc, x_reg, h_reg):
    qc.crx(np.pi,x_reg[0], h_reg[0])
    qc.crx(np.pi,x_reg[1], h_reg[1])
    qc.crx(np.pi,x_reg[2], h_reg[2])
    qc.crx(np.pi,x_reg[3], h_reg[3])

    # --- CNOTs after Layer 1 (on h_reg) ---
    qc.cx(h_reg[3], h_reg[0])
    qc.cx(h_reg[2], h_reg[3])
    qc.cx(h_reg[1], h_reg[2])
    qc.cx(h_reg[0], h_reg[1])

    #Layer 2
    qc.crx(np.pi,x_reg[4], h_reg[0]) 
    qc.crx(np.pi,x_reg[5], h_reg[1]) 
    qc.crx(np.pi,x_reg[6], h_reg[2]) 
    qc.crx(np.pi,x_reg[7], h_reg[3])

    # --- CNOTs after Layer 2 (on h_reg) ---
    qc.cx(h_reg[3], h_reg[2])
    qc.cx(h_reg[0], h_reg[3])
    qc.cx(h_reg[1], h_reg[0])
    qc.cx(h_reg[2], h_reg[1])

# --- Build Unitary PQC Hash Function Circuit ---
# This is the 'U_H' operator for the Grover Oracle
def build_unitary_pqc_hash_function_circuit():
    x_reg = QuantumRegister(8, 'x_input') # 8 input qubits (control qubits)
    h_reg = QuantumRegister(4, 'h_output') # 4 hash output qubits (target qubits)
    qc = QuantumCircuit(x_reg, h_reg, name='U_Hash')

    # Apply the user's quantum hash circuit logic
    quantum_hash_circuit_logic(qc, x_reg, h_reg)

    return qc.to_gate() # Convert the circuit to a reusable gate

# --- Precomputation Phase of BHT ---
def perform_bht_precomputation_phase(num_inputs_K=3, shots=100):
    """
    Performs the precomputation phase of the BHT algorithm.
    Hashes K random inputs and searches for collisions.
    Args:
        num_inputs_K (int): Number of random inputs K to hash.
        shots (int): Number of shots for each hash computation simulation.
    Returns:
        tuple: (list_L, found_collisions)
            list_L (list): Sorted list of (hash_value_tuple, input_string_tuple) pairs.
            found_collisions (list): List of collision tuples (hash_value, input_A, input_B).
    """
    print(f"\n--- Starting BHT Precomputation Phase with K={num_inputs_K} inputs ---")
    list_L = []
    simulator = AerSimulator()

    # Get the unitary hash gate once (removed redundant .to_gate())
    uh_pqc_gate = build_unitary_pqc_hash_function_circuit()

    for i in range(num_inputs_K):
        # Generate a random 8-bit string
        random_input_str = ''.join(random.choice('01') for _ in range(8))

        # Build a circuit to compute the hash for this classical input
        x_reg_temp = QuantumRegister(8, 'x_temp')
        h_reg_temp = QuantumRegister(4, 'h_temp')
        c_reg_temp = ClassicalRegister(4, 'c_temp')
        temp_qc = QuantumCircuit(x_reg_temp, h_reg_temp, c_reg_temp)

        # Initialize x_reg_temp with random_input_str
        for j, bit in reversed(list(enumerate(random_input_str))):
            if bit == '1':
                temp_qc.x(x_reg_temp[j])
        
        # Apply U_H_PQC
        temp_qc.append(uh_pqc_gate, x_reg_temp[:] + h_reg_temp[:])
        temp_qc.measure(h_reg_temp, c_reg_temp) # Measure the hash output

        # Run simulation
        compiled_circuit = transpile(temp_qc, simulator) 
        job = simulator.run(compiled_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts(compiled_circuit)

        # Get the most common outcome (hash output)
        # Qiskit returns bit strings in reverse order (q3q2q1q0)
        most_common_outcome_str = max(counts, key=counts.get)
        
        # Reverse bit string to match [q0, q1, q2, q3] order
        hash_value = tuple(int(b) for b in most_common_outcome_str[::-1])

        list_L.append((hash_value, tuple(int(b) for b in random_input_str))) # Store input_str as tuple for hashable comparison
        
        # Print progress (optional)
        print(f"  Hashed input: {random_input_str} -> Hash: {''.join(str(b) for b in hash_value)}")


    print(f"\nCompleted hashing {num_inputs_K} inputs. Starting sorting and collision search...")

    # Sort List L by hash value
    list_L.sort(key=lambda item: item[0]) # Sort by hash_value

    found_collisions = []
    # Search for collisions in the sorted list
    for i in range(len(list_L) - 1):
        hash_val_curr, input_tup_curr = list_L[i]
        hash_val_next, input_tup_next = list_L[i+1]

        # Collision if hash values are the same AND input strings are different
        if hash_val_curr == hash_val_next and input_tup_curr != input_tup_next:
            # Store the collision pair, normalize order to avoid duplicates
            # and include the common hash value
            collision_entry = (hash_val_curr, tuple(sorted((input_tup_curr, input_tup_next))))

            if collision_entry not in found_collisions: # Ensure no duplicate entries
                found_collisions.append(collision_entry)

    print("\n--- BHT Precomputation Phase Results ---")
    print(f"Total collision pairs found: {len(found_collisions)}")
    if found_collisions:
        print("Collision Pairs (Hash_Value, (Input_A, Input_B)):")
        for idx, (h_val, (in_A, in_B)) in enumerate(found_collisions):
            in_A_str = "".join(str(b) for b in in_A)
            in_B_str = "".join(str(b) for b in in_B)
            h_val_str = "".join(str(b) for b in h_val)
            print(f"  {idx+1}. Hash: {h_val_str} | ({in_A_str}, {in_B_str})")
    else:
        print("No collisions found during precomputation phase.")

    return list_L, found_collisions

# --- Build Grover Oracle for Collision ---
# This oracle marks states |y> where H(y) == H(x_i) AND y != x_i for any x_i in precomputed_L
def check_output(circuit, qreg_q, output_bits):
    if len(output_bits) != 4 or not all(bit in '01' for bit in output_bits):
        raise ValueError("Output must be a 4-bit binary string")
    for i, bit in enumerate((output_bits)):
        if bit == '0':
            circuit.x(qreg_q[i])
    # Apply a multi-controlled Toffoli gate to check if the output matches
    gate = MCXGate(4)
    circuit.append(gate, [8,9,10,11,12])
    for i, bit in enumerate((output_bits)):
        if bit == '0':
            circuit.x(qreg_q[i])

def grover_oracle_for_collision(circuit, x_reg, h_reg,precomputed_L_tuples):
    uh_pqc_gate = build_unitary_pqc_hash_function_circuit()
    circuit.append(uh_pqc_gate, x_reg[:] + h_reg[:])  # Apply the hash function circuit
    # quantum_hash_circuit_logic(circuit, x_reg, h_reg)
    for i in range(len(precomputed_L_tuples)):
        hash_value = precomputed_L_tuples[i]
        circuit.barrier()
        # Check if the output matches the hash value
        check_output(circuit, h_reg, hash_value)
    #Apply reverse quantum hash circuit gate
    circuit.append(uh_pqc_gate.inverse(), x_reg[:] + h_reg[:])  # Apply the inverse hash function circuit

def diffusion_operator(qc, x_reg):
    for i in range(len(x_reg)):
        qc.h(x_reg[i])
    for i in range(len(x_reg)):
        qc.x(x_reg[i])
    qc.h(x_reg[-1])  # Apply H to the last qubit
    # Apply multi-controlled Toffoli gate (all qubits control, last qubit target)
    gate = MCXGate(len(x_reg) - 1)
    qc.append(gate, [0,1,2,3,4,5,6,7])
    qc.h(x_reg[-1])  # Apply H to the last qubit again
    for i in range(len(x_reg)):
        qc.x(x_reg[i])
    for i in range(len(x_reg)):
        qc.h(x_reg[i])

def main(test_num = 5, simulate=True):
    x_reg = QuantumRegister(8, 'x_input')  # 8 input qubits
    h_reg = QuantumRegister(4, 'h_output')  # 4 hash output qubits
    c_reg = ClassicalRegister(8, 'c_output')  # 8 classical output bits
    q_a = QuantumRegister(1, 'q_a')  # Auxiliary qubits for Grover's algorithm
    qc = QuantumCircuit(x_reg, h_reg, q_a, c_reg)
    # Perform BHT precomputation phase
    N = 2**8
    r = 2**4
    num_inputs_K = int((N/r)**(1/3))  # Number of inputs to hash
    
    list_L, found_collisions = perform_bht_precomputation_phase(num_inputs_K, shots=100)
    if found_collisions:
        print("Collisions found during precomputation phase. Exiting...")
        return
    # Extract hash values from list_L
    hash_values = []
    # Each entry in list_L is a tuple (hash_value_tuple, input_string_tuple)
    for hash_value_tuple, _ in list_L:
        hash_str = ''.join(str(bit) for bit in hash_value_tuple)
        hash_values.append(hash_str)
    print("\n--- Starting Grover's Search for Collisions ---")
    #number of iterations for Grover's search
    num_iterations = num_inputs_K  # Optimal number of iterations for Grover's search
    #superposition of qubits 
    for i in range(8):
        qc.h(x_reg[i])  # Apply Hadamard to all input qubits
    # Auxiliary qubit for Grover's search
    qc.x(q_a[0])  # Initialize auxiliary qubit to |1>
    qc.h(q_a[0])  # Apply Hadamard to auxiliary qubit
    
    for _ in range(num_iterations):
        # Apply Grover oracle for collision detection
        qc.barrier()
        grover_oracle_for_collision(qc, x_reg, h_reg, hash_values)
        qc.barrier()
        # Apply diffusion operator
        diffusion_operator(qc, x_reg)

    
    #measure the output qubits
    qc.measure(x_reg, c_reg)  # Measure the hash output qubits into classical bits
    #simulate the circuit
    qc.draw(output='mpl', fold=-1, scale= 0.6)
    plt.show()
    if simulate:
        simulator = AerSimulator()
        job = simulator.run(qc.decompose(reps=3))
        result = job.result()
        print("\n--- Grover's Search Results ---")
        counts = result.get_counts(qc)
        # Print the counts of the measurement results   
        #Sorting the counts for better readability
        counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
                # Plot histogram of results
    else:
        #run on a real backend
        service = QiskitRuntimeService(channel="ibm_quantum", token="") #put your IBM token here
        backend = service.backend("ibm_sherbrooke")
        pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
        transpiled_circuit = pm.run(qc.decompose(reps=3))
        sampler_pub = (transpiled_circuit)
        sampler = Sampler(backend)
        job = sampler.run([sampler_pub], shots=10000)
        print(f"job id: {job.job_id()}")
        print(f">>> Job Status: {job.status()}")
        job_result = job.result()
        print(job_result)


        # Get results from the real device
        sampler_pub_result = job_result[0]
        bit_array = sampler_pub_result.data.c_output

        # Convert BitArray to counts dictionary
        counts = {}
        # The BitArray contains all measurements
        for shot in range(bit_array.num_shots):
            # Get bits for this shot and convert to string
            bits = bit_array.array[shot]
            outcome = format(int(''.join(str(int(b)) for b in bits)), '08b')

            counts[outcome] = counts.get(outcome, 0) + 1
        counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


    plot_histogram(counts, title="Grover's Search Results", bar_labels=True)
    plt.show()

    K = test_num
    print(f"Length of L: {len(list_L)}")
    top_outcomes = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:K+1]
    potential_collisions = []
    
    print("\nAnalyzing top outcomes for collisions:")
    for outcome, count in top_outcomes:
        outcome_bits = tuple(int(b) for b in outcome[::-1])  
        
        found_in_L = False
        for hash_val, input_str in list_L:
            if tuple(int(b) for b in input_str) == outcome_bits:
                found_in_L = True
                break
                
        if not found_in_L:
            print(f"\nTesting outcome: {''.join(str(b) for b in outcome_bits)}")
            
            x_reg_temp = QuantumRegister(8, 'x_temp')
            h_reg_temp = QuantumRegister(4, 'h_temp')
            c_reg_temp = ClassicalRegister(4, 'c_temp')
            temp_qc = QuantumCircuit(x_reg_temp, h_reg_temp, c_reg_temp)
            
            for i, bit in reversed(list(enumerate(outcome_bits))):
                if bit == 1:
                    temp_qc.x(x_reg_temp[i])
            
            uh_pqc_gate = build_unitary_pqc_hash_function_circuit()
            temp_qc.append(uh_pqc_gate, x_reg_temp[:] + h_reg_temp[:])
            temp_qc.measure(h_reg_temp, c_reg_temp)
            
            # Simulate
            simulator = AerSimulator()
            compiled = transpile(temp_qc, simulator)
            result = simulator.run(compiled, shots=100).result()
            hash_counts = result.get_counts(compiled)
            hash_value = tuple(int(b) for b in max(hash_counts, key=hash_counts.get)[::-1])

            print(f"Hash value: {''.join(str(b) for b in hash_value)}")
            
            # Find collisions
            for l_hash, l_input in list_L:
                if l_hash == hash_value and tuple(int(b) for b in l_input) != outcome_bits:
                    collision = (hash_value, (outcome_bits, tuple(int(b) for b in l_input)))
                    potential_collisions.append(collision)
                    print(f"Found collision!")
                    print(f"Input 1: {''.join(str(b) for b in outcome_bits)}")
                    print(f"Input 2: {''.join(str(b) for b in l_input)}")
                    print(f"Hash: {''.join(str(b) for b in hash_value)}")
                    break
    
    print("\n=== Final Collision Results ===")
    if potential_collisions:
        print(f"Found {len(potential_collisions)} collision(s):")
        for idx, (hash_val, (in1, in2)) in enumerate(potential_collisions, 1):
            print(f"\nCollision {idx}:")
            print(f"Hash value: {''.join(str(b) for b in hash_val)}")
            print(f"Input 1: {''.join(str(b) for b in in1)}")
            print(f"Input 2: {''.join(str(b) for b in in2)}")
    else:
        print("No collisions found.")

    
if __name__ == "__main__":
    main(test_num=10, simulate=True)  # Set to False to run on real backend
