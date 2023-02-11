import os
import time

import relax
import protein as afprotein


def relax_protein(unrelaxed_protein, output_directory, output_name, model_device):
    amber_relaxer = relax.AmberRelaxation(
        use_gpu=(model_device != "cpu"),
        max_iterations=0,
        tolerance=2.39,
        stiffness=10.0,
        max_outer_iterations=20,
        exclude_residues=[]
        )
    
    t = time.perf_counter()
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default="")
    if "cuda" in model_device:
        device_no = model_device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = device_no
    relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)

    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    relaxation_time = time.perf_counter() - t

    print(f"Relaxation time: {relaxation_time}")
    # Save the relaxed PDB.
    relaxed_output_path = os.path.join(
        output_directory, f'{output_name}_relaxed.pdb'
    )
    with open(relaxed_output_path, 'w') as fp:
        fp.write(relaxed_pdb_str)

    print(f"Relaxed output written to {relaxed_output_path}...")

if __name__ == '__main__':
    with open("./data/2KL8-99.pdb", "r") as f:
        lines = "\n".join(f.readlines())
        protein = afprotein.from_pdb_string(lines)
    relax_protein(
        unrelaxed_protein=protein,
        output_directory="./data",
        output_name="2KL8-99",
        model_device="cuda:0")