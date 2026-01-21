import gradio as gr

PLY_PATH = "/home/robi/projects/real2sim/MV-SAM3D/demo/demo_visualization/drug/medicine_box/drug_medicine_box_10v_s1a30_s2v30t4_20260121_214310/result.ply"

def interactive_visualizer(ply_path):
    with gr.Blocks() as demo:
        gr.Markdown("# 3D Gaussian Splatting (black-screen loading might take a while)")
        gr.Model3D(
            value=ply_path,  # splat file
            label="3D Scene",
        )
    demo.launch(share=True)

if __name__ == "__main__":
    interactive_visualizer(PLY_PATH)
