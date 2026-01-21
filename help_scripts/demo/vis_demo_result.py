import gradio as gr

PLY_PATH = "splat.ply"

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
