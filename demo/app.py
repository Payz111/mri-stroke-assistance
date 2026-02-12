"""Gradio demo app for MRI Stroke Assist."""

from __future__ import annotations


def create_app():
    """Create and return the Gradio app."""
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Install gradio: pip install gradio")

    def predict(dwi_file, adc_file, flair_file):
        """Run inference on uploaded NIfTI files."""
        # TODO: Implement inference pipeline
        return "Inference pipeline not yet implemented", None, None

    with gr.Blocks(title="MRI Stroke Assist") as app:
        gr.Markdown(
            """
            # MRI Stroke Assist
            AI-powered assistant for ischemic stroke detection on brain MRI.

            **Upload DWI, ADC, and FLAIR NIfTI files to get started.**

            > This is a research tool. All results require expert review.
            """
        )

        with gr.Row():
            with gr.Column():
                dwi_input = gr.File(label="DWI (NIfTI)", file_types=[".nii", ".nii.gz"])
                adc_input = gr.File(label="ADC (NIfTI)", file_types=[".nii", ".nii.gz"])
                flair_input = gr.File(label="FLAIR (NIfTI)", file_types=[".nii", ".nii.gz"])
                run_btn = gr.Button("Run Analysis", variant="primary")

            with gr.Column():
                report_output = gr.Textbox(label="Draft Report", lines=15)
                json_output = gr.JSON(label="Structured Findings")
                preview_output = gr.Image(label="Preview")

        run_btn.click(
            fn=predict,
            inputs=[dwi_input, adc_input, flair_input],
            outputs=[report_output, json_output, preview_output],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
