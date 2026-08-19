from __future__ import annotations

from typing import Any

from msentity import MSDataset, Spectrum, load_ms_dataset


def spectrum_to_viewer_value(spectrum: Spectrum) -> dict[str, list[float]]:
    """Convert one spectrum to the custom component's value schema."""
    return {
        "mz": spectrum.mz.astype(float, copy=False).tolist(),
        "intensity": spectrum.intensity.astype(float, copy=False).tolist(),
    }


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def spectrum_view(dataset: MSDataset, index: int | float) -> tuple[int, str, dict[str, list[float]], dict[str, Any]]:
    """Return only the selected spectrum and its metadata for a UI update."""
    if not len(dataset):
        raise ValueError("Cannot display an empty dataset.")

    selected = max(0, min(int(index), len(dataset) - 1))
    record = dataset[selected]
    metadata = {name: _json_value(record[name]) for name in record.columns}
    return (
        selected,
        f"Spectrum {selected + 1} / {len(dataset)}",
        spectrum_to_viewer_value(record.spectrum),
        metadata,
    )


def create_app(dataset: MSDataset) -> Any:
    """Create the Gradio application for an already loaded dataset."""
    import gradio as gr
    from gradio_msentityviewer import MSEntityViewer

    if not len(dataset):
        raise ValueError("Cannot display an empty dataset.")

    initial_index, initial_label, initial_spectrum, initial_metadata = spectrum_view(
        dataset, 0
    )

    with gr.Blocks(title="msentity Spectrum Viewer") as demo:
        gr.Markdown("# msentity Spectrum Viewer")
        with gr.Row():
            previous = gr.Button("Previous")
            index = gr.Number(
                value=initial_index,
                minimum=0,
                maximum=len(dataset) - 1,
                step=1,
                precision=0,
                label="Spectrum index (zero-based)",
            )
            position = gr.Markdown(initial_label)
            next_button = gr.Button("Next")

        viewer = MSEntityViewer(value=initial_spectrum, label="Spectrum")
        metadata = gr.JSON(value=initial_metadata, label="Spectrum metadata")

        def select(requested_index: int | float) -> tuple[int, str, dict[str, list[float]], dict[str, Any]]:
            return spectrum_view(dataset, requested_index)

        def move(current_index: int | float, amount: int) -> tuple[int, str, dict[str, list[float]], dict[str, Any]]:
            return select(int(current_index) + amount)

        outputs = [index, position, viewer, metadata]
        index.change(select, inputs=index, outputs=outputs)
        previous.click(lambda current: move(current, -1), inputs=index, outputs=outputs)
        next_button.click(lambda current: move(current, 1), inputs=index, outputs=outputs)

    return demo


def run_gui(
    input_file: str,
    *,
    file_type: str | None = None,
    spec_id_prefix: str | None = None,
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    inbrowser: bool = True,
) -> None:
    dataset = load_ms_dataset(
        input_file,
        file_type=file_type,
        spec_id_prefix=spec_id_prefix,
    )
    demo = create_app(dataset)
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=inbrowser,
    )
