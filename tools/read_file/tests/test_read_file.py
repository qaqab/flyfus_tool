from tools.read_file.read_file import ReadFileTool


def test_format_context_numbers_images_and_files_separately() -> None:
    context = ReadFileTool._format_context(
        [
            "https://example.com/a.jpg",
            "https://example.com/b.jpg",
            "https://example.com/c.jpg",
            "https://example.com/data.csv",
        ]
    )

    assert context == (
        "<FLYFUS_CONTEXT>\n"
        'image1: "https://example.com/a.jpg"\n'
        'image2: "https://example.com/b.jpg"\n'
        'image3: "https://example.com/c.jpg"\n'
        'file1: "https://example.com/data.csv"\n'
        "</FLYFUS_CONTEXT>"
    )
