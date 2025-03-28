# Converting Markdown to PowerPoint

This guide explains how to convert the Markdown presentation files in this repository to PowerPoint format.

## Using the Presentation File

The complete presentation is now available as a single file:
- `rag-presentation.md`: Complete RAG presentation ready for conversion

You no longer need to combine multiple files - the presentation is complete and ready to convert directly.

## Option 1: Using Marp (Recommended)

[Marp](https://marp.app/) is the tool these slides were designed for. It can export to PPTX directly.

### Installation

```bash
# Install Marp CLI globally
npm install -g @marp-team/marp-cli
```

### Conversion to PPTX

```bash
# Navigate to the presentation directory
cd presentation

# Convert to PPTX
marp rag-presentation.md --output rag-presentation.pptx
```

If you encounter any issues, you can try these alternatives:

```bash
# Alternative 1: Use absolute path
marp C:\Users\malle\rag-with-openai\presentation\rag-presentation.md --output rag-presentation.pptx

# Alternative 2: Run from parent directory
cd rag-with-openai
marp presentation/rag-presentation.md --output presentation/rag-presentation.pptx
```

## Option 2: Using VSCode with Marp Extension

This is often the easiest approach:

1. Install [Visual Studio Code](https://code.visualstudio.com/)
2. Install the [Marp for VS Code extension](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode)
3. Open the repository in VSCode
4. Open `rag-presentation.md` in VSCode
5. You should see a preview of the slides in the right panel
6. Click the "Marp: Export Slide Deck..." button in the top-right corner
7. Select "PowerPoint (.pptx)" as the export format

## Option 3: Using Pandoc

[Pandoc](https://pandoc.org/) is a universal document converter that can convert Markdown to PPTX.

### Installation

- **Windows**: Download and install from [pandoc.org](https://pandoc.org/installing.html)
- **macOS**: `brew install pandoc`
- **Linux**: `sudo apt-get install pandoc`

### Conversion to PPTX

```bash
# Navigate to the presentation directory
cd presentation

# Convert to PPTX
pandoc rag-presentation.md -o rag-presentation.pptx
```

Note: Pandoc won't preserve all the styling from the Marp format, but it will create a functional PPTX.

## Option 4: Online Conversion Services

If you prefer not to install software:

1. Use online services like [Slidev](https://sli.dev/) or [Markdown to PowerPoint Converter](https://products.aspose.app/slides/conversion/markdown-to-powerpoint)
2. Copy the content from `rag-presentation.md` and paste into these services
3. Download the PPTX output

## Troubleshooting

If you encounter issues with Marp:

1. **Check that the file exists**: Verify that `rag-presentation.md` is in the current directory
2. **Check Marp installation**: Run `marp --version` to verify it's installed properly
3. **Try VSCode extension**: The VSCode extension often works when the CLI has issues
4. **Check for special characters**: Some special characters might cause issues with Marp
5. **Check the file encoding**: Ensure the file is saved with UTF-8 encoding

## Need Help?

If you continue to have issues, please:
1. Open an issue on the GitHub repository
2. Try one of the alternative methods listed above
