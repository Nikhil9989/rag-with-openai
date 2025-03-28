# Converting Markdown to PowerPoint

This guide explains how to convert the Markdown presentation files in this repository to PowerPoint format.

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

## Option 2: Using Pandoc

[Pandoc](https://pandoc.org/) is a universal document converter that can convert Markdown to PPTX.

### Installation

- **Windows**: Download and install from [pandoc.org](https://pandoc.org/installing.html)
- **macOS**: `brew install pandoc`
- **Linux**: `sudo apt-get install pandoc`

### Conversion to PPTX

```bash
# Navigate to the presentation directory
cd presentation

# Combine both parts of the presentation (if necessary)
cat rag-presentation.md rag-presentation-part2.md > full-presentation.md

# Convert to PPTX
pandoc full-presentation.md -o rag-presentation.pptx
```

Note: Pandoc won't preserve all the styling from the Marp format, but it will create a functional PPTX.

## Option 3: Using VSCode with Marp Extension

1. Install [Visual Studio Code](https://code.visualstudio.com/)
2. Install the [Marp for VS Code extension](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode)
3. Open the repository in VSCode
4. Open `rag-presentation.md` in VSCode
5. Click the "Marp: Export Slide Deck..." button in the top-right corner
6. Select "PowerPoint (.pptx)" as the export format

## Option 4: Online Conversion Services

If you prefer not to install software:

1. Use online services like [Slidev](https://sli.dev/) or [Markdown to PowerPoint Converter](https://products.aspose.app/slides/conversion/markdown-to-powerpoint)
2. Copy the content from the markdown files and paste into these services
3. Download the PPTX output

## Need to Combine Files First?

If you need to combine the presentation files before conversion:

```bash
# Navigate to the presentation directory
cd presentation

# For Unix/Linux/MacOS
cat rag-presentation.md rag-presentation-part2.md > complete-presentation.md

# For Windows PowerShell
Get-Content rag-presentation.md, rag-presentation-part2.md | Set-Content complete-presentation.md
```

Then convert the combined file using any of the methods above.
