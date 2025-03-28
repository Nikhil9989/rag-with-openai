# Complete RAG Presentation with Speaker Notes

To create the full presentation, you need to combine the following files:

1. `rag-presentation-complete-part1.md`
2. `rag-presentation-complete-part2.md` 
3. `rag-presentation-complete-part2-continued.md`

You can combine them using the following command:

### On Windows (PowerShell)
```powershell
Get-Content rag-presentation-complete-part1.md, rag-presentation-complete-part2.md, rag-presentation-complete-part2-continued.md | Set-Content rag-presentation-complete.md
```

### On macOS/Linux
```bash
cat rag-presentation-complete-part1.md rag-presentation-complete-part2.md rag-presentation-complete-part2-continued.md > rag-presentation-complete.md
```

The resulting file (`rag-presentation-complete.md`) will be a complete Marp presentation with:
- All presentation slides
- Detailed speaker notes for each slide
- Your name and company on the final slide

## Converting to PowerPoint

Once you have the complete file, follow the instructions in `CONVERSION_GUIDE.md` to convert it to PowerPoint format.

The simplest method is to use Marp CLI:

```bash
marp rag-presentation-complete.md --allow-local-files --output rag-presentation.pptx
```

Or use VS Code with the Marp extension for a visual preview and export.

## Features of This Presentation

- **Comprehensive Content**: Covers the entire RAG workflow from fundamentals to implementation
- **Speaker Notes**: Detailed talking points for each slide
- **Code Examples**: Practical implementation steps with real Python code
- **Visual Architecture**: Clear diagrams of the RAG system components
- **Best Practices**: Performance optimization, quality enhancement, and responsible AI
- **NetApp Context**: Use cases and applications specific to NetApp

## Troubleshooting

If you have issues with Marp CLI:

1. Make sure you have Node.js installed
2. Install Marp CLI globally: `npm install -g @marp-team/marp-cli`
3. If you're encountering permission issues on macOS/Linux, try: `sudo npm install -g @marp-team/marp-cli`
4. Ensure the paths to the files are correct

Alternatively, the VS Code extension method typically has fewer issues:
1. Install VS Code: https://code.visualstudio.com/
2. Install the Marp extension: search for "Marp" in the Extensions view
3. Open the complete presentation file
4. Use the "Export slide deck..." option in the Marp menu
