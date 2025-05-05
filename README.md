<div align="center">
  <img src="https://github.com/The-Dino-Stack/Chatosaurus/blob/b90cf6181d38c9aa4693277a90d77743fe2dfaac/assets/logo.png" alt="Plugin Logo" width="350" />

  <p>
    Chatosaurus is an AI assistant plugin for Docusaurus that brings your docs to life with conversational search. Powered by retrieval-augmented generation, it lets users interact with your documentation like they're chatting with a brainy, violet-scaled dev dino.

   **This is the embedding project for [Chatosaurus](https://github.com/The-Dino-Stack/Chatosaurus/)**
  </p>
</div>

---

## ✨ Features

- Embedding .md and .mdx files.
- Powered by several AI models.
- Custom input and output paths.
- Advanced logging.

---

## 🚀 Usage

To generate embeddings from markdown files, use the following command:

```bash
python main.py --provider openai --model <MODEL_NAME> --api-key <API_KEY> --input-path <INPUT_PATH> --output-path <OUTPUT_PATH>
```

### Example

```bash
python main.py --provider openai --model text-embedding-ada-002 --api-key sk-abc123 --input-path ./docs --output-path embeddings.json
```

### Arguments

| Argument         | Description                                                                                     | Required |
|------------------|-------------------------------------------------------------------------------------------------|----------|
| `--provider`     | The embedding provider to use (e.g., `openai`).                                                | Yes      |
| `--model`        | The model name to use for the embedding provider.                                              | Yes      |
| `--api-key`      | The API key for the embedding provider (required for OpenAI).                                  | Yes      |
| `--input-path`   | Path to the folder containing markdown files.                                                  | Yes      |
| `--output-path`  | Path to save the generated embeddings JSON file (default: `embeddings.json`).                  | No       |
| `--verbose` or `-v` | Enable verbose logging for debugging purposes.                                              | No       |

### Supported providers

Here is a list of supported providers. To see the full list of models, check the provider's documentation.

- [OpenAI](https://platform.openai.com/docs/guides/embeddings#embedding-models)

---

## 🤝 Contributing

Pull requests, feature ideas, and bug reports are welcome!

---

## 📝 License

MIT

---

> Built with ❤️ to make docs smarter and more human.